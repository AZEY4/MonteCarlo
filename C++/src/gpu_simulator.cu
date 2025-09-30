#include "gpu_simulator.cuh"
#include "option.hpp"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdio>

// This file implements two kernels:
//  - mc_kernel_european: pathless European payoff.
//  - mc_kernel_asian: path-based Asian option (discrete arithmetic average).

// European kernel (pathless)
__global__ void mc_kernel_european(Option opt, std::size_t num_paths, double *d_payoffs, unsigned long long seed) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    // initialize RNG per-thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double z = curand_normal_double(&state);
    double ST = opt.S0 * exp((opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T + opt.sigma * sqrt(opt.T) * z);
    double payoff = 0.0;
    if (opt.type == OptionType::EuropeanCall) payoff = fmax(ST - opt.K, 0.0);
    else if (opt.type == OptionType::EuropeanPut) payoff = fmax(opt.K - ST, 0.0);
    d_payoffs[idx] = payoff;
}

// Asian kernel (discrete arithmetic average)
__global__ void mc_kernel_asian(Option opt, std::size_t num_paths, int steps, double *d_payoffs, unsigned long long seed) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    curandStatePhilox4_32_10_t state;
    // Use idx as subsequence to ensure different sequences per thread
    curand_init(seed, idx, 0, &state);

    double dt = opt.T / static_cast<double>(steps);
    double drift_dt = (opt.r - 0.5 * opt.sigma * opt.sigma) * dt;
    double vol_sqrt_dt = opt.sigma * sqrt(dt);

    double S = opt.S0;
    double avg = 0.0;
    // simulate discrete path
    for (int t = 0; t < steps; ++t) {
        double z = curand_normal_double(&state);
        S *= exp(drift_dt + vol_sqrt_dt * z);
        avg += S;
    }
    avg /= static_cast<double>(steps);

    double payoff = 0.0;
    if (opt.type == OptionType::AsianCall) payoff = fmax(avg - opt.K, 0.0);
    else if (opt.type == OptionType::AsianPut) payoff = fmax(opt.K - avg, 0.0);

    d_payoffs[idx] = payoff;
}

// Host wrapper
extern "C" void monte_carlo_gpu_c(const Option *opt_ptr, std::size_t num_paths, double *out_price, double *out_stderr) {
    if (!opt_ptr || !out_price || !out_stderr) return;
    Option opt = *opt_ptr; // copy to local

    // basic validation
    if (num_paths == 0) {
        *out_price = 0.0;
        *out_stderr = 0.0;
        return;
    }

    // allocate device memory
    double *d_payoffs = nullptr;
    size_t bytes = num_paths * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&d_payoffs, bytes);
    if (err != cudaSuccess) {
        // allocation failed
        *out_price = NAN; *out_stderr = NAN;
        return;
    }

    const int block = 256;
    int grid = static_cast<int>((num_paths + block - 1) / block);
    unsigned long long seed = 123456789ULL;

    // Dispatch kernel based on option type
    if (opt.type == OptionType::EuropeanCall || opt.type == OptionType::EuropeanPut) {
        mc_kernel_european<<<grid, block>>>(opt, num_paths, d_payoffs, seed);
    } else {
        // Asian option: choose steps.
        const int steps = 100;
        mc_kernel_asian<<<grid, block>>>(opt, num_paths, steps, d_payoffs, seed);
    }

    // sync and check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_payoffs);
        *out_price = NAN; *out_stderr = NAN;
        return;
    }

    // copy back to host
    std::vector<double> h_payoffs;
    try {
        h_payoffs.resize(num_paths);
    } catch (...) {
        cudaFree(d_payoffs);
        *out_price = NAN; *out_stderr = NAN;
        return;
    }
    err = cudaMemcpy(h_payoffs.data(), d_payoffs, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_payoffs);
    if (err != cudaSuccess) {
        *out_price = NAN; *out_stderr = NAN;
        return;
    }

    // host-side reduction to compute mean and stderr
    double sum = 0.0;
    for (std::size_t i = 0; i < num_paths; ++i) sum += h_payoffs[i];
    double mean = sum / static_cast<double>(num_paths);

    double sq_sum = 0.0;
    for (std::size_t i = 0; i < num_paths; ++i) {
        double d = h_payoffs[i] - mean;
        sq_sum += d * d;
    }
    double sample_var = (num_paths > 1) ? (sq_sum / static_cast<double>(num_paths - 1)) : 0.0;
    double stderr = (num_paths > 0) ? sqrt(sample_var / static_cast<double>(num_paths)) : 0.0;

    *out_price = exp(-opt.r * opt.T) * mean;
    *out_stderr = exp(-opt.r * opt.T) * stderr;
}
