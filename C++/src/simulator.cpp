#include "simulator.hpp"
#include "utils.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <omp.h>

MonteCarloSimulator::MonteCarloSimulator(std::size_t paths) : num_paths(paths) {}

std::pair<double,double> MonteCarloSimulator::price(const Option &opt) const {
    if (opt.type == OptionType::EuropeanCall || opt.type == OptionType::EuropeanPut) {
        return price_european(opt);
    } else {
        return price_asian(opt);
    }
}

std::pair<double,double> MonteCarloSimulator::price_european(const Option &opt) const {
    const double drift = (opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T;
    const double vol_sqrt = opt.sigma * std::sqrt(opt.T);

    std::vector<double> payoffs;
    payoffs.resize(num_paths);

    #pragma omp parallel
    {
        auto &rng = thread_rng();
        std::normal_distribution<double> nd(0.0,1.0);

        #pragma omp for
        for (std::size_t i = 0; i < num_paths; ++i) {
            double z = nd(rng);
            double ST = opt.S0 * std::exp(drift + vol_sqrt * z);
            double payoff = 0.0;
            if (opt.type == OptionType::EuropeanCall) payoff = std::max(ST - opt.K, 0.0);
            else payoff = std::max(opt.K - ST, 0.0);
            payoffs[i] = payoff;
        }
    }

    double sum = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
    double mean = sum / static_cast<double>(num_paths);

    // sample standard deviation
    double sq_sum = 0.0;
    for (double x : payoffs) sq_sum += (x - mean) * (x - mean);
    double sample_var = sq_sum / static_cast<double>(num_paths - 1);
    double stderr = std::sqrt(sample_var / static_cast<double>(num_paths));

    double discounted = std::exp(-opt.r * opt.T) * mean;
    double discounted_stderr = std::exp(-opt.r * opt.T) * stderr;

    return {discounted, discounted_stderr};
}

std::pair<double,double> MonteCarloSimulator::price_asian(const Option &opt) const {
    const int steps = 100; // fixed discretization steps for the Asian average
    const double dt = opt.T / static_cast<double>(steps);
    const double drift_dt = (opt.r - 0.5 * opt.sigma * opt.sigma) * dt;
    const double vol_sqrt_dt = opt.sigma * std::sqrt(dt);

    std::vector<double> payoffs;
    payoffs.resize(num_paths);

    #pragma omp parallel
    {
        auto &rng = thread_rng();
        std::normal_distribution<double> nd(0.0,1.0);

        #pragma omp for
        for (std::size_t i = 0; i < num_paths; ++i) {
            double S = opt.S0;
            double avg = 0.0;
            for (int t = 0; t < steps; ++t) {
                double z = nd(rng);
                S *= std::exp(drift_dt + vol_sqrt_dt * z);
                avg += S;
            }
            avg /= static_cast<double>(steps);
            double payoff = 0.0;
            if (opt.type == OptionType::AsianCall) payoff = std::max(avg - opt.K, 0.0);
            else payoff = std::max(opt.K - avg, 0.0);
            payoffs[i] = payoff;
        }
    }

    double sum = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
    double mean = sum / static_cast<double>(num_paths);

    double sq_sum = 0.0;
    for (double x : payoffs) sq_sum += (x - mean) * (x - mean);
    double sample_var = sq_sum / static_cast<double>(num_paths - 1);
    double stderr = std::sqrt(sample_var / static_cast<double>(num_paths));

    double discounted = std::exp(-opt.r * opt.T) * mean;
    double discounted_stderr = std::exp(-opt.r * opt.T) * stderr;

    return {discounted, discounted_stderr};
}
