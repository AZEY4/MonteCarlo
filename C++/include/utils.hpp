#pragma once

#include <random>
#include <cmath>
#include <thread>

inline double norm_pdf(double x) {
    return 0.3989422804014327 * std::exp(-0.5 * x * x);
}

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Thread-safe RNG helper: each thread should use its own generator
inline std::mt19937_64 &thread_rng() {
    thread_local std::mt19937_64 rng([](){
        std::random_device rd;
        std::seed_seq seq{rd(), (unsigned)std::hash<std::thread::id>()(std::this_thread::get_id())};
        return std::mt19937_64(seq);
    }());
    return rng;
}
