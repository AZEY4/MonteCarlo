#pragma once

#include "option.hpp"
#include <cstddef>

// Host-callable wrapper around GPU implementation.
extern "C" void monte_carlo_gpu_c(
    const Option *opt, // pointer to option (copy-safe)
    std::size_t num_paths,
    double *out_price, // out: price
    double *out_stderr // out: stderr
);
