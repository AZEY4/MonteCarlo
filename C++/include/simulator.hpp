#pragma once

#include "option.hpp"
#include <cstddef>
#include <utility>

class MonteCarloSimulator {
public:
    MonteCarloSimulator(std::size_t paths = 100000);

    // Price returns discounted price (mean). The second value is estimated standard error.
    std::pair<double,double> price(const Option &opt) const;

private:
    std::size_t num_paths;
    std::pair<double,double> price_european(const Option &opt) const;
    std::pair<double,double> price_asian(const Option &opt) const;
};
