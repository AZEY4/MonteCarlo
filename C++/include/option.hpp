#pragma once

enum class OptionType { EuropeanCall = 0, EuropeanPut = 1, AsianCall = 2, AsianPut = 3 };

struct Option {
    OptionType type;
    double S0;    // initial price
    double K;     // strike
    double T;     // maturity (years)
    double r;     // risk-free rate
    double sigma; // volatility
};
