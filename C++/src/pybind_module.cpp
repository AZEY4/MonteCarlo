#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "option.hpp"
#include "simulator.hpp"
#include "gpu_simulator.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_backend, m) {
    m.doc() = "MonteCarloProCUDA backend (CPU + GPU)";

    py::enum_<OptionType>(m, "OptionType")
        .value("EuropeanCall", OptionType::EuropeanCall)
        .value("EuropeanPut", OptionType::EuropeanPut)
        .value("AsianCall", OptionType::AsianCall)
        .value("AsianPut", OptionType::AsianPut)
        .export_values();

    py::class_<Option>(m, "Option")
        .def(py::init<>())
        .def_readwrite("type", &Option::type)
        .def_readwrite("S0", &Option::S0)
        .def_readwrite("K", &Option::K)
        .def_readwrite("T", &Option::T)
        .def_readwrite("r", &Option::r)
        .def_readwrite("sigma", &Option::sigma);

    m.def("price_cpu", [](const Option &opt, std::size_t paths) {
        MonteCarloSimulator sim(paths);
        auto res = sim.price(opt);
        return py::make_tuple(res.first, res.second);
    }, py::arg("opt"), py::arg("paths") = 100000);

    m.def("price_gpu", [](const Option &opt, std::size_t paths) {
        double price=0.0, stderr=0.0;
        monte_carlo_gpu_c(&opt, paths, &price, &stderr);
        return py::make_tuple(price, stderr);
    }, py::arg("opt"), py::arg("paths") = 100000);
}
