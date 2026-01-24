#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic string/vector conversion
#include "types.hpp"
#include "flux/HLLSolver.hpp"

namespace py = pybind11;
using namespace fluxus;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fluxus High-Performance Core";

    // 1. Bind the State struct
    py::class_<State>(m, "State")
        .def(py::init<double, double, double, double>(), 
             py::arg("rho"), py::arg("u"), py::arg("v"), py::arg("p"))
        .def_readwrite("rho", &State::rho)
        .def_readwrite("u", &State::u)
        .def_readwrite("v", &State::v)
        .def_readwrite("p", &State::p)
        .def("__repr__", [](const State &s) {
            return "<State rho=" + std::to_string(s.rho) + " p=" + std::to_string(s.p) + ">";
        });

    // 2. Bind the Flux (Vector4) struct
    py::class_<Flux>(m, "Flux")
        // We only need read access in Python for tests
        .def_property_readonly("rho", [](const Flux& f){ return f[0]; })
        .def_property_readonly("mom_x", [](const Flux& f){ return f[1]; })
        .def_property_readonly("mom_y", [](const Flux& f){ return f[2]; })
        .def_property_readonly("E", [](const Flux& f){ return f[3]; })
        .def("__repr__", [](const Flux &f) {
            return "<Flux mass=" + std::to_string(f[0]) + " E=" + std::to_string(f[3]) + ">";
        });

    // 3. Bind the Solvers
    // We bind the Base Class first...
    py::class_<RiemannSolver, std::shared_ptr<RiemannSolver>>(m, "RiemannSolver");

    // ...then the HLL Solver inheriting from it
    py::class_<HLLSolver, RiemannSolver, std::shared_ptr<HLLSolver>>(m, "HLLSolver")
        .def(py::init<double>(), py::arg("gamma") = 1.4)
        .def("solve", &HLLSolver::solve, py::arg("L"), py::arg("R"), 
             "Compute flux between two states");
}