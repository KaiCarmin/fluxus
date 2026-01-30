#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic string/vector conversion
#include "types.hpp"
#include "flux/HLLSolver.hpp"
#include "flux/HLLCSolver.hpp"
#include "integrator/TimeIntegrator.hpp"
#include "integrator/Godunov.hpp"
namespace py = pybind11;
using namespace fluxus;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fluxus High-Performance Core";

    // 1. Bind the State struct
    py::class_<State>(m, "State")
        // 2D Constructor (existing)
        .def(py::init<double, double, double, double>(), 
             py::arg("rho"), py::arg("u"), py::arg("v"), py::arg("p"))
        
        // --- NEW: 3D Constructor ---
        .def(py::init<double, double, double, double, double>(), 
             py::arg("rho"), py::arg("u"), py::arg("v"), py::arg("w"), py::arg("p"))
        
        // Properties
        .def_readwrite("rho", &State::rho)
        .def_readwrite("u", &State::u)
        .def_readwrite("v", &State::v)
        .def_readwrite("w", &State::w) // <-- New: expose Z-velocity
        .def_readwrite("p", &State::p)
        
        .def("__repr__", [](const State &s) {
            return "<State rho=" + std::to_string(s.rho) + 
                   " u=" + std::to_string(s.u) + 
                   " v=" + std::to_string(s.v) + 
                   " w=" + std::to_string(s.w) + 
                   " p=" + std::to_string(s.p) + ">";
        });

    // 2. Bind the Flux (Vector5) struct
    py::class_<Flux>(m, "Flux")
        // We use lambdas to access the union members safely
        .def_property_readonly("rho",   [](const Flux& f){ return f.rho; })
        .def_property_readonly("mom_x", [](const Flux& f){ return f.mom_x; })
        .def_property_readonly("mom_y", [](const Flux& f){ return f.mom_y; })
        .def_property_readonly("mom_z", [](const Flux& f){ return f.mom_z; }) // <-- New
        .def_property_readonly("E",     [](const Flux& f){ return f.E; })
        
        .def("__repr__", [](const Flux &f) {
            return "<Flux mass=" + std::to_string(f.rho) + 
                   " mx=" + std::to_string(f.mom_x) + 
                   " my=" + std::to_string(f.mom_y) + 
                   " mz=" + std::to_string(f.mom_z) + 
                   " E=" + std::to_string(f.E) + ">";
        });

    // 3. Bind the Solvers
    // bind the Base Class first...
    py::class_<RiemannSolver, std::shared_ptr<RiemannSolver>>(m, "RiemannSolver");

    // Bind the HLL Solver
    py::class_<HLLSolver, RiemannSolver, std::shared_ptr<HLLSolver>>(m, "HLLSolver")
        .def(py::init<double>(), py::arg("gamma") = 1.4)
        .def("solve", &HLLSolver::solve, py::arg("L"), py::arg("R"), 
             "Compute flux between two states");

    // Bind the HLLC Solver
    py::class_<HLLCSolver, RiemannSolver, std::shared_ptr<HLLCSolver>>(m, "HLLCSolver")
        .def(py::init<double>(), py::arg("gamma") = 1.4)
        .def("solve", &HLLCSolver::solve, py::arg("L"), py::arg("R"), 
             "Compute flux using HLLC (restores contact surface)");

    // 4. Bind Integrators
    py::class_<TimeIntegrator, std::shared_ptr<TimeIntegrator>>(m, "TimeIntegrator");

    // Bind the Godunov Integrator
    py::class_<GodunovIntegrator, TimeIntegrator, std::shared_ptr<GodunovIntegrator>>(m, "GodunovIntegrator")
        .def(py::init<std::shared_ptr<RiemannSolver>>())
        .def("step", &GodunovIntegrator::step, py::arg("grid"), py::arg("dt"), 
             "Advance the grid by one time step");
}