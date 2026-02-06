#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic string/vector conversion
#include <pybind11/numpy.h> // For numpy array support

#include "types.hpp"
#include "Grid.hpp"

#include "flux/HLL.hpp"
#include "flux/HLLC.hpp"
#include "flux/Roe.hpp"

#include "source-term/SourceTerm.hpp"
#include "source-term/Gravity.hpp"

#include "integrator/TimeIntegrator.hpp"
#include "integrator/Godunov.hpp"
#include "integrator/MUSCLHancock.hpp"

#include "reconstruct/Reconstructor.hpp"
#include "reconstruct/PiecewiseConstant.hpp"
#include "reconstruct/Minmod.hpp"
#include "reconstruct/Superbee.hpp"
#include "reconstruct/VanLeer.hpp"

namespace py = pybind11;
using namespace fluxus;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fluxus High-Performance Core";

    // ------------------------------------------
    // 1. Bind the State struct
    py::class_<State>(m, "State")
        // 2D Constructor (existing)
        .def(py::init<double, double, double, double>(), 
             py::arg("rho"), py::arg("u"), py::arg("v"), py::arg("p"))
        
        // 3D Constructor
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

    // ------------------------------------------
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

    // ------------------------------------------
    // 3. Bind Boundary Enum Type
    py::enum_<BoundaryType>(m, "BoundaryType")
        .value("Transmissive", BoundaryType::Transmissive)
        .value("Reflective", BoundaryType::Reflective)
        .value("Periodic", BoundaryType::Periodic)
        .export_values();

    // ------------------------------------------
    // 4. Bind Grid
    py::class_<Grid>(m, "Grid")
        .def(py::init<py::array_t<double>, int, int, int, int, int, double, double, double>(),
             py::arg("data"), py::arg("dim"), py::arg("nx"), py::arg("ny"), py::arg("nz"), 
             py::arg("ng"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "Create a Grid from a numpy array")
        .def_readonly("nx", &Grid::nx)
        .def_readonly("ny", &Grid::ny)
        .def_readonly("nz", &Grid::nz)
        .def_readonly("ndim", &Grid::ndim)
        .def_readonly("ng", &Grid::ng)
        .def_readonly("dx", &Grid::dx)
        .def_readonly("dy", &Grid::dy)
        .def_readonly("dz", &Grid::dz)
        .def("get_state", &Grid::get_state, 
             py::arg("i"), py::arg("j"), py::arg("k") = 0,
             "Get state at cell (i, j, k)")
        .def("apply_flux", py::overload_cast<int, int, int, const Flux&, double>(&Grid::apply_flux),
             py::arg("i"), py::arg("j"), py::arg("k"), py::arg("flux"), py::arg("dt_over_dx"),
             "Apply flux to cell (i, j, k)")
        .def("set_boundaries", &Grid::set_boundaries, py::arg("x_min"), py::arg("x_max"), py::arg("y_min"), py::arg("y_max"))
        .def("apply_boundaries", &Grid::apply_boundaries);

    // ------------------------------------------
    // 5. Bind the Solvers
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
    
    // Bind the Roe Solver
    py::class_<RoeSolver, RiemannSolver, std::shared_ptr<RoeSolver>>(m, "RoeSolver")
        .def(py::init<double>(), py::arg("gamma") = 1.4)
        .def("solve", &RoeSolver::solve, py::arg("L"), py::arg("R"), 
             "Compute flux using Roe's approximate Riemann solver");
    
    // ------------------------------------------
    // 6. Bind Reconstructors
    py::class_<Reconstructor, std::shared_ptr<Reconstructor>>(m, "Reconstructor");

    // Bind PiecewiseConstant
    py::class_<PiecewiseConstantReconstructor, Reconstructor, std::shared_ptr<PiecewiseConstantReconstructor>>(m, "PiecewiseConstantReconstructor")
        .def(py::init<>());

    // Bind MinmodReconstructor
    py::class_<MinmodReconstructor, Reconstructor, std::shared_ptr<MinmodReconstructor>>(m, "MinmodReconstructor")
        .def(py::init<>());

    // Bind SuperbeeReconstructor
    py::class_<SuperbeeReconstructor, Reconstructor, std::shared_ptr<SuperbeeReconstructor>>(m, "SuperbeeReconstructor")
        .def(py::init<>());
    
    // Bind VanLeerReconstructor
    py::class_<VanLeerReconstructor, Reconstructor, std::shared_ptr<VanLeerReconstructor>>(m, "VanLeerReconstructor")
        .def(py::init<>());

    // ------------------------------------------
    // 7. Bind Source Terms
    py::class_<SourceTerm, std::shared_ptr<SourceTerm>>(m, "SourceTerm");

    // Bind Gravity Source Term
    py::class_<Gravity, SourceTerm, std::shared_ptr<Gravity>>(m, "Gravity")
        .def(py::init<double, double, double>(), 
             py::arg("gx")=0.0, py::arg("gy")=0.0, py::arg("gz")=0.0);

    // ------------------------------------------
    // 8. Bind Integrators
    // Base class exposes add_source
    py::class_<TimeIntegrator, std::shared_ptr<TimeIntegrator>>(m, "TimeIntegrator")
        .def("add_source", &TimeIntegrator::add_source, py::arg("source"));
        
    // Bind GodunovIntegrator
    py::class_<GodunovIntegrator, TimeIntegrator, std::shared_ptr<GodunovIntegrator>>(m, "GodunovIntegrator")
        // Constructor
        .def(py::init<std::shared_ptr<RiemannSolver>>())
        .def(py::init<std::shared_ptr<RiemannSolver>, std::shared_ptr<Reconstructor>>())
        // Step function
        .def("step", &GodunovIntegrator::step, py::arg("grid"), py::arg("dt"), 
             "Advance the grid by one time step")
        // compute_dt function
        .def("compute_dt", &GodunovIntegrator::compute_dt, py::arg("grid"), py::arg("cfl"),
         "Calculate stable time step based on CFL condition");
    
    // Bind MUSCLHancockIntegrator
    py::class_<MUSCLHancockIntegrator, TimeIntegrator, std::shared_ptr<MUSCLHancockIntegrator>>(m, "MUSCLHancockIntegrator")
        .def(py::init<std::shared_ptr<RiemannSolver>, std::shared_ptr<Reconstructor>>())
        .def("step", &MUSCLHancockIntegrator::step, py::arg("grid"), py::arg("dt"))
        .def("compute_dt", &MUSCLHancockIntegrator::compute_dt, py::arg("grid"), py::arg("cfl"));
}