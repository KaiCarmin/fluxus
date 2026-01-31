#pragma once
#include <memory>
#include "flux/RiemannSolver.hpp"
#include "Grid.hpp"

namespace fluxus {

    class TimeIntegrator {
    protected:
        std::shared_ptr<RiemannSolver> m_riemann_solver;

    public:
        // Inject the physics engine (HLL/HLLC)
        TimeIntegrator(std::shared_ptr<RiemannSolver> solver) 
            : m_riemann_solver(solver) {}
        
        virtual ~TimeIntegrator() = default;

        // The main command: "Advance the physics by dt"
        virtual void step(Grid& grid, double dt) = 0;
    };

}