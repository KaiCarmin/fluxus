#pragma once
#include <memory>
#include <vector>
#include "flux/RiemannSolver.hpp"
#include "source-term/SourceTerm.hpp"
#include "Grid.hpp"

namespace fluxus {

    class TimeIntegrator {
    protected:
        std::shared_ptr<RiemannSolver> m_riemann_solver;
        std::vector<std::shared_ptr<SourceTerm>> m_sources;

        // apply all sources
        void apply_all_sources(Grid& grid, double dt) {
            for (auto& source : m_sources) {
                source->apply(grid, dt);
            }
        }

    public:
        // Inject the flux solver
        TimeIntegrator(std::shared_ptr<RiemannSolver> solver) 
            : m_riemann_solver(solver) {}
        
        virtual ~TimeIntegrator() = default;

        // Generic Source Management
        void add_source(std::shared_ptr<SourceTerm> source) {
            m_sources.push_back(source);
        }

        // The main command: "Advance the physics by dt"
        virtual void step(Grid& grid, double dt) = 0;
    };

}