#pragma once
#include "RiemannSolver.hpp"
#include <algorithm> // for std::abs

namespace fluxus {

    class RoeSolver : public RiemannSolver {
    public:
        // Inherit constructor (expects gamma)
        using RiemannSolver::RiemannSolver;

        // Override the solve function
        // Note: Returns ConservedVector (which acts as our Flux type)
        Flux solve(const State& L, const State& R) const override;
    };

}
