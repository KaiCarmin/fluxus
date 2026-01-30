#pragma once
#include "RiemannSolver.hpp"
#include <algorithm>

namespace fluxus {

    class HLLCSolver : public RiemannSolver {
    public:
        using RiemannSolver::RiemannSolver; // Use parent constructor

        // Override the solve function
        Flux solve(const State& L, const State& R) const override;
    };

}