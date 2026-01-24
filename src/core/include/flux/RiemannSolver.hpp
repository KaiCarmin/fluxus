// src/core/include/flux/RiemannSolver.hpp
#pragma once
#include "types.hpp"

namespace fluxus {

    class RiemannSolver {
    protected:
        double m_gamma; // Adiabatic index (1.4 for air)

    public:
        // Constructor injects physics constants
        RiemannSolver(double gamma) : m_gamma(gamma) {}
        
        virtual ~RiemannSolver() = default;

        /**
         * The Main Contract.
         * Inputs: L and R states (Primitive variables: rho, u, v, p).
         * Output: Flux vector (Conserved fluxes: Mass, MomX, MomY, Energy).
         * * Note: The solver assumes 'u' is the normal velocity. 
         * If solving for Y-interface, the caller must swap u/v first.
         */
        virtual Flux solve(const State& L, const State& R) const = 0;
    };
}