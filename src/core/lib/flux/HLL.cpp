// src/core/lib/flux/HLL.cpp
#include "flux/HLL.hpp"
#include <algorithm>

namespace fluxus {

    Flux HLLSolver::solve(const State& L, const State& R) const {
        double a_L = L.sound_speed(m_gamma);
        double a_R = R.sound_speed(m_gamma);

        // Davis Wave Speed Estimates
        double S_L = std::min(L.u - a_L, R.u - a_R);
        double S_R = std::max(L.u + a_L, R.u + a_R);

        if (S_L >= 0.0) {
            return L.to_flux(m_gamma);
        }
        if (S_R <= 0.0) {
            return R.to_flux(m_gamma);
        }

        // HLL Formula
        Conserved U_L = L.to_conserved(m_gamma);
        Conserved U_R = R.to_conserved(m_gamma);
        Flux F_L = L.to_flux(m_gamma);
        Flux F_R = R.to_flux(m_gamma);

        // The math works because Flux and Conserved are both Vector4
        return (F_L * S_R - F_R * S_L + (U_R - U_L) * (S_L * S_R)) * (1.0 / (S_R - S_L));
    }
}