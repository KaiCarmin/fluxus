#include "flux/HLLCSolver.hpp"
#include <cmath>

namespace fluxus {

    Flux HLLCSolver::solve(const State& L, const State& R) const {
        double a_L = L.sound_speed(m_gamma);
        double a_R = R.sound_speed(m_gamma);

        // 1. Wave Speed Estimates (Same as HLL)
        double S_L = std::min(L.u - a_L, R.u - a_R);
        double S_R = std::max(L.u + a_L, R.u + a_R);

        // 2. Trivial Supersonic Cases
        if (S_L >= 0.0) return L.to_flux(m_gamma);
        if (S_R <= 0.0) return R.to_flux(m_gamma);

        // 3. Compute Star Speed (S_star)
        // This is the velocity of the Contact Discontinuity
        double rho_L = L.rho;
        double rho_R = R.rho;
        
        // Numerator: p_R - p_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R)
        double numer = R.p - L.p + rho_L * L.u * (S_L - L.u) - rho_R * R.u * (S_R - R.u);
        // Denominator: rho_L*(S_L - u_L) - rho_R*(S_R - u_R)
        double denom = rho_L * (S_L - L.u) - rho_R * (S_R - R.u);
        
        double S_star = numer / denom;

        // 4. Select the Correct Flux
        // We know we are inside the wave fan (S_L < 0 < S_R).
        // Check which side of the Contact Discontinuity we are on.
        
        if (S_star >= 0.0) {
            // --- Left Star Region (Between S_L and S_star) ---
            Conserved U_L = L.to_conserved(m_gamma);
            Flux F_L = L.to_flux(m_gamma);
            
            // HLLC Flux Formula for Left Star State:
            // F*_L = F_L + S_L * (U*_L - U_L)
            // Where U*_L = rho_L * ((S_L - u_L) / (S_L - S_star)) * [1, S_star, v_L, E_L/rho_L + (S_star - u_L)*(S_star + p_L/(rho_L*(S_L - u_L)))]
            
            // Simplified Factor
            double factor = rho_L * (S_L - L.u) / (S_L - S_star);
            
            Conserved U_star_L;
            U_star_L.rho   = factor;              
            U_star_L.mom_x = factor * S_star;     
            U_star_L.mom_y = factor * L.v;   // v is preserved
            U_star_L.mom_z = factor * L.w;   // w is preserved
            
            // Energy* is complicated:
            // E* = E_L + (S_star - u_L) * (S_star + p_L/(rho_L*(S_L - u_L))) * factor? 
            // Actually, simpler form: 
            // U*_E = factor * ( E_L/rho_L + (S_star - L.u)*(S_star + L.p/(rho_L*(S_L - L.u))) )
            double E_term = (U_L.E / rho_L) + (S_star - L.u) * (S_star + L.p / (rho_L * (S_L - L.u)));            
            U_star_L.E = factor * E_term;

            return F_L + (U_star_L - U_L) * S_L;
        } 
        else {
            // --- Right Star Region (Between S_star and S_R) ---
            Conserved U_R = R.to_conserved(m_gamma);
            Flux F_R = R.to_flux(m_gamma);
            
            double factor = rho_R * (S_R - R.u) / (S_R - S_star);
            
            Conserved U_star_R;
            U_star_R.rho   = factor;
            U_star_R.mom_x = factor * S_star;
            U_star_R.mom_y = factor * R.v;   // v is preserved
            U_star_R.mom_z = factor * R.w;   // w is preserved
            
            double E_term = (U_R.E / rho_R) + (S_star - R.u) * (S_star + R.p / (rho_R * (S_R - R.u)));
            U_star_R.E = factor * E_term;

            return F_R + (U_star_R - U_R) * S_R;
        }
    }
}