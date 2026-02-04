#include "flux/RoeSolver.hpp"
#include <cmath>

namespace fluxus {

    Flux RoeSolver::solve(const State& L, const State& R) const {
        // Get left and right fluxes
        Flux F_L = L.to_flux(m_gamma);
        Flux F_R = R.to_flux(m_gamma);
        
        // Get left and right conserved variables
        Conserved U_L = L.to_conserved(m_gamma);
        Conserved U_R = R.to_conserved(m_gamma);

        // Compute Roe-averaged quantities
        double sqrt_rho_L = std::sqrt(L.rho);
        double sqrt_rho_R = std::sqrt(R.rho);
        double denom = sqrt_rho_L + sqrt_rho_R;
        double rho_roe = sqrt_rho_L * sqrt_rho_R;
        
        // Roe-averaged velocities
        double u_roe = (sqrt_rho_L * L.u + sqrt_rho_R * R.u) / denom;
        double v_roe = (sqrt_rho_L * L.v + sqrt_rho_R * R.v) / denom;
        double w_roe = (sqrt_rho_L * L.w + sqrt_rho_R * R.w) / denom;
        
        // Roe-averaged enthalpy
        double H_L = (U_L.E + L.p) / L.rho;
        double H_R = (U_R.E + R.p) / R.rho;
        double H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom;
        
        // Roe-averaged sound speed
        double q_sq = u_roe*u_roe + v_roe*v_roe + w_roe*w_roe;
        double a_roe = std::sqrt((m_gamma - 1.0) * (H_roe - 0.5 * q_sq));
        
        // Eigenvalues (wave speeds)
        double lambda1 = u_roe - a_roe;
        double lambda2 = u_roe;
        double lambda3 = u_roe;
        double lambda4 = u_roe;
        double lambda5 = u_roe + a_roe;

        // Differences in primitive variables
        double dp = R.p - L.p;
        double du = R.u - L.u;
        double dv = R.v - L.v;
        double dw = R.w - L.w;
        
        // Differences in conserved variables
        double drho   = R.rho - L.rho;
        double dmom_x = U_R.mom_x - U_L.mom_x;
        double dmom_y = U_R.mom_y - U_L.mom_y;
        double dmom_z = U_R.mom_z - U_L.mom_z;
        double dE     = U_R.E - U_L.E;
        
        // Wave strengths (projection onto characteristic variables)
        double alpha1 = (dp - rho_roe * a_roe * du) / (2.0 * a_roe * a_roe);
        double alpha2 = (R.rho - L.rho) - dp / (a_roe * a_roe);
        double alpha3 = dv; // Shear strength Y
        double alpha4 = dw; // Shear strength Z
        double alpha5 = (dp + rho_roe * a_roe * du) / (2.0 * a_roe * a_roe);
        
        // Right eigenvectors multiplied by wave strengths
        Conserved wave1, wave2, wave3, wave4, wave5;
        
        // Wave 1: (u - a) wave
        wave1.rho   = alpha1 * 1.0;
        wave1.mom_x = alpha1 * (u_roe - a_roe);
        wave1.mom_y = alpha1 * v_roe;
        wave1.mom_z = alpha1 * w_roe;
        wave1.E     = alpha1 * (H_roe - u_roe * a_roe);
        
        // Wave 2: entropy wave (u)
        wave2.rho   = alpha2 * 1.0;
        wave2.mom_x = alpha2 * u_roe;
        wave2.mom_y = alpha2 * v_roe;
        wave2.mom_z = alpha2 * w_roe;
        wave2.E     = alpha2 * (0.5 * q_sq);
        
        // Wave 3: shear wave (u, v)
        wave3.rho   = alpha3 * 0.0;
        wave3.mom_x = alpha3 * 0.0;
        wave3.mom_y = alpha3 * rho_roe;
        wave3.mom_z = alpha3 * 0.0;
        wave3.E     = alpha3 * rho_roe * v_roe;
        
        // Wave 4: shear wave (u, w)
        wave4.rho   = alpha4 * 0.0;
        wave4.mom_x = alpha4 * 0.0;
        wave4.mom_y = alpha4 * 0.0;
        wave4.mom_z = alpha4 * rho_roe;
        wave4.E     = alpha4 * rho_roe * w_roe;
        
        // Wave 5: (u + a) wave
        wave5.rho   = alpha5 * 1.0;
        wave5.mom_x = alpha5 * (u_roe + a_roe);
        wave5.mom_y = alpha5 * v_roe;
        wave5.mom_z = alpha5 * w_roe;
        wave5.E     = alpha5 * (H_roe + u_roe * a_roe);
        
        // Roe flux: F_roe = 0.5 * (F_L + F_R) - 0.5 * sum(|lambda_i| * wave_i)
        Flux F_roe;
        F_roe.rho   = 0.5 * (F_L.rho + F_R.rho) - 0.5 * (std::abs(lambda1) * wave1.rho + 
                                                           std::abs(lambda2) * wave2.rho + 
                                                           std::abs(lambda3) * wave3.rho + 
                                                           std::abs(lambda4) * wave4.rho + 
                                                           std::abs(lambda5) * wave5.rho);
        
        F_roe.mom_x = 0.5 * (F_L.mom_x + F_R.mom_x) - 0.5 * (std::abs(lambda1) * wave1.mom_x + 
                                                               std::abs(lambda2) * wave2.mom_x + 
                                                               std::abs(lambda3) * wave3.mom_x + 
                                                               std::abs(lambda4) * wave4.mom_x + 
                                                               std::abs(lambda5) * wave5.mom_x);
        
        F_roe.mom_y = 0.5 * (F_L.mom_y + F_R.mom_y) - 0.5 * (std::abs(lambda1) * wave1.mom_y + 
                                                               std::abs(lambda2) * wave2.mom_y + 
                                                               std::abs(lambda3) * wave3.mom_y + 
                                                               std::abs(lambda4) * wave4.mom_y + 
                                                               std::abs(lambda5) * wave5.mom_y);
        
        F_roe.mom_z = 0.5 * (F_L.mom_z + F_R.mom_z) - 0.5 * (std::abs(lambda1) * wave1.mom_z + 
                                                               std::abs(lambda2) * wave2.mom_z + 
                                                               std::abs(lambda3) * wave3.mom_z + 
                                                               std::abs(lambda4) * wave4.mom_z + 
                                                               std::abs(lambda5) * wave5.mom_z);
        
        F_roe.E     = 0.5 * (F_L.E + F_R.E) - 0.5 * (std::abs(lambda1) * wave1.E + 
                                                       std::abs(lambda2) * wave2.E + 
                                                       std::abs(lambda3) * wave3.E + 
                                                       std::abs(lambda4) * wave4.E + 
                                                       std::abs(lambda5) * wave5.E);
        
        return F_roe;
    }
}
