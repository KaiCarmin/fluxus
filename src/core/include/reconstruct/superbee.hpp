#pragma once
#include "Reconstructor.hpp"
#include <algorithm>
#include <cmath>

namespace fluxus {
    class SuperbeeReconstructor : public Reconstructor {
    public:
        
        // The Superbee Limiter: A less dissipative limiter that allows steeper gradients.
        // Returns max(0, min(2*a, b), min(a, 2*b)) with proper sign handling.
        inline double superbee(double a, double b) {
            if (a * b <= 0.0) {
                return 0.0;
            }
            
            double sign = (a > 0.0) ? 1.0 : -1.0;
            double abs_a = std::abs(a);
            double abs_b = std::abs(b);
            
            double option1 = std::min(2.0 * abs_a, abs_b);
            double option2 = std::min(abs_a, 2.0 * abs_b);
            
            return sign * std::max(option1, option2);
        }

        // Helper to reconstruct a single scalar variable
        // We need 3 values: Left-of-center, Center, Right-of-center
        // to compute the slope at the Center.
        // Returns the value at the boundary face.
        // side: +1 for Right Face (U + 0.5*slope), -1 for Left Face (U - 0.5*slope)
        double reconstruct_scalar(double val_minus, double val_cen, double val_plus, int side) {
            double slope_L = val_cen - val_minus;
            double slope_R = val_plus - val_cen;
            
            double slope = superbee(slope_L, slope_R);
            
            return val_cen + (side * 0.5 * slope);
        }

        std::pair<State, State> reconstruct_interface(
            const Grid& grid, int i, int j, int k, int axis
        ) override {
            State L_out, R_out;
            
            // Indices for the stencil:
            // Interface is between Cell "Left" (idx-1) and Cell "Right" (idx)
            // To reconstruct Cell "Left" at its right face, we need (idx-2, idx-1, idx)
            // To reconstruct Cell "Right" at its left face, we need (idx-1, idx, idx+1)
            
            State U_LL, U_L, U_R, U_RR;

            if (axis == 0) { // X-Axis
                U_LL = grid.get_state(i - 2, j, k);
                U_L  = grid.get_state(i - 1, j, k);
                U_R  = grid.get_state(i,     j, k);
                U_RR = grid.get_state(i + 1, j, k);
            } 
            else if (axis == 1) { // Y-Axis
                U_LL = grid.get_state(i, j - 2, k);
                U_L  = grid.get_state(i, j - 1, k);
                U_R  = grid.get_state(i, j,     k);
                U_RR = grid.get_state(i, j + 1, k);
            }
            else { // Z-Axis
                U_LL = grid.get_state(i, j, k - 2);
                U_L  = grid.get_state(i, j, k - 1);
                U_R  = grid.get_state(i, j, k);
                U_RR = grid.get_state(i, j, k + 1);
            }

            // --- Reconstruct Left State (at Right Face of Cell i-1) ---
            L_out.rho = reconstruct_scalar(U_LL.rho, U_L.rho, U_R.rho,  1);
            L_out.u   = reconstruct_scalar(U_LL.u,   U_L.u,   U_R.u,    1);
            L_out.v   = reconstruct_scalar(U_LL.v,   U_L.v,   U_R.v,    1);
            L_out.w   = reconstruct_scalar(U_LL.w,   U_L.w,   U_R.w,    1);
            L_out.p   = reconstruct_scalar(U_LL.p,   U_L.p,   U_R.p,    1);

            // --- Reconstruct Right State (at Left Face of Cell i) ---
            R_out.rho = reconstruct_scalar(U_L.rho, U_R.rho, U_RR.rho, -1);
            R_out.u   = reconstruct_scalar(U_L.u,   U_R.u,   U_RR.u,   -1);
            R_out.v   = reconstruct_scalar(U_L.v,   U_R.v,   U_RR.v,   -1);
            R_out.w   = reconstruct_scalar(U_L.w,   U_R.w,   U_RR.w,   -1);
            R_out.p   = reconstruct_scalar(U_L.p,   U_R.p,   U_RR.p,   -1);

            return {L_out, R_out};
        }
    };
}
