#include "integrator/Godunov.hpp"
#include <omp.h> // Optional: For future parallelization

namespace fluxus {

    void GodunovIntegrator::step(Grid& grid, double dt) {
        // 1. Always do X-Sweep
        sweep_x(grid, dt);

        // 2. If 2D or 3D, do Y-Sweep
        if (grid.ndim >= 2) {
            sweep_y(grid, dt);
        }

        // 3. If 3D, do Z-Sweep
        if (grid.ndim == 3) {
            sweep_z(grid, dt);
        }
    }

    // --- X SWEEP ---
    void GodunovIntegrator::sweep_x(Grid& grid, double dt) {
        double dt_dx = dt / grid.dx;
        
        // Loop over Row (j) and Depth (k)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                
                // Loop over Interfaces (i)
                for (int i = 0; i < grid.nx + 1; ++i) {
                    State L = grid.get_state(i - 1, j, k);
                    State R = grid.get_state(i, j, k);

                    Flux F = m_riemann_solver->solve(L, R);

                    // Update neighbors
                    grid.apply_flux(i - 1, j, k, F,  dt_dx);
                    grid.apply_flux(i,     j, k, F, -dt_dx);
                }
            }
        }
    }

    // --- Y SWEEP ---
    void GodunovIntegrator::sweep_y(Grid& grid, double dt) {
        double dt_dy = dt / grid.dy;

        for (int k = 0; k < grid.nz; ++k) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // Loop over Y-Interfaces (j)
                for (int j = 0; j < grid.ny + 1; ++j) {
                    
                    State L_raw = grid.get_state(i, j - 1, k);
                    State R_raw = grid.get_state(i, j, k);

                    // ROTATE: Normal velocity becomes 'u'
                    State L_rot = L_raw; L_rot.u = L_raw.v; L_rot.v = L_raw.u;
                    State R_rot = R_raw; R_rot.u = R_raw.v; R_rot.v = R_raw.u;

                    Flux F_rot = m_riemann_solver->solve(L_rot, R_rot);

                    // UN-ROTATE Flux: Swap mom_x and mom_y
                    Flux F_final = F_rot;
                    F_final.mom_x = F_rot.mom_y;
                    F_final.mom_y = F_rot.mom_x;

                    grid.apply_flux(i, j - 1, k, F_final,  dt_dy);
                    grid.apply_flux(i, j,     k, F_final, -dt_dy);
                }
            }
        }
    }

    // --- Z SWEEP ---
    void GodunovIntegrator::sweep_z(Grid& grid, double dt) {
        double dt_dz = dt / grid.dz;

        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // Loop over Z-Interfaces (k)
                for (int k = 0; k < grid.nz + 1; ++k) {
                    
                    State L_raw = grid.get_state(i, j, k - 1);
                    State R_raw = grid.get_state(i, j, k);

                    // ROTATE: Normal velocity becomes 'u' (swap u and w)
                    State L_rot = L_raw; L_rot.u = L_raw.w; L_rot.w = L_raw.u;
                    State R_rot = R_raw; R_rot.u = R_raw.w; R_rot.w = R_raw.u;

                    Flux F_rot = m_riemann_solver->solve(L_rot, R_rot);

                    // UN-ROTATE Flux: Swap mom_x and mom_z
                    Flux F_final = F_rot;
                    F_final.mom_x = F_rot.mom_z;
                    F_final.mom_z = F_rot.mom_x;

                    grid.apply_flux(i, j, k - 1, F_final,  dt_dz);
                    grid.apply_flux(i, j, k,     F_final, -dt_dz);
                }
            }
        }
    }
}