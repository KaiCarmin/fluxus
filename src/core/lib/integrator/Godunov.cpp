#include "integrator/Godunov.hpp"
#include <vector>

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
        
        // Temporary buffer to hold fluxes for one row
        // Size = nx + 1 (number of interfaces)
        std::vector<Flux> fluxes(grid.nx + 1);

        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                
                // STEP 1: Compute ALL fluxes for this row using old state
                for (int i = 0; i < grid.nx + 1; ++i) {
                    State L = grid.get_state(i - 1, j, k);
                    State R = grid.get_state(i, j, k);
                    fluxes[i] = m_riemann_solver->solve(L, R);
                }

                // STEP 2: Apply fluxes to update cells
                for (int i = 0; i < grid.nx + 1; ++i) {
                    // Interface i affects Cell i-1 (Left) and Cell i (Right)
                    // We only update Real cells (0 to nx-1). 
                    // Ghosts are read-only during the sweep (usually).
                    
                    // Update Cell i-1 (if it's real)
                    if (i > 0) {
                        grid.apply_flux(i - 1, j, k, fluxes[i], dt_dx);
                    }
                    
                    // Update Cell i (if it's real)
                    if (i < grid.nx) {
                        grid.apply_flux(i, j, k, fluxes[i], -dt_dx);
                    }
                }
            }
        }
    }

    // --- Y SWEEP ---
    void GodunovIntegrator::sweep_y(Grid& grid, double dt) {
        double dt_dy = dt / grid.dy;
        std::vector<Flux> fluxes(grid.ny + 1);

        for (int k = 0; k < grid.nz; ++k) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // STEP 1: Compute Fluxes
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
                    
                    fluxes[j] = F_final;
                }

                // STEP 2: Apply Fluxes
                for (int j = 0; j < grid.ny + 1; ++j) {
                    if (j > 0)        grid.apply_flux(i, j - 1, k, fluxes[j], dt_dy);
                    if (j < grid.ny)  grid.apply_flux(i, j,     k, fluxes[j], -dt_dy);
                }
            }
        }
    }

    // --- Z SWEEP ---
    void GodunovIntegrator::sweep_z(Grid& grid, double dt) {
        double dt_dz = dt / grid.dz;
        std::vector<Flux> fluxes(grid.nz + 1);

        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // STEP 1: Compute Fluxes
                for (int k = 0; k < grid.nz + 1; ++k) {
                    State L_raw = grid.get_state(i, j, k - 1);
                    State R_raw = grid.get_state(i, j, k);

                    State L_rot = L_raw; L_rot.u = L_raw.w; L_rot.w = L_raw.u;
                    State R_rot = R_raw; R_rot.u = R_raw.w; R_rot.w = R_raw.u;

                    Flux F_rot = m_riemann_solver->solve(L_rot, R_rot);

                    Flux F_final = F_rot;
                    F_final.mom_x = F_rot.mom_z;
                    F_final.mom_z = F_rot.mom_x;

                    fluxes[k] = F_final;
                }

                // STEP 2: Apply Fluxes
                for (int k = 0; k < grid.nz + 1; ++k) {
                    if (k > 0)        grid.apply_flux(i, j, k - 1, fluxes[k], dt_dz);
                    if (k < grid.nz)  grid.apply_flux(i, j, k,     fluxes[k], -dt_dz);
                }
            }
        }
    }
}