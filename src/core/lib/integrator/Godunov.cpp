#include "integrator/Godunov.hpp"
#include <vector>
#include <cmath>
#include <omp.h> // Optional: For parallel execution

namespace fluxus {

    double GodunovIntegrator::compute_dt(const Grid& grid, double cfl) {
        double max_sx = 1e-9;
        double max_sy = 1e-9;
        double max_sz = 1e-9;
        double gamma = 1.4; // Ideally, get this from the solver or grid

        // 1. Find Maximum Wave Speeds (Hydrodynamics)
        // Use OpenMP reduction to make this fast on large grids
        #pragma omp parallel for reduction(max:max_sx, max_sy, max_sz)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    State s = grid.get_state(i, j, k);

                    if (s.rho > 1e-9 && s.p > 1e-9) {
                        double c = std::sqrt(gamma * s.p / s.rho);
                        
                        // Wave speed = Fluid Velocity + Sound Speed
                        if (std::abs(s.u) + c > max_sx) max_sx = std::abs(s.u) + c;
                        if (std::abs(s.v) + c > max_sy) max_sy = std::abs(s.v) + c;
                        if (std::abs(s.w) + c > max_sz) max_sz = std::abs(s.w) + c;
                    }
                }
            }
        }

        // 2. Compute Hydrodynamic dt limit
        double dt_x = grid.dx / max_sx;
        double dt_hydro = dt_x;

        if (grid.ndim >= 2) {
            double dt_y = grid.dy / max_sy;
            dt_hydro = std::min(dt_hydro, dt_y);
        }
        if (grid.ndim == 3) {
            double dt_z = grid.dz / max_sz;
            dt_hydro = std::min(dt_hydro, dt_z);
        }

        // 3. Compute Gravity dt limit (if gravity is active)
        // Physics: Distance = 0.5 * a * t^2  -->  t = sqrt(2*d/a)
        double dt_grav = 1e30; // Infinite
        if (std::abs(m_gravity_y) > 1e-12) {
            double min_d = grid.dx;
            if (grid.ndim >= 2) min_d = std::min(min_d, grid.dy);
            if (grid.ndim == 3) min_d = std::min(min_d, grid.dz);
            
            dt_grav = std::sqrt(2.0 * min_d / std::abs(m_gravity_y));
        }

        // 4. Return safe dt
        return cfl * std::min(dt_hydro, dt_grav);
    }

    void GodunovIntegrator::step(Grid& grid, double dt) {
        // 1. Fluid Dynamics (Fluxes) with Dimensional Splitting
        sweep_x(grid, dt);
        
        if (grid.ndim >= 2) sweep_y(grid, dt);
        if (grid.ndim == 3) sweep_z(grid, dt);

        // 2. External Forces (Source Terms)
        // Only run if gravity is effectively non-zero
        if (std::abs(m_gravity_y) > 1e-12) {
            apply_sources(grid, dt);
        }
    }

    // --- X SWEEP (BUFFERED) ---
    void GodunovIntegrator::sweep_x(Grid& grid, double dt) {
        double dt_dx = dt / grid.dx;
        
        // We use a buffer to store fluxes for one row to prevent "in-place update" bugs
        std::vector<Flux> fluxes(grid.nx + 1);

        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                
                // STEP 1: Compute ALL fluxes for this row using the OLD state
                for (int i = 0; i < grid.nx + 1; ++i) {
                    State L = grid.get_state(i - 1, j, k);
                    State R = grid.get_state(i, j, k);
                    fluxes[i] = m_riemann_solver->solve(L, R);
                }

                // STEP 2: Apply fluxes to update cells
                for (int i = 0; i < grid.nx + 1; ++i) {
                    // Update Left Cell (i-1)
                    if (i > 0) {
                        grid.apply_flux(i - 1, j, k, fluxes[i], dt_dx);
                    }
                    // Update Right Cell (i)
                    if (i < grid.nx) {
                        grid.apply_flux(i, j, k, fluxes[i], -dt_dx);
                    }
                }
            }
        }
    }

    // --- Y SWEEP (BUFFERED) ---
    void GodunovIntegrator::sweep_y(Grid& grid, double dt) {
        double dt_dy = dt / grid.dy;
        std::vector<Flux> fluxes(grid.ny + 1);

        for (int k = 0; k < grid.nz; ++k) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // STEP 1: Compute Fluxes
                for (int j = 0; j < grid.ny + 1; ++j) {
                    State L_raw = grid.get_state(i, j - 1, k);
                    State R_raw = grid.get_state(i, j, k);

                    // ROTATE: Normal velocity becomes 'u' (swap u and v)
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

    // --- Z SWEEP (BUFFERED) ---
    void GodunovIntegrator::sweep_z(Grid& grid, double dt) {
        double dt_dz = dt / grid.dz;
        std::vector<Flux> fluxes(grid.nz + 1);

        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                
                // STEP 1: Compute Fluxes
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

    // --- SOURCE TERMS ---
    void GodunovIntegrator::apply_sources(Grid& grid, double dt) {
        
        // Loop over internal domain
        #pragma omp parallel for collapse(3) 
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    
                    State s = grid.get_state(i, j, k);
                    
                    // Momentum Change: d(rho*v) = rho * g * dt
                    double d_mom_y = s.rho * m_gravity_y * dt;
                    
                    // Energy Change: dE = (rho * v * g) * dt
                    // Work done by gravity force
                    double d_energy = (s.rho * s.v) * m_gravity_y * dt;
                    
                    Flux source;
                    source.rho   = 0.0;
                    source.mom_x = 0.0;
                    source.mom_y = d_mom_y; 
                    source.mom_z = 0.0;
                    source.E     = d_energy;
                    
                    // Add source: ptr += source
                    // apply_flux does: ptr -= factor * flux
                    // So we use factor = -1.0
                    grid.apply_flux(i, j, k, source, -1.0);
                }
            }
        }
    }
}