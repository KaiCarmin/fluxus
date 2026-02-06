#include "integrator/MUSCLHancock.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <vector>

namespace fluxus {

    // --- ADAPTIVE TIME STEPPING ---
    double MUSCLHancockIntegrator::compute_dt(const Grid& grid, double cfl) {
        double max_sx = 1e-9;
        double max_sy = 1e-9;
        double max_sz = 1e-9;
        double gamma = 1.4;

        #pragma omp parallel for reduction(max:max_sx, max_sy, max_sz)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    State s = grid.get_state(i, j, k);

                    if (s.rho > 1e-9 && s.p > 1e-9) {
                        double c = std::sqrt(gamma * s.p / s.rho);
                        
                        if (std::abs(s.u) + c > max_sx) max_sx = std::abs(s.u) + c;
                        if (std::abs(s.v) + c > max_sy) max_sy = std::abs(s.v) + c;
                        if (std::abs(s.w) + c > max_sz) max_sz = std::abs(s.w) + c;
                    }
                }
            }
        }

        double dt_hydro = grid.dx / max_sx;
        if (grid.ndim >= 2) dt_hydro = std::min(dt_hydro, grid.dy / max_sy);
        if (grid.ndim == 3) dt_hydro = std::min(dt_hydro, grid.dz / max_sz);

        return cfl * dt_hydro;
    }

    void MUSCLHancockIntegrator::step(Grid& grid, double dt) {
        // 1. Directional Splitting Sweeps
        sweep_x(grid, dt);
        
        if (grid.ndim >= 2) sweep_y(grid, dt);
        if (grid.ndim == 3) sweep_z(grid, dt);

        // 2. Apply Generic Sources (Gravity, etc.)
        // This iterates over m_sources and calls apply() on each
        apply_all_sources(grid, dt);
    }

    // --- X SWEEP ---
    void MUSCLHancockIntegrator::sweep_x(Grid& grid, double dt) {
        double dt_dx = dt / grid.dx;
        int nx = grid.nx;
        
        // Thread-private buffers for evolved states
        #pragma omp parallel for
        for (int k = 0; k < grid.nz; ++k) {
            // Allocate inside the loop for thread safety
            std::vector<State> evolved_L(nx + 2);
            std::vector<State> evolved_R(nx + 2);

            for (int j = 0; j < grid.ny; ++j) {
                
                // PASS 1: Reconstruct & Evolve (Predictor)
                // We iterate i from 0 to nx-1 to cover all cells that need evolution
                for (int i = 0; i < nx; ++i) { 
                    // To update cell 'i', we need its Left Face (from interface i)
                    // and its Right Face (from interface i+1).
                    // reconstruct_interface(i) returns {RightFace(i-1), LeftFace(i)}
                    
                    // We need Cell i's LEFT boundary: (Right side of interface i)
                    auto [dummy, face_L] = m_reconstructor->reconstruct_interface(grid, i, j, k, 0);
                    
                    // We need Cell i's RIGHT boundary: (Left side of interface i+1)
                    auto [face_R, dummy2] = m_reconstructor->reconstruct_interface(grid, i + 1, j, k, 0);

                    // Compute Fluxes at these boundaries
                    Flux F_L = state_to_flux(face_L);
                    Flux F_R = state_to_flux(face_R);
                    
                    // Half-Step Evolution (dt/2)
                    // U^* = U^n - 0.5 * (dt/dx) * (F_R - F_L)
                    double factor = 0.5 * dt_dx;
                    
                    // Helper to evolve
                    auto evolve = [&](double u, double f_l, double f_r) { return u - factor * (f_r - f_l); };
                    
                    Conserved U_L = face_L.to_conserved(1.4);
                    Conserved U_R = face_R.to_conserved(1.4);
                    
                    // Evolve both boundary states using the cell-average flux gradient
                    // (Note: standard MUSCL evolves primitive, but conserved is also valid and often more stable)
                    U_L.rho   = evolve(U_L.rho,   F_L.rho,   F_R.rho);
                    U_L.mom_x = evolve(U_L.mom_x, F_L.mom_x, F_R.mom_x);
                    U_L.mom_y = evolve(U_L.mom_y, F_L.mom_y, F_R.mom_y);
                    U_L.mom_z = evolve(U_L.mom_z, F_L.mom_z, F_R.mom_z);
                    U_L.E     = evolve(U_L.E,     F_L.E,     F_R.E);

                    U_R.rho   = evolve(U_R.rho,   F_L.rho,   F_R.rho);
                    U_R.mom_x = evolve(U_R.mom_x, F_L.mom_x, F_R.mom_x);
                    U_R.mom_y = evolve(U_R.mom_y, F_L.mom_y, F_R.mom_y);
                    U_R.mom_z = evolve(U_R.mom_z, F_L.mom_z, F_R.mom_z);
                    U_R.E     = evolve(U_R.E,     F_L.E,     F_R.E);
                    
                    evolved_L[i] = State::from_conserved(U_L.rho, U_L.mom_x, U_L.mom_y, U_L.mom_z, U_L.E, 1.4);
                    evolved_R[i] = State::from_conserved(U_R.rho, U_R.mom_x, U_R.mom_y, U_R.mom_z, U_R.E, 1.4);
                }
                
                // PASS 2: Riemann Solve (Corrector)
                for (int i = 0; i < nx + 1; ++i) {
                    // Flux at interface i is between Cell i-1 (Right Face) and Cell i (Left Face)
                    // Be careful with buffer indices.
                    // evolved_R[i-1] is the RIGHT face of cell i-1
                    // evolved_L[i]   is the LEFT face of cell i
                    
                    // Safely handle domain edges if i=0 or i=nx, rely on reconstruction ghost handling
                    State L, R;
                    
                    if (i == 0) L = m_reconstructor->reconstruct_interface(grid, i, j, k, 0).first; // Fallback to 1st order/boundary at very edge?
                    else        L = evolved_R[i - 1];

                    if (i == nx) R = m_reconstructor->reconstruct_interface(grid, i, j, k, 0).second;
                    else         R = evolved_L[i];

                    Flux f = m_riemann_solver->solve(L, R);
                    
                    if (i > 0)  grid.apply_flux(i - 1, j, k, f, dt_dx);
                    if (i < nx) grid.apply_flux(i,     j, k, f, -dt_dx);
                }
            }
        }
    }

    // --- Y SWEEP ---
    void MUSCLHancockIntegrator::sweep_y(Grid& grid, double dt) {
        double dt_dy = dt / grid.dy;
        int ny = grid.ny;

        #pragma omp parallel for
        for (int k = 0; k < grid.nz; ++k) {
            std::vector<State> evolved_L(ny + 2);
            std::vector<State> evolved_R(ny + 2);
            
            // ROTATE: Swap u and v so normal velocity is first
            auto rotate = [](State s) { std::swap(s.u, s.v); return s; };
            
            for (int i = 0; i < grid.nx; ++i) {
                
                // PASS 1: Reconstruct & Evolve
                for (int j = 0; j < ny; ++j) {
                    // Reconstruct along Y axis (1)
                    auto [dummy, face_L_raw] = m_reconstructor->reconstruct_interface(grid, i, j, k, 1);
                    auto [face_R_raw, dummy2] = m_reconstructor->reconstruct_interface(grid, i, j + 1, k, 1);
                    State face_L = rotate(face_L_raw);
                    State face_R = rotate(face_R_raw);

                    Flux F_L = state_to_flux(face_L);
                    Flux F_R = state_to_flux(face_R);

                    double factor = 0.5 * dt_dy;
                    auto evolve = [&](double u, double f_l, double f_r) { return u - factor * (f_r - f_l); };
                    
                    // Evolve Rotated Conserved Variables
                    Conserved U_L = face_L.to_conserved(1.4);
                    Conserved U_R = face_R.to_conserved(1.4);

                    U_L.rho = evolve(U_L.rho, F_L.rho, F_R.rho);
                    U_L.mom_x = evolve(U_L.mom_x, F_L.mom_x, F_R.mom_x); // This is phys Y-mom
                    U_L.mom_y = evolve(U_L.mom_y, F_L.mom_y, F_R.mom_y); // This is phys X-mom
                    U_L.mom_z = evolve(U_L.mom_z, F_L.mom_z, F_R.mom_z);
                    U_L.E     = evolve(U_L.E,     F_L.E,     F_R.E);
                    
                    // Do same for U_R...
                    U_R.rho = evolve(U_R.rho, F_L.rho, F_R.rho);
                    U_R.mom_x = evolve(U_R.mom_x, F_L.mom_x, F_R.mom_x);
                    U_R.mom_y = evolve(U_R.mom_y, F_L.mom_y, F_R.mom_y);
                    U_R.mom_z = evolve(U_R.mom_z, F_L.mom_z, F_R.mom_z);
                    U_R.E     = evolve(U_R.E,     F_L.E,     F_R.E);

                    evolved_L[j] = State::from_conserved(U_L.rho, U_L.mom_x, U_L.mom_y, U_L.mom_z, U_L.E, 1.4);
                    evolved_R[j] = State::from_conserved(U_R.rho, U_R.mom_x, U_R.mom_y, U_R.mom_z, U_R.E, 1.4);
                }

                // PASS 2: Solve
                for (int j = 0; j < ny + 1; ++j) {
                    State L = (j==0) ? rotate(m_reconstructor->reconstruct_interface(grid, i, j, k, 1).first) : evolved_R[j - 1];
                    State R = (j==ny) ? rotate(m_reconstructor->reconstruct_interface(grid, i, j, k, 1).second) : evolved_L[j];

                    Flux F_rot = m_riemann_solver->solve(L, R);

                    // UN-ROTATE Flux: Swap mom_x and mom_y
                    Flux F_final = F_rot;
                    std::swap(F_final.mom_x, F_final.mom_y);

                    if (j > 0)  grid.apply_flux(i, j - 1, k, F_final, dt_dy);
                    if (j < ny) grid.apply_flux(i,     j, k, F_final, -dt_dy);
                }
            }
        }
    }

    // --- Z SWEEP ---
    void MUSCLHancockIntegrator::sweep_z(Grid& grid, double dt) {
        double dt_dz = dt / grid.dz;
        int nz = grid.nz;

        #pragma omp parallel for
        for (int j = 0; j < grid.ny; ++j) {
            std::vector<State> evolved_L(nz + 2);
            std::vector<State> evolved_R(nz + 2);
            
            for (int i = 0; i < grid.nx; ++i) {
                // ROTATE: Swap u and w
                auto rotate = [](State s) { std::swap(s.u, s.w); return s; };
                
                // PASS 1
                for (int k = 0; k < nz; ++k) {
                    auto [dummy, face_L_raw] = m_reconstructor->reconstruct_interface(grid, i, j, k, 2);
                    auto [face_R_raw, dummy2] = m_reconstructor->reconstruct_interface(grid, i, j, k + 1, 2);
                    State face_L = rotate(face_L_raw);
                    State face_R = rotate(face_R_raw);

                    Flux F_L = state_to_flux(face_L);
                    Flux F_R = state_to_flux(face_R);

                    double factor = 0.5 * dt_dz;
                    auto evolve = [&](double u, double f_l, double f_r) { return u - factor * (f_r - f_l); };
                    
                    Conserved U_L = face_L.to_conserved(1.4);
                    Conserved U_R = face_R.to_conserved(1.4);

                    // Evolve Rotated
                    U_L.rho = evolve(U_L.rho, F_L.rho, F_R.rho);
                    U_L.mom_x = evolve(U_L.mom_x, F_L.mom_x, F_R.mom_x); // Phys Z-mom
                    U_L.mom_z = evolve(U_L.mom_z, F_L.mom_z, F_R.mom_z); // Phys X-mom
                    // ... (rest same as Y loop) ...
                    U_L.mom_y = evolve(U_L.mom_y, F_L.mom_y, F_R.mom_y);
                    U_L.E     = evolve(U_L.E,     F_L.E,     F_R.E);
                    
                    U_R.rho = evolve(U_R.rho, F_L.rho, F_R.rho);
                    U_R.mom_x = evolve(U_R.mom_x, F_L.mom_x, F_R.mom_x);
                    U_R.mom_z = evolve(U_R.mom_z, F_L.mom_z, F_R.mom_z);
                    U_R.mom_y = evolve(U_R.mom_y, F_L.mom_y, F_R.mom_y);
                    U_R.E     = evolve(U_R.E,     F_L.E,     F_R.E);

                    evolved_L[k] = State::from_conserved(U_L.rho, U_L.mom_x, U_L.mom_y, U_L.mom_z, U_L.E, 1.4);
                    evolved_R[k] = State::from_conserved(U_R.rho, U_R.mom_x, U_R.mom_y, U_R.mom_z, U_R.E, 1.4);
                }

                // PASS 2
                for (int k = 0; k < nz + 1; ++k) {
                    State L = (k==0) ? rotate(m_reconstructor->reconstruct_interface(grid, i, j, k, 2).first) : evolved_R[k - 1];
                    State R = (k==nz) ? rotate(m_reconstructor->reconstruct_interface(grid, i, j, k, 2).second) : evolved_L[k];

                    Flux F_rot = m_riemann_solver->solve(L, R);

                    // UN-ROTATE: Swap mom_x and mom_z
                    Flux F_final = F_rot;
                    std::swap(F_final.mom_x, F_final.mom_z);

                    if (k > 0)  grid.apply_flux(i, j, k - 1, F_final, dt_dz);
                    if (k < nz) grid.apply_flux(i, j, k,     F_final, -dt_dz);
                }
            }
        }
    }
}