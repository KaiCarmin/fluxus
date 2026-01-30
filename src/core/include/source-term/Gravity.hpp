#pragma once
#include "SourceTerm.hpp"

namespace fluxus {

    class Gravity : public SourceTerm {
        double g_x, g_y, g_z;
    public:
        Gravity(double x=0, double y=0, double z=0) 
            : g_x(x), g_y(y), g_z(z) {}

        void apply(Grid& grid, double dt) override {
            #pragma omp parallel for collapse(3)
            for(int k=0; k<grid.nz; ++k) {
                for(int j=0; j<grid.ny; ++j) {
                    for(int i=0; i<grid.nx; ++i) {
                        State s = grid.get_state(i, j, k);
                        
                        // F = m*a -> Momentum Change = rho * g * dt
                        double d_mom_x = s.rho * g_x * dt;
                        double d_mom_y = s.rho * g_y * dt;
                        double d_mom_z = s.rho * g_z * dt;
                        
                        // Work = Force * Velocity -> Energy Change = (rho*g) . v * dt
                        double work = (d_mom_x * s.u) + (d_mom_y * s.v) + (d_mom_z * s.w);

                        // Construct the update vector
                        // Note: Grid::apply_flux usually subtracts ( flux_out - flux_in ).
                        // For sources, we usually ADD. 
                        // So we can pass -1.0 as the scale factor to apply_flux if we want to "add".
                        Flux source_vector;
                        source_vector.rho = 0;
                        source_vector.mom_x = d_mom_x; 
                        source_vector.mom_y = d_mom_y;
                        source_vector.mom_z = d_mom_z;
                        source_vector.E = work;

                        // apply_flux does: ptr -= scale * flux
                        // We want: ptr += source
                        // So: ptr -= (-1.0) * source
                        grid.apply_flux(i, j, k, source_vector, -1.0);
                    }
                }
            }
        }
    };
}