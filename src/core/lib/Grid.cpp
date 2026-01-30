#include "Grid.hpp"

namespace fluxus {

    void Grid::apply_boundaries() {
        // --- X Boundaries ---
        // Iterate over Y and Z faces
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                
                // Left Boundary (X_MIN)
                // Ghosts are at i = -1, -2 ... -ng
                // Real data starts at i = 0
                for (int g = 1; g <= ng; ++g) {
                    int src_i = (bc_x_min == BoundaryType::Periodic) ? (nx - g) : (g - 1);
                    set_ghost_cell(-g, j, k, src_i, j, k, bc_x_min, 0); // 0 = X-axis
                }

                // Right Boundary (X_MAX)
                // Ghosts start at i = nx
                for (int g = 0; g < ng; ++g) {
                    int src_i = (bc_x_max == BoundaryType::Periodic) ? g : (nx - 1 - g);
                    set_ghost_cell(nx + g, j, k, src_i, j, k, bc_x_max, 0);
                }
            }
        }

        // --- Y Boundaries (if 2D/3D) ---
        if (ndim >= 2) {
            for (int k = 0; k < nz; ++k) {
                for (int i = -ng; i < nx + ng; ++i) { // Note: iterate full X range including corners
                    
                    // Bottom (Y_MIN)
                    for (int g = 1; g <= ng; ++g) {
                        int src_j = (bc_y_min == BoundaryType::Periodic) ? (ny - g) : (g - 1);
                        set_ghost_cell(i, -g, k, i, src_j, k, bc_y_min, 1); // 1 = Y-axis
                    }

                    // Top (Y_MAX)
                    for (int g = 0; g < ng; ++g) {
                        int src_j = (bc_y_max == BoundaryType::Periodic) ? g : (ny - 1 - g);
                        set_ghost_cell(i, ny + g, k, i, src_j, k, bc_y_max, 1);
                    }
                }
            }
        }
        
        // Z Boundaries omitted for brevity, but same pattern applies
    }

    void Grid::set_ghost_cell(int dst_i, int dst_j, int dst_k, 
                              int src_i, int src_j, int src_k, 
                              BoundaryType type, int axis_normal) 
    {
        // 1. Copy State
        State s = get_state(src_i, src_j, src_k);

        // 2. Modify based on type
        if (type == BoundaryType::Reflective) {
            // Invert velocity normal to the boundary
            if (axis_normal == 0) s.u = -s.u;
            if (axis_normal == 1) s.v = -s.v;
            if (axis_normal == 2) s.w = -s.w;
        }
        
        // 3. Write back (We need a direct writer since apply_flux doesn't set state directly)
        // We can reuse get_index and pointer math
        int idx = get_index(dst_i, dst_j, dst_k);
        Conserved U = s.to_conserved(1.4); // Assuming gamma=1.4 for storage
        
        ptr[idx]     = U.rho;
        ptr[idx + 1] = U.mom_x;
        ptr[idx + 2] = U.mom_y;
        ptr[idx + 3] = U.mom_z;
        ptr[idx + 4] = U.E;
    }
}