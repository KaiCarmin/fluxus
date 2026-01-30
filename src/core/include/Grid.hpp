#pragma once
#include <pybind11/numpy.h>
#include "types.hpp"

namespace py = pybind11;

namespace fluxus {

    class Grid {
    public:
        // Dimensions
        int nx, ny, nz, ndim, ng;
        double dx, dy, dz;

        // Boundary Configuration
        BoundaryType bc_x_min = BoundaryType::Transmissive;
        BoundaryType bc_x_max = BoundaryType::Transmissive;
        BoundaryType bc_y_min = BoundaryType::Transmissive;
        BoundaryType bc_y_max = BoundaryType::Transmissive;
        BoundaryType bc_z_min = BoundaryType::Transmissive;
        BoundaryType bc_z_max = BoundaryType::Transmissive;

        // Constructor
        Grid(py::array_t<double> data, int dim, int _nx, int _ny, int _nz, int _ng, double _dx, double _dy, double _dz)
            : m_data(data), ndim(dim), nx(_nx), ny(_ny), nz(_nz), ng(_ng), dx(_dx), dy(_dy), dz(_dz)
        {
            // Get raw pointer for fast access
            ptr = static_cast<double*>(m_data.request().ptr);            
            
            // Strides for indexing (5 variables per cell)
            stride_x = 5; 
            stride_y = (nx + 2 * ng) * stride_x; 
            stride_z = (ny + 2 * ng) * stride_y;
        }

        // --- Boundary Methods ---
        void set_boundaries(BoundaryType x_min, BoundaryType x_max, BoundaryType y_min, BoundaryType y_max) {
            bc_x_min = x_min; bc_x_max = x_max;
            bc_y_min = y_min; bc_y_max = y_max;
        }

        // Implementation is in Grid.cpp
        void apply_boundaries(); 

        // --- Data Accessors ---
        
        // Read State from the grid
        State get_state(int i, int j, int k = 0) const {
            int idx = get_index(i, j, k);
            
            // Reconstruct State from Conserved Variables
            // Ptr layout: [rho, mom_x, mom_y, mom_z, E]
            return State::from_conserved(
                ptr[idx],     // rho
                ptr[idx + 1], // mom_x
                ptr[idx + 2], // mom_y
                ptr[idx + 3], // mom_z
                ptr[idx + 4], // Total Energy
                1.4           // Gamma (Hardcoded for now, ideal to pass as member)
            );
        }

        // Apply Flux Update (or Source Term)
        // Operation: U_new = U_old - factor * Flux
        void apply_flux(int i, int j, int k, const Flux& f, double factor) {
            int idx = get_index(i, j, k);
            
            ptr[idx]     -= factor * f.rho;
            ptr[idx + 1] -= factor * f.mom_x;
            ptr[idx + 2] -= factor * f.mom_y;
            ptr[idx + 3] -= factor * f.mom_z;
            ptr[idx + 4] -= factor * f.E;
        }

    private:
        py::array_t<double> m_data;
        double* ptr;
        int stride_x, stride_y, stride_z;

        // Flatten 3D index to 1D memory offset
        inline int get_index(int i, int j, int k) const {
            return k * stride_z + (j + ng) * stride_y + (i + ng) * stride_x;
        }

        // Helper for apply_boundaries (Implemented in Grid.cpp)
        void set_ghost_cell(int dst_i, int dst_j, int dst_k, 
                            int src_i, int src_j, int src_k, 
                            BoundaryType type, int axis_normal);
    };
}