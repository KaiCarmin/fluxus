#pragma once
#include <pybind11/numpy.h>
#include <vector>
#include "types.hpp"

namespace py = pybind11;

namespace fluxus {

    class Grid {
    public:
        // Dimensions
        int nx, ny, nz, ndim, ng;
        double dx, dy, dz;

        // Boundary Configuration (Default to Transmissive)
        BoundaryType bc_x_min = BoundaryType::Transmissive;
        BoundaryType bc_x_max = BoundaryType::Transmissive;
        BoundaryType bc_y_min = BoundaryType::Transmissive;
        BoundaryType bc_y_max = BoundaryType::Transmissive;

        // Constructor
        Grid(py::array_t<double> data, int dim, int _nx, int _ny, int _nz, int _ng, double _dx, double _dy, double _dz)
            : m_data(data), ndim(dim), nx(_nx), ny(_ny), nz(_nz), ng(_ng), dx(_dx), dy(_dy), dz(_dz)
        {
            ptr = static_cast<double*>(m_data.request().ptr);            
            stride_x = 5; 
            stride_y = (nx + 2 * ng) * stride_x; 
            stride_z = (ny + 2 * ng) * stride_y;
        }

        // --- NEW: Setters for Python ---
        void set_boundaries(BoundaryType x_min, BoundaryType x_max, BoundaryType y_min, BoundaryType y_max) {
            bc_x_min = x_min; bc_x_max = x_max;
            bc_y_min = y_min; bc_y_max = y_max;
        }

        // --- NEW: The Heavy Lifter ---
        void apply_boundaries();

        // (Keep get_state, apply_flux, and private members the same)
        // ... 
        State get_state(int i, int j, int k = 0) const { /* ... */ }
        void apply_flux(int i, int j, int k, const Flux& f, double dt_over_dx) { /* ... */ }

    private:
        py::array_t<double> m_data;
        double* ptr;
        int stride_x, stride_y, stride_z;
        
        inline int get_index(int i, int j, int k) const {
            return k * stride_z + (j + ng) * stride_y + (i + ng) * stride_x;
        }
        
        // Helper to copy/invert state
        void set_ghost_cell(int dest_i, int dest_j, int dest_k, 
                            int src_i, int src_j, int src_k, 
                            BoundaryType type, int axis_normal);
    };
}