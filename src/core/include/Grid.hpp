#pragma once
#include <pybind11/numpy.h>
#include "types.hpp"

namespace py = pybind11;

namespace fluxus {

    class Grid {
    public:
        // Constructor wraps existing NumPy data (Zero Copy!)
        Grid(py::array_t<double> data, int nx, int ny, int ghosts) 
            : m_data(data), nx(nx), ny(ny), ng(ghosts) {
            
            // Get raw C-pointer to the data for speed
            ptr = static_cast<double*>(m_data.request().ptr);
            
            // Compute strides (how far to jump in memory to get to the next cell/row)
            stride_x = 4; // 4 variables per cell
            stride_y = (nx + 2 * ng) * 4; 
        }

        // The "Magic" Accessor - Handles 2D indexing mapping to 1D memory
        // Returns a State object from the raw array
        State get_state(int i, int j) const {
            int idx = get_index(i, j);
            // Assuming the array is structured [rho, rhou, rhov, E] (Conserved)
            double rho   = ptr[idx];
            double mom_x = ptr[idx + 1];
            double mom_y = ptr[idx + 2];
            double E     = ptr[idx + 3];
            
            // Convert to primitive for the solver
            return State::from_conserved(rho, mom_x, mom_y, E, 1.4); 
        }

        // Write flux update back to memory
        void apply_flux(int i, int j, const Flux& f, double dt_over_dx) {
            int idx = get_index(i, j);
            ptr[idx]     -= dt_over_dx * f.mass;
            ptr[idx + 1] -= dt_over_dx * f.momentum_x;
            ptr[idx + 2] -= dt_over_dx * f.momentum_y;
            ptr[idx + 3] -= dt_over_dx * f.energy;
        }

    private:
        py::array_t<double> m_data;
        double* ptr;
        int nx, ny, ng;
        int stride_x, stride_y;

        // Flatten 2D (i,j) to 1D index
        inline int get_index(int i, int j) const {
            // Adjust for ghost cells offset
            return (j + ng) * stride_y + (i + ng) * stride_x;
        }
    };
}