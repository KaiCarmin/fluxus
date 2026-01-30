#pragma once
#include <pybind11/numpy.h>
#include "types.hpp"

namespace py = pybind11;

namespace fluxus {

    class Grid {
    public:
        // Dimensions must be public for the Integrator to see them
        int nx, ny, nz;
        int ndim; 
        int ng;
        double dx, dy, dz;

        // Constructor
        Grid(py::array_t<double> data, int dim, int _nx, int _ny, int _nz, int _ng, double _dx, double _dy, double _dz)
            : m_data(data), ndim(dim), nx(_nx), ny(_ny), nz(_nz), ng(_ng), dx(_dx), dy(_dy), dz(_dz)
        {
            ptr = static_cast<double*>(m_data.request().ptr);            
            
            // 5 variables per cell [rho, mom_x, mom_y, mom_z, E]
            stride_x = 5; 
            stride_y = (nx + 2 * ng) * stride_x; 
            stride_z = (ny + 2 * ng) * stride_y;
        }

        // Accessor: Now accepts optional 'k' for 3D (default 0 for 2D)
        State get_state(int i, int j, int k = 0) const {
            int idx = get_index(i, j, k);
            
            // Read 5 components (Vector5)
            // Using indices [0]..[4] corresponds to rho, mom_x, mom_y, mom_z, E
            return State::from_conserved(
                ptr[idx],     // rho
                ptr[idx + 1], // mom_x
                ptr[idx + 2], // mom_y
                ptr[idx + 3], // mom_z  <-- NEW
                ptr[idx + 4], // E      <-- Shifted
                1.4           // gamma (should ideally be passed in, but fixed for now)
            ); 
        }

        // Writer: Now writes all 5 components
        void apply_flux(int i, int j, int k, const Flux& f, double dt_over_dx) {
            int idx = get_index(i, j, k);
            
            // Vectorized update if compiler is smart, otherwise manual unroll:
            ptr[idx]     -= dt_over_dx * f.rho;
            ptr[idx + 1] -= dt_over_dx * f.mom_x;
            ptr[idx + 2] -= dt_over_dx * f.mom_y;
            ptr[idx + 3] -= dt_over_dx * f.mom_z; // <-- NEW
            ptr[idx + 4] -= dt_over_dx * f.E;
        }
        
        // Overload for 2D legacy calls (just assumes k=0)
        void apply_flux(int i, int j, const Flux& f, double dt_over_dx) {
            apply_flux(i, j, 0, f, dt_over_dx);
        }

    private:
        py::array_t<double> m_data;
        double* ptr;
        int stride_x, stride_y, stride_z;

        // Flatten 3D (i,j,k) to 1D index
        inline int get_index(int i, int j, int k) const {
            // Adjust for ghost cells offset in all directions
            // If 2D, k=0 and nz=1, so we just assume 'ng' offset in Z is 0 if not used? 
            // For safety in 2D simulations, usually we don't use ghost cells in Z.
            // Simplified logic:
            return k * stride_z + (j + ng) * stride_y + (i + ng) * stride_x;
        }
    };
}