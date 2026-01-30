#pragma once
#include "TimeIntegrator.hpp"

namespace fluxus {

    class GodunovIntegrator : public TimeIntegrator {
    public:
        using TimeIntegrator::TimeIntegrator; // Inherit constructor

        // Main stepping function
        void step(Grid& grid, double dt) override;

        // Compute stable timestep based on CFL condition
        double compute_dt(const Grid& grid, double cfl);

        // Setter for gravity (Simple Y-direction gravity for now)
        void set_gravity(double g_y) { m_gravity_y = g_y; }

    private:
        // Store gravity acceleration (default 0.0)
        double m_gravity_y = 0.0;

        // Dimensional Sweeps
        void sweep_x(Grid& grid, double dt);
        void sweep_y(Grid& grid, double dt);
        void sweep_z(Grid& grid, double dt);

        // NEW: Source Term Helper
        void apply_sources(Grid& grid, double dt);
    };

}