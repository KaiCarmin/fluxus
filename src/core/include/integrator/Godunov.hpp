#pragma once
#include "TimeIntegrator.hpp"

namespace fluxus {

    class GodunovIntegrator : public TimeIntegrator {
    public:
        using TimeIntegrator::TimeIntegrator; // Inherit constructor

        void step(Grid& grid, double dt) override;

    private:
        // Dimensional Sweeps
        void sweep_x(Grid& grid, double dt);
        void sweep_y(Grid& grid, double dt);
        void sweep_z(Grid& grid, double dt);
    };

}