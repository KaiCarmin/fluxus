#pragma once
#include "TimeIntegrator.hpp"
#include "reconstruct/PiecewiseConstant.hpp" // Default reconstructor

namespace fluxus {

    class GodunovIntegrator : public TimeIntegrator {
    public:
        // Updated Constructor: Accepts Solver AND Reconstructor
        GodunovIntegrator(std::shared_ptr<RiemannSolver> solver, 
                          std::shared_ptr<Reconstructor> reconstructor)
            : TimeIntegrator(solver), m_reconstructor(reconstructor) {}

        // Keep the old constructor for backward compatibility (defaults to PiecewiseConstant)
        GodunovIntegrator(std::shared_ptr<RiemannSolver> solver)
            : TimeIntegrator(solver) {
            m_reconstructor = std::make_shared<PiecewiseConstantReconstructor>();
        }

        void step(Grid& grid, double dt) override;
        void set_gravity(double g_y) { m_gravity_y = g_y; }
        double compute_dt(const Grid& grid, double cfl);

    private:
        double m_gravity_y = 0.0;
        std::shared_ptr<Reconstructor> m_reconstructor;

        void sweep_x(Grid& grid, double dt);
        void sweep_y(Grid& grid, double dt);
        void sweep_z(Grid& grid, double dt); // If 3D
        void apply_sources(Grid& grid, double dt);
    };
}