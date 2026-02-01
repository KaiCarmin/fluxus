#pragma once
#include "TimeIntegrator.hpp"
#include "reconstruct/Reconstructor.hpp"
#include <memory>
#include <vector>

namespace fluxus {

    class MUSCLHancockIntegrator : public TimeIntegrator {
    public:
        MUSCLHancockIntegrator(std::shared_ptr<RiemannSolver> solver, 
                               std::shared_ptr<Reconstructor> reconstructor)
            : TimeIntegrator(solver), m_reconstructor(reconstructor) {}

        void step(Grid& grid, double dt) override;
        void set_gravity(double g_y) { m_gravity_y = g_y; }
        
        // Re-use the stability calculation from Godunov (copy-paste logic or inheritance)
        // For simplicity, we can declare it here and implement similarly
        double compute_dt(const Grid& grid, double cfl);

    private:
        double m_gravity_y = 0.0;
        std::shared_ptr<Reconstructor> m_reconstructor;

        void sweep_x(Grid& grid, double dt);
        void sweep_y(Grid& grid, double dt);
        void sweep_z(Grid& grid, double dt);
        void apply_sources(Grid& grid, double dt);
        
        // Helper to convert State to Flux vector
        Flux state_to_flux(const State& s) {
            Flux f;
            f.rho   = s.rho * s.u;
            f.mom_x = s.rho * s.u * s.u + s.p;
            f.mom_y = s.rho * s.v * s.u;
            f.mom_z = s.rho * s.w * s.u;
            double E = s.p/(1.4 - 1.0) + 0.5*s.rho*(s.u*s.u + s.v*s.v + s.w*s.w);
            f.E     = s.u * (E + s.p);
            return f;
        }
    };
}