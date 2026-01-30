// src/core/include/types.hpp
#pragma once
#include <cmath>

namespace fluxus {

    // 1. The fundamental Math Object
    struct Vector5 {
        // Anonymous union: 'data' and the struct share the same memory.
        union {
            double data[5];
            struct {
                double rho;    // data[0]
                double mom_x;  // data[1]
                double mom_y;  // data[2]
                double mom_z;  // data[3]
                double E;      // data[4]
            };
        };

        // Constructor
        Vector5(double d0=0, double d1=0, double d2=0, double d3=0, double d4=0)
            : rho(d0), mom_x(d1), mom_y(d2), mom_z(d3), E(d4) {}

        // 1. Array Access (for loops)
        double& operator[](int i) { return data[i]; }
        const double& operator[](int i) const { return data[i]; }

        // 2. Math Operators (Vector arithmetic)
        Vector5 operator+(const Vector5& other) const {
            return {rho + other.rho, mom_x + other.mom_x, mom_y + other.mom_y, mom_z + other.mom_z, E + other.E};
        }
        Vector5 operator-(const Vector5& other) const {
            return {rho - other.rho, mom_x - other.mom_x, mom_y - other.mom_y, mom_z - other.mom_z, E - other.E};
        }
        Vector5 operator*(double scalar) const {
            return {rho * scalar, mom_x * scalar, mom_y * scalar, mom_z * scalar, E * scalar};
        }
    };

    // 2. Meaningful Aliases
    // "Flux" and "Conserved" are mathematically the same structure
    using Flux = Vector5;
    using Conserved = Vector5;

    // 3. The Primitive State (what we reconstruct)
    struct State {
        double rho, u, v, w, p;
        
        // ---------- Constructors ----------
        // Default
        State() : rho(0), u(0), v(0), w(0), p(0) {}

        // 2D Constructor (what Python uses): w defaults to 0
        State(double _rho, double _u, double _v, double _p)
            : rho(_rho), u(_u), v(_v), w(0.0), p(_p) {}

        // 3D Constructor (Full control)
        State(double _rho, double _u, double _v, double _w, double _p)
            : rho(_rho), u(_u), v(_v), w(_w), p(_p) {}
        // ------------------------------------------

        double sound_speed(double gamma) const {
            return std::sqrt(gamma * p / rho);
        }

        
        // Convert Primitive -> Conserved (U)
        Conserved to_conserved(double gamma) const {
            double kinetic = 0.5 * rho * (u*u + v*v + w*w);
            double internal_energy = p / (gamma - 1.0);
            return {rho, rho * u, rho * v, rho * w, internal_energy + kinetic};
        }

        // Convert Conserved (U) -> Primitive
        static State from_conserved(double rho, double mom_x, double mom_y, double mom_z, double energy, double gamma) {
            State s;
            s.rho = rho;
            s.u = mom_x / rho;
            s.v = mom_y / rho;
            s.w = mom_z / rho; // <-- Added w
            
            // Kinetic = 0.5 * rho * (u^2 + v^2 + w^2)
            double kinetic = 0.5 * rho * (s.u*s.u + s.v*s.v + s.w*s.w);
            s.p = (energy - kinetic) * (gamma - 1.0);
            return s;
        }

        // Convert Primitive -> Flux (F) in X-direction
        Flux to_flux(double gamma) const {
            double E = to_conserved(gamma).E; // Get Energy
            return {
                rho * u,
                (rho * u * u) + p,
                rho * u * v,
                rho * u * w,
                (E + p) * u
            };
        }
    };
}