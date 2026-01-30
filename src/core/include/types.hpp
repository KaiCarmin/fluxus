// src/core/include/types.hpp
#pragma once
#include <cmath>

namespace fluxus {

    // 1. The fundamental Math Object
    struct Vector4 {
        // Anonymous union: 'data' and the struct share the same memory.
        union {
            double data[4];
            struct {
                double rho;    // data[0]
                double mom_x;  // data[1]
                double mom_y;  // data[2]
                double E;      // data[3]
            };
        };

        // Constructor
        Vector4(double d0 = 0, double d1 = 0, double d2 = 0, double d3 = 0) 
            : rho(d0), mom_x(d1), mom_y(d2), E(d3) {}

        // 1. Array Access (for loops)
        double& operator[](int i) { return data[i]; }
        const double& operator[](int i) const { return data[i]; }

        // 2. Math Operators (Vector arithmetic)
        Vector4 operator+(const Vector4& other) const {
            return {rho + other.rho, mom_x + other.mom_x, mom_y + other.mom_y, E + other.E};
        }
        Vector4 operator-(const Vector4& other) const {
            return {rho - other.rho, mom_x - other.mom_x, mom_y - other.mom_y, E - other.E};
        }
        Vector4 operator*(double scalar) const {
            return {rho * scalar, mom_x * scalar, mom_y * scalar, E * scalar};
        }
    };

    // 2. Meaningful Aliases
    // "Flux" and "Conserved" are mathematically the same structure
    using Flux = Vector4;
    using Conserved = Vector4;

    // 3. The Primitive State (what we reconstruct)
    struct State {
        double rho, u, v, p;

        double sound_speed(double gamma) const {
            return std::sqrt(gamma * p / rho);
        }

        // Convert Primitive -> Conserved (U)
        Conserved to_conserved(double gamma) const {
            double kinetic = 0.5 * rho * (u*u + v*v);
            double internal_energy = p / (gamma - 1.0);
            return {rho, rho * u, rho * v, internal_energy + kinetic};
        }

        // Convert Primitive -> Flux (F) in X-direction
        Flux to_flux(double gamma) const {
            double E = to_conserved(gamma)[3]; // Get Energy
            return {
                rho * u,
                (rho * u * u) + p,
                rho * u * v,
                (E + p) * u
            };
        }
    };
}