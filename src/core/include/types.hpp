// src/core/include/types.hpp
#pragma once
#include <cmath>

namespace fluxus {

    // 1. The fundamental Math Object
    struct Vector4 {
        double data[4]; // rho, mom_x, mom_y, E

        // Constructor
        Vector4(double d0 = 0, double d1 = 0, double d2 = 0, double d3 = 0) 
            : data{d0, d1, d2, d3} {}

        // Accessors for readability
        double& operator[](int i) { return data[i]; }
        const double& operator[](int i) const { return data[i]; }

        // Math Operators (Essential for Godunov: U - dt/dx * F)
        Vector4 operator+(const Vector4& other) const {
            return {data[0] + other.data[0], data[1] + other.data[1], 
                    data[2] + other.data[2], data[3] + other.data[3]};
        }
        Vector4 operator-(const Vector4& other) const {
            return {data[0] - other.data[0], data[1] - other.data[1], 
                    data[2] - other.data[2], data[3] - other.data[3]};
        }
        Vector4 operator*(double scalar) const {
            return {data[0] * scalar, data[1] * scalar, 
                    data[2] * scalar, data[3] * scalar};
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