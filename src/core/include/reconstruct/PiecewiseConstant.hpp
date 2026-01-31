#pragma once
#include "Reconstructor.hpp"

namespace fluxus {
    class PiecewiseConstantReconstructor : public Reconstructor {
    public:
        std::pair<State, State> reconstruct_interface(
            const Grid& grid, int i, int j, int k, int axis
        ) override {
            if (axis == 0) {
                return { grid.get_state(i - 1, j, k), grid.get_state(i, j, k) };
            } 
            else if (axis == 1) {
                return { grid.get_state(i, j - 1, k), grid.get_state(i, j, k) };
            } 
            else {
                return { grid.get_state(i, j, k - 1), grid.get_state(i, j, k) };
            }
        }
    };
}