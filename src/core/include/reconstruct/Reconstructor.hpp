#pragma once
#include "Grid.hpp"
#include <tuple>

namespace fluxus {

    class Reconstructor {
    public:
        virtual ~Reconstructor() = default;

        // Returns {State_Left, State_Right} at the interface i+1/2
        // We pass (i,j,k) of the LEFT cell. The interface is between i and i+1.
        virtual std::pair<State, State> reconstruct_interface(
            const Grid& grid, int i, int j, int k, int axis
        ) = 0;
    };
}