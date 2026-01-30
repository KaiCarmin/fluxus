#pragma once
#include "Grid.hpp"

namespace fluxus {

    class SourceTerm {
    public:
        virtual ~SourceTerm() = default;

        // The contract: Modify the grid in-place for a duration 'dt'
        virtual void apply(Grid& grid, double dt) = 0;
    };

}