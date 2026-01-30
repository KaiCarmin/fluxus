import numpy as np
from fluxus.core import Grid, State, HLLCSolver, GodunovIntegrator, BoundaryType

class Simulation:
    def __init__(self, nx, ny, extent_x=1.0, extent_y=1.0, ng=2, gamma=1.4):
        self.nx = nx
        self.ny = ny
        self.ng = ng
        self.gamma = gamma
        self.dx = extent_x / nx
        self.dy = extent_y / ny
        
        # 1. Allocate Unified Memory (1D/2D/3D supported)
        # 5 vars: rho, mom_x, mom_y, mom_z, E
        self.n_total = (ny + 2*ng) * (nx + 2*ng) * 5
        self.data = np.zeros(self.n_total, dtype=np.float64)
        
        # 2. Create C++ Grid
        # Pass nz=1 for 2D
        self.grid = Grid(self.data, 2, nx, ny, 1, ng, self.dx, self.dy, 1.0)
        
        # 3. Setup Physics Engine
        self.solver = HLLCSolver(gamma)
        self.integrator = GodunovIntegrator(self.solver)
        
        # Default Boundaries (Transmissive everywhere)
        self.grid.set_boundaries(
            BoundaryType.Transmissive, BoundaryType.Transmissive,
            BoundaryType.Transmissive, BoundaryType.Transmissive
        )

    def set_boundaries(self, left, right, bottom, top):
        """
        Example: sim.set_boundaries(BoundaryType.Periodic, BoundaryType.Periodic, ...)
        """
        self.grid.set_boundaries(left, right, bottom, top)

    def set_gravity(self, gy):
        self.integrator.set_gravity(gy)

    def step(self, dt):
        # IMPORTANT: Apply boundaries BEFORE stepping!
        # This ensures ghost cells have valid data for the fluxes.
        self.grid.apply_boundaries()
        self.integrator.step(self.grid, dt)
        
    @property
    def density(self):
        """Returns density array for visualization (internal domain only)"""
        view = self.data.reshape(self.ny + 2*self.ng, self.nx + 2*self.ng, 5)
        # Slice out ghosts
        return view[self.ng:-self.ng, self.ng:-self.ng, 0]