import numpy as np
from fluxus.core import Grid, State, HLLCSolver, GodunovIntegrator, BoundaryType

class Simulation:
    def __init__(self, nx, ny, extent_x=1.0, extent_y=1.0, ng=2, gamma=1.4, cfl=0.8):
        self.nx = nx
        self.ny = ny
        self.ng = ng
        self.gamma = gamma
        self.dx = extent_x / nx
        self.dy = extent_y / ny
        self.cfl = cfl
        
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
    
    def set_initial_condition(self, func):
        """
        Apply a function func(x, y) -> State to every cell in the domain.
        """
        # Create a view of the full data including ghosts: (NY, NX, 5)
        # We assume 2D layout for simplicity, 1D is just NY=1
        view = self.data.reshape(self.ny + 2*self.ng, self.nx + 2*self.ng, 5)
        
        # Iterate over the INTERNAL domain (skipping ghost cells for now)
        # Ghost cells will be filled by apply_boundaries() later.
        for j in range(self.ny):
            # Physical Y coordinate (Cell Center)
            y_phys = (j + 0.5) * self.dy
            
            for i in range(self.nx):
                # Physical X coordinate (Cell Center)
                x_phys = (i + 0.5) * self.dx
                
                # 1. Call User Function
                # Expects a fluxus.core.State object back
                s = func(x_phys, y_phys)
                
                # 2. Convert to Conserved Variables manually
                # Conserved U = [rho, rho*u, rho*v, rho*w, E]
                rho = s.rho
                u, v, w = s.u, s.v, s.w
                p = s.p
                
                # Kinetic Energy = 0.5 * rho * |V|^2
                kinetic = 0.5 * rho * (u**2 + v**2 + w**2)
                
                # Internal Energy = P / (gamma - 1)
                internal = p / (self.gamma - 1.0)
                
                E = internal + kinetic
                
                # 3. Write to Array
                # We must offset indices by 'ng' to write to the real domain
                ji = j + self.ng
                ii = i + self.ng
                
                view[ji, ii, 0] = rho
                view[ji, ii, 1] = rho * u
                view[ji, ii, 2] = rho * v
                view[ji, ii, 3] = rho * w
                view[ji, ii, 4] = E

    def set_boundaries(self, left, right, bottom, top):
        """
        Example: sim.set_boundaries(BoundaryType.Periodic, BoundaryType.Periodic, ...)
        """
        self.grid.set_boundaries(left, right, bottom, top)

    def set_gravity(self, gy):
        self.integrator.set_gravity(gy)

    def step(self, dt=None):
        """
        Advance the simulation.
        If dt is provided, run for that fixed time (dangerous!).
        If dt is None, calculate it dynamically using CFL.
        Returns the actual dt used.
        """
        # 1. Apply Boundaries first (ghosts needed for dt calculation?)
        # Strictly speaking, CFL only depends on internal cells, 
        # but boundaries ensure safety at edges.
        self.grid.apply_boundaries()
        
        # 2. Determine Time Step
        actual_dt = dt
        if actual_dt is None:
            # Ask C++ to calculate stable dt
            actual_dt = self.integrator.compute_dt(self.grid, self.cfl)
            
        # 3. Advance Physics
        self.integrator.step(self.grid, actual_dt)
        
        return actual_dt
        
    @property
    def density(self):
        """Returns density array for visualization (internal domain only)"""
        view = self.data.reshape(self.ny + 2*self.ng, self.nx + 2*self.ng, 5)
        # Slice out ghosts
        return view[self.ng:-self.ng, self.ng:-self.ng, 0]