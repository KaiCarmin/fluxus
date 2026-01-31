import numpy as np
from fluxus.core import Grid, State, BoundaryType
from fluxus.utils import setup_logger

# Setup logger
logger = setup_logger(level="DEBUG")

class Simulation:
    def __init__(self, nx, ny, integrator, extent_x=1.0, extent_y=1.0, ng=2, cfl=0.8):
        """
        The Simulation Runner.
        
        Args:
            nx, ny: Grid resolution
            integrator: A configured TimeIntegrator instance (e.g., GodunovIntegrator)
            extent_x, extent_y: Physical domain size
            ng: Number of ghost cells (must match the integrator's requirement, usually 2)
            cfl: CFL stability factor (default 0.8)
        """
        logger.info(f"Initializing Simulation: nx={nx}, ny={ny}, extent=({extent_x}, {extent_y}), ng={ng}, cfl={cfl}")
        self.nx = nx
        self.ny = ny
        self.ng = ng
        self.cfl = cfl
        
        self.dx = extent_x / nx
        self.dy = extent_y / ny
        logger.debug(f"Grid spacing: dx={self.dx:.6f}, dy={self.dy:.6f}")
        
        # 1. Dependency Injection: The physics engine is passed in
        self.integrator = integrator
        logger.debug(f"Integrator: {type(integrator).__name__}")
        
        # 2. Allocate Unified Memory
        # 5 vars: rho, mom_x, mom_y, mom_z, E
        self.n_total = (ny + 2*ng) * (nx + 2*ng) * 5
        self.data = np.zeros(self.n_total, dtype=np.float64)
        logger.debug(f"Allocated {self.n_total} elements for grid data")
        
        # 3. Create Grid
        # Pass nz=1 for 2D/1D
        self.grid = Grid(self.data, 2, nx, ny, 1, ng, self.dx, self.dy, 1.0)
        logger.debug("Grid created successfully")
        
        # Default Boundaries
        self.grid.set_boundaries(
            BoundaryType.Transmissive, BoundaryType.Transmissive,
            BoundaryType.Transmissive, BoundaryType.Transmissive
        )
        logger.debug("Default boundaries set: Transmissive on all sides")
        
        # 4. Callback System
        self.callbacks = []
        self.step_count = 0
        self.sim_time = 0.0
        logger.info("Simulation initialization complete")

    def add_callback(self, callback):
        """
        Register a function to be called after every step.
        Signature: callback(simulation_instance)
        """
        self.callbacks.append(callback)
        logger.debug(f"Callback registered: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def set_initial_condition(self, func):
        """Apply a function func(x, y) -> State to every cell."""
        logger.info("Setting initial conditions...")
        view = self.data.reshape(self.ny + 2*self.ng, self.nx + 2*self.ng, 5)
        
        for j in range(self.ny):
            y_phys = (j + 0.5) * self.dy
            for i in range(self.nx):
                x_phys = (i + 0.5) * self.dx
                
                s = func(x_phys, y_phys)
                
                rho = s.rho
                u, v, w = s.u, s.v, s.w
                p = s.p
                kinetic = 0.5 * rho * (u**2 + v**2 + w**2)
                internal = p / 1.4 # FIXME: Gamma should technically be passed or stored in State
                E = internal + kinetic
                
                view[j + self.ng, i + self.ng, 0] = rho
                view[j + self.ng, i + self.ng, 1] = rho * u
                view[j + self.ng, i + self.ng, 2] = rho * v
                view[j + self.ng, i + self.ng, 3] = rho * w
                view[j + self.ng, i + self.ng, 4] = E
        
        logger.info(f"Initial conditions applied to {self.nx * self.ny} cells")

    def set_boundaries(self, left, right, bottom, top):
        self.grid.set_boundaries(left, right, bottom, top)
        logger.debug(f"Boundaries set: left={left.name}, right={right.name}, bottom={bottom.name}, top={top.name}")

    def step(self, dt=None):
        """
        Advance the simulation.
        If dt is None, it is calculated automatically using self.cfl.
        """
        # 1. Apply Boundaries
        self.grid.apply_boundaries()
        
        # 2. Determine Time Step
        actual_dt = dt
        if actual_dt is None:
            actual_dt = self.integrator.compute_dt(self.grid, self.cfl)
            logger.debug(f"Auto-computed dt={actual_dt:.6e}")
        else:
            logger.debug(f"Using fixed dt={actual_dt:.6e}")
            
        # 3. Advance Physics
        self.integrator.step(self.grid, actual_dt)
        
        # 4. Update Counters
        self.sim_time += actual_dt
        self.step_count += 1
        logger.debug(f"Step {self.step_count} completed: sim_time={self.sim_time:.6e}")
        
        # 5. Execute Callbacks
        for cb in self.callbacks:
            cb(self)
            
        return actual_dt
        
    @property
    def density(self):
        """Returns density array for visualization (internal domain only)"""
        view = self.data.reshape(self.ny + 2*self.ng, self.nx + 2*self.ng, 5)
        return view[self.ng:-self.ng, self.ng:-self.ng, 0]