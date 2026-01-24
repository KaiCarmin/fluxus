import numpy as np

class GodunovScheme:
    """ 1st order 1D planar Godunov scheme for solving hyperbolic PDEs using a riemann solver. """
    def __init__(self, riemann_solver, nx, L=1, gamma=1.4, cfl=0.5, use_numba=False):
        self.riemann_solver = riemann_solver  # HLL or HLLC solver
        self.nx = nx
        self.L = L
        self.cfl = cfl
        self.gamma = gamma
        self.use_numba = use_numba

        self.dx = self.L / self.nx
        self.S_max = np.inf # Maximum wave speed

        # U stores conserved variables: [rho, rho*u, E] for each cell
        self.U = np.zeros((nx, 3)) 
        # Grid cell centers (for plotting or initial conditions)
        self.x_centers = np.linspace(self.dx / 2, self.L - self.dx / 2, self.nx)    

    def _primitive_to_conserved(self, rho, u, p):
        """Converts primitive (rho, u, p) to conserved [rho, rho*u, E]."""
        rho_u = rho * u
        E = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        return np.array([rho, rho_u, E])
    
    def _conserved_to_primitive(self, U_vec):
        """Converts conserved [rho, rho*u, E] to primitive (rho, u, p)."""
        rho = U_vec[0]
        if rho < 1e-9: # Avoid division by zero for vacuum or near-vacuum
            return np.array([rho, 0.0, 0.0]) # Or handle as error
        rho_u = U_vec[1]
        E = U_vec[2]
        
        u = rho_u / rho
        p = (self.gamma - 1.0) * (E - 0.5 * rho * u**2)
        if p < 1e-9: # Floor pressure if it becomes negative due to numerical error
            p = 1e-9
        return np.array([rho, u, p])
    
    def initialize_from_primitive(self, initial_condition_func):
        for i in range(self.nx):
            rho_init, u_init, p_init = initial_condition_func(self.x_centers[i])
            self.U[i, :] = self._primitive_to_conserved(rho_init, u_init, p_init)

    def _apply_boundary_conditions(self, U_all_cells, bc_type='reflective'):
        """
        Applies boundary conditions based on the specified type.
        U_all_cells is the interior solution array.
        Returns a padded array with ghost cells.
        
        Args:
            U_all_cells: Interior cell states
            bc_type: 'reflective' for rigid walls, 'periodic' for periodic boundaries
        """
        U_padded = np.zeros((self.nx + 2, 3))
        U_padded[1:-1, :] = U_all_cells 
        
        if bc_type == 'periodic':
            # Periodic boundary conditions
            # Left ghost cell gets data from rightmost interior cell
            U_padded[0, :] = U_all_cells[-1, :]
            # Right ghost cell gets data from leftmost interior cell  
            U_padded[-1, :] = U_all_cells[0, :]
            
        else:  # bc_type == 'reflective' (default)
            # Left rigid wall (ghost cell at index 0)
            rho_L_interior, u_L_interior, p_L_interior = self._conserved_to_primitive(U_all_cells[0, :])
            rho_L_ghost = rho_L_interior
            u_L_ghost = -u_L_interior # Reflect velocity
            p_L_ghost = p_L_interior
            U_padded[0, :] = self._primitive_to_conserved(rho_L_ghost, u_L_ghost, p_L_ghost)
            
            # Right rigid wall (ghost cell at index -1, or nx+1)
            rho_R_interior, u_R_interior, p_R_interior = self._conserved_to_primitive(U_all_cells[-1, :])
            rho_R_ghost = rho_R_interior
            u_R_ghost = -u_R_interior # Reflect velocity
            p_R_ghost = p_R_interior
            U_padded[-1, :] = self._primitive_to_conserved(rho_R_ghost, u_R_ghost, p_R_ghost)
        
        return U_padded

    def _calculate_dt(self):
        """Calculate stable time step dt based on CFL condition."""
        max_s = 1e-9  # To avoid division by zero if all speeds are zero
        for i in range(self.nx):
            rho, u, p = self._conserved_to_primitive(self.U[i, :])
            if rho > 1e-9: # Avoid issues with vacuum
                max_s = max(max_s, abs(u) + np.sqrt(self.gamma * p / rho))
        
        return self.cfl * self.dx / max_s

    def step(self, dt, bc_type='reflective'):
        """ Perform a single time step. """
        U_old = np.copy(self.U)
        U_padded = self._apply_boundary_conditions(U_old, bc_type)
        fluxes = np.zeros((self.nx + 1, 3)) 

        for j in range(self.nx + 1): 
            W_L = self._conserved_to_primitive(U_padded[j, :])
            W_R = self._conserved_to_primitive(U_padded[j+1, :])
            
            solver = self.riemann_solver(W_L, W_R, self.gamma, use_numba=self.use_numba)
            F_interface, _ = solver.solve()
            fluxes[j, :] = F_interface
        
        for i in range(self.nx):
            self.U[i, :] = U_old[i, :] - (dt / self.dx) * (fluxes[i+1, :] - fluxes[i, :])

    def solve(self, total_time):
        """ Solve the hyperbolic PDE until the total time is reached. """
        t = 0.0
        while t < total_time:
            dt = self._calculate_dt()
            if t + dt > total_time:
                dt = total_time - t 
            
            if dt <= 1e-12: # Adjusted threshold for very small dt
                print("Time step too small, stopping.")
                break
                
            self.step(dt)
            t += dt

        print(f"Simulation completed: t = {t:.4f}")

        # Convert final U to primitive variables for returning
        rho_final = np.zeros(self.nx)
        u_final = np.zeros(self.nx)
        p_final = np.zeros(self.nx)
        for i in range(self.nx):
            prim_vars = self._conserved_to_primitive(self.U[i, :])
            rho_final[i] = prim_vars[0]
            u_final[i] = prim_vars[1]
            p_final[i] = prim_vars[2]
            
        return t, rho_final, u_final, p_final # Return final time and primitive variables


