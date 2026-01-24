import numpy as np
from numba_accelrated import hllc_solve_numba

class HLL:
    def __init__(self, w_left, w_right, gamma):
        """
        Initialize the HLL Riemann solver.

        Parameters:
        w_left (tuple): Left state (density, velocity, pressure)
        w_right (tuple): Right state (density, velocity, pressure)
        gamma (float): Adiabatic index
        """
        self.rho_l, self.u_l, self.p_l = w_left
        self.rho_r, self.u_r, self.p_r = w_right
        self.gamma = gamma

        self._calculate_derived_quantities()

        self.s_l = None
        self.s_r = None

    def _calculate_derived_quantities(self):
        """Helper method to compute energies, conserved variables, fluxes, and sound speeds."""
        # Energies
        # E = rho * (p / ((gamma - 1) * rho) + 0.5 * u^2) = p / (gamma - 1) + 0.5 * rho * u^2
        self.E_l = self.p_l / (self.gamma - 1.0) + 0.5 * self.rho_l * self.u_l**2
        self.E_r = self.p_r / (self.gamma - 1.0) + 0.5 * self.rho_r * self.u_r**2

        # Conserved variables U = [rho, rho*u, E]
        self.U_l = np.array([self.rho_l, self.rho_l * self.u_l, self.E_l])
        self.U_r = np.array([self.rho_r, self.rho_r * self.u_r, self.E_r])

        # Flux vectors F(U) = [rho*u, rho*u^2 + p, u*(E+p)]
        self.F_l = np.array([self.rho_l * self.u_l,
                             self.rho_l * self.u_l**2 + self.p_l,
                             self.u_l * (self.E_l + self.p_l)])
        self.F_r = np.array([self.rho_r * self.u_r,
                             self.rho_r * self.u_r**2 + self.p_r,
                             self.u_r * (self.E_r + self.p_r)])
        
        # Sound speeds
        if self.rho_l <= 0 or self.p_l < 0:
            raise ValueError("Invalid left state: density must be positive and pressure non-negative.")
        self.c_l = np.sqrt(self.gamma * self.p_l / self.rho_l)
        
        if self.rho_r <= 0 or self.p_r < 0:
            raise ValueError("Invalid right state: density must be positive and pressure non-negative.")
        self.c_r = np.sqrt(self.gamma * self.p_r / self.rho_r)


    def calculate_wave_speeds(self):
        """ Calculate the wave speeds for the HLL solver.
        This method calculates the left and right wave speeds based on the
        left and right states.
        """
        # Using pre-calculated sound speeds from __init__
        self.s_l = min(self.u_l - self.c_l, self.u_r - self.c_r)
        self.s_r = max(self.u_l + self.c_l, self.u_r + self.c_r)
    

    def _compute_hll_flux(self):
        """Compute the HLL intercell flux."""
        if self.s_l is None or self.s_r is None:
            self._calculate_wave_speeds()

        if self.s_l >= 0:
            return self.F_l
        elif self.s_r <= 0:
            return self.F_r
        else: # S_L < 0 < S_R
            if self.s_r == self.s_l: 
                 # This should ideally not be hit if S_L < 0 < S_R.
                 # If S_L = S_R = 0, it's covered by s_l >= 0.
                 raise ValueError("s_l and s_r are equal in the intermediate HLL flux calculation zone.")

            F_hll = (self.s_r * self.F_l - self.s_l * self.F_r + 
                     self.s_l * self.s_r * (self.U_r - self.U_l)) / (self.s_r - self.s_l)
            return F_hll
        

    def solve(self):
        """
        Solve the Riemann problem using the HLL method.
        """
        self.calculate_wave_speeds()
        flux = self._compute_hll_flux()
        return flux, (self.s_l, self.s_r)
    

class HLLC:
    def __init__(self, w_left, w_right, gamma, use_numba=False):
        """
        Initialize the HLLC Riemann solver.

        Parameters:
        w_left (tuple): Left state (density, velocity, pressure)
        w_right (tuple): Right state (density, velocity, pressure)
        gamma (float): Adiabatic index
        use_numba (bool): Whether to use Numba-accelerated computation
        """
        self.rho_l, self.u_l, self.p_l = w_left
        self.rho_r, self.u_r, self.p_r = w_right
        self.gamma = gamma
        self.use_numba = use_numba

        if not use_numba:
            self._calculate_initial_derived_quantities()

        # Wave speeds and star region properties
        self.s_l = None
        self.s_r = None
        self.s_m = None # S_M or S_star (contact wave speed)
        self.p_star = None

        # Star states and fluxes
        self.U_l_star = None
        self.U_r_star = None
        self.F_l_star = None
        self.F_r_star = None

    def _calculate_initial_derived_quantities(self):
        """Helper method to compute initial energies, conserved variables, fluxes, and sound speeds."""
        # Total energy per unit volume E = p/(gamma-1) + 0.5*rho*u^2
        self.E_l = self.p_l / (self.gamma - 1.0) + 0.5 * self.rho_l * self.u_l**2
        self.E_r = self.p_r / (self.gamma - 1.0) + 0.5 * self.rho_r * self.u_r**2

        # Conserved variables U = [rho, rho*u, E]
        self.U_l = np.array([self.rho_l, self.rho_l * self.u_l, self.E_l])
        self.U_r = np.array([self.rho_r, self.rho_r * self.u_r, self.E_r])

        # Flux vectors F(U) = [rho*u, rho*u^2 + p, u*(E+p)]
        self.F_l = np.array([self.rho_l * self.u_l,
                             self.rho_l * self.u_l**2 + self.p_l,
                             self.u_l * (self.E_l + self.p_l)])
        self.F_r = np.array([self.rho_r * self.u_r,
                             self.rho_r * self.u_r**2 + self.p_r,
                             self.u_r * (self.E_r + self.p_r)])
        
        # Sound speeds
        if self.rho_l <= 1e-9 or self.p_l < 0: # Allow for very small positive density
            raise ValueError(f"Invalid left state: density ({self.rho_l}) must be positive and pressure ({self.p_l}) non-negative.")
        self.c_l = np.sqrt(self.gamma * self.p_l / self.rho_l) if self.p_l >=0 else 0.0
        
        if self.rho_r <= 1e-9 or self.p_r < 0:
            raise ValueError(f"Invalid right state: density ({self.rho_r}) must be positive and pressure ({self.p_r}) non-negative.")
        self.c_r = np.sqrt(self.gamma * self.p_r / self.rho_r) if self.p_r >=0 else 0.0

    def _calculate_wave_speeds_and_star_pressure(self):
        """
        Calculate wave speeds S_L, S_R, S_M (contact) and p_star (pressure in star region).
        Uses simple estimates for S_L and S_R.
        """
        self.s_l = min(self.u_l - self.c_l, self.u_r - self.c_r)
        self.s_r = max(self.u_l + self.c_l, self.u_r + self.c_r)

        # Calculate S_M (contact wave speed S_star)
        numerator = self.p_r - self.p_l + self.rho_l * self.u_l * (self.s_l - self.u_l) - \
                      self.rho_r * self.u_r * (self.s_r - self.u_r)
        denominator = self.rho_l * (self.s_l - self.u_l) - self.rho_r * (self.s_r - self.u_r)
        
        if abs(denominator) < 1e-9: # Avoid division by zero
            if np.allclose(self.U_l, self.U_r):
                self.s_m = self.u_l
            else:
                # If the states are not identical, this indicates a pathological case.
                raise ValueError(f"Denominator for S_M calculation is near zero ({denominator:.2e}) with non-identical states. S_L={self.s_l:.2f}, S_R={self.s_r:.2f}")
        else:
            self.s_m = numerator / denominator
        
        # Calculate p_star (pressure in star region)
        self.p_star = self.p_l + self.rho_l * (self.s_l - self.u_l) * (self.s_m - self.u_l)
        
        if self.p_star < 0:
            raise ValueError(f"Calculated p_star is negative ({self.p_star:.4e}). S_L={self.s_l:.2f}, S_M={self.s_m:.2f}, S_R={self.s_r:.2f}")

    def _calculate_star_states_and_fluxes(self):
        """
        Calculate the conserved variables (U_K_star) and fluxes (F_K_star)
        in the star regions (K=L,R).
        """
        if self.s_m is None or self.p_star is None: # Ensure previous step is done
            self._calculate_wave_speeds_and_star_pressure()

        # Left star region (U_L_star, F_L_star)
        # Density in left star region
        if abs(self.s_l - self.s_m) < 1e-9:
            rho_l_star = self.rho_l * (self.s_l - self.u_l) / (self.s_l - self.s_m)
        else:
            rho_l_star = self.rho_l * (self.s_l - self.u_l) / (self.s_l - self.s_m)

        u_l_star = self.s_m # Velocity in star region is S_M
        E_l_star = self.p_star / (self.gamma - 1.0) + 0.5 * rho_l_star * u_l_star**2
        self.U_l_star = np.array([rho_l_star, rho_l_star * u_l_star, E_l_star])
        
        # Flux in left star region F_L_star
        self.F_l_star = self.F_l + self.s_l * (self.U_l_star - self.U_l)

        # Right star region (U_R_star, F_R_star)
        if abs(self.s_r - self.s_m) < 1e-9:
            rho_r_star = self.rho_r * (self.s_r - self.u_r) / (self.s_r - self.s_m + 1e-9) # Add epsilon
        else:
            rho_r_star = self.rho_r * (self.s_r - self.u_r) / (self.s_r - self.s_m)
        
        u_r_star = self.s_m
        E_r_star = self.p_star / (self.gamma - 1.0) + 0.5 * rho_r_star * u_r_star**2
        self.U_r_star = np.array([rho_r_star, rho_r_star * u_r_star, E_r_star])

        # Flux in right star region F_R_star
        self.F_r_star = self.F_r + self.s_r * (self.U_r_star - self.U_r)

    def _compute_hllc_flux(self):
        """
        Compute the HLLC flux vector based on the wave speeds and star states.
        """
        if self.s_l >= 0:
            return self.F_l
        elif self.s_l <= 0 and self.s_m >= 0:
            return self.F_l_star
        elif self.s_m <= 0 and self.s_r >= 0:
            return self.F_r_star
        elif self.s_r <= 0:
            return self.F_r
        else: 
            # This case should not occur if wave speeds are calculated correctly.
            if not (self.s_l <= self.s_m + 1e-9 and self.s_m <= self.s_r + 1e-9): # Add tolerance for floating point
                 raise ValueError(f"Wave speeds S_L, S_M, S_R are not correctly ordered: {self.s_l:.3f}, {self.s_m:.3f}, {self.s_r:.3f}")
            raise ValueError("HLLC flux conditions not met, unexpected wave speed configuration.")



    def solve(self):
        """
        Solve the Riemann problem using the HLLC method.
        Returns the numerical flux at the interface (x/t=0) and the wave speeds (S_L, S_M, S_R).
        """
        if self.use_numba:
            # Use Numba-accelerated solver
            flux, s_l, s_m, s_r = hllc_solve_numba(
                self.rho_l, self.u_l, self.p_l,
                self.rho_r, self.u_r, self.p_r,
                self.gamma
            )
            
            # Update instance variables for compatibility
            self.s_l, self.s_m, self.s_r = s_l, s_m, s_r
            
            return flux, (s_l, s_m, s_r)
        else:        
            # Use original Python implementation
            self._calculate_initial_derived_quantities()
            self._calculate_wave_speeds_and_star_pressure()
            self._calculate_star_states_and_fluxes()
            flux = self._compute_hllc_flux()
            return flux, (self.s_l, self.s_m, self.s_r)