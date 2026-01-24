import numpy as np
import csv
import time
from numba_accelrated import (conserved_to_primitive_numba, calculate_metrics_numba, calculate_dt_numba, 
                             calculate_bubble_spike_heights_numba, apply_gravity_numba, 
                             calculate_vorticity_field_numba, primitive_to_conserved_numba)
from godunov_scheme import GodunovScheme

class Godunov2D:
    """
    A 2D Godunov scheme solver using Strang splitting.
    It uses a 1D Godunov scheme for sweeps in x and y directions.
    """
    def __init__(self, riemann_solver, nx, ny, Lx=1, Ly=2, gamma=1.4, cfl=0.4, gravity=0.0, use_numba=False):
        self.use_numba = use_numba
        self.riemann_solver = riemann_solver
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.gamma = gamma
        self.cfl = cfl
        self.g = gravity

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

        self.U = np.zeros((self.nx, self.ny, 4))
        
        self.x_centers = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.nx)
        self.y_centers = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.ny)
        self.iteration = 0
        self.metrics_file = None
        self.csv_writer = None
        self.previous_total_energy = 0.0
        self.step_start_time = None

    def _primitive_to_conserved(self, rho, u, v, p):
        """Converts primitive variables to conserved variables."""
        if self.use_numba:
            return primitive_to_conserved_numba(rho, u, v, p, self.gamma)
        
        rho_u = rho * u
        rho_v = rho * v
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho_u, rho_v, E])

    def _conserved_to_primitive(self, U_vec):
        """Converts a vector of conserved variables to primitive variables."""
        if self.use_numba:
            return conserved_to_primitive_numba(U_vec, self.gamma)
        
        rho = U_vec[0]
        if rho < 1e-9:
            return np.array([rho, 0.0, 0.0, 0.0])
        rho_u = U_vec[1]
        rho_v = U_vec[2]
        E = U_vec[3]
        
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
        if p < 1e-9:
            p = 1e-9
        return np.array([rho, u, v, p])

    def initialize_metrics_logging(self, filename):
        """Initialize CSV file for metrics logging."""
        self.metrics_file = open(filename, 'w', newline='')
        fieldnames = [
            'Iteration', 'Time', 'TimeStep', 'MaxVelocity', 'MaxSoundSpeed',
            'KineticEnergy', 'PotentialEnergy', 'TotalEnergy', 'EnergyChange',
            'EnergyChangePercent', 'MaxDensity', 'MinDensity', 'MaxPressure',
            'MinPressure', 'MaxVorticity', 'AvgTemperature', 'ActualCFL',
            'BubbleHeight', 'SpikeHeight', 'MixingWidth',
            'ExecutionTime'
        ]
        self.csv_writer = csv.DictWriter(self.metrics_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.metrics_file.flush()

    def _calculate_metrics_numba(self, t, dt, execution_time):
        """Calculate simulation metrics using Numba-accelerated functions."""
        if not self.use_numba:
            return self.calculate_metrics(t, dt, execution_time)
        
        metrics = {}
        
        metrics['Iteration'] = self.iteration
        metrics['Time'] = t
        metrics['TimeStep'] = dt
        metrics['ExecutionTime'] = execution_time
        (max_velocity, max_sound_speed, kinetic_energy, potential_energy,
         total_energy, max_density, min_density, max_pressure,
         min_pressure, max_vorticity, avg_temperature) = calculate_metrics_numba(
            self.U, self.dx, self.dy, self.gamma, self.g, self.y_centers
        )
        
        metrics['MaxVelocity'] = max_velocity
        metrics['MaxSoundSpeed'] = max_sound_speed
        metrics['KineticEnergy'] = kinetic_energy
        metrics['PotentialEnergy'] = potential_energy
        metrics['TotalEnergy'] = total_energy
        
        energy_change = total_energy - self.previous_total_energy
        metrics['EnergyChange'] = energy_change
        if self.previous_total_energy != 0:
            metrics['EnergyChangePercent'] = (energy_change / self.previous_total_energy) * 100
        else:
            metrics['EnergyChangePercent'] = 0.0
        self.previous_total_energy = total_energy
        
        metrics['MaxDensity'] = max_density
        metrics['MinDensity'] = min_density
        metrics['MaxPressure'] = max_pressure
        metrics['MinPressure'] = min_pressure
        metrics['MaxVorticity'] = max_vorticity
        metrics['AvgTemperature'] = avg_temperature
        
        max_signal_speed = max(
            max_velocity + max_sound_speed if max_velocity and max_sound_speed else 0.0,
            1e-9
        )
        metrics['ActualCFL'] = max_signal_speed * dt / min(self.dx, self.dy)
        
        bubble_height, spike_height, mixing_width = calculate_bubble_spike_heights_numba(
            self.U, self.gamma, self.y_centers, self.Ly)
        metrics['BubbleHeight'] = bubble_height
        metrics['SpikeHeight'] = spike_height
        metrics['MixingWidth'] = mixing_width
        
        return metrics

    def calculate_metrics(self, t, dt, execution_time):
        """Calculate all simulation metrics."""
        if self.use_numba:
            return self._calculate_metrics_numba(t, dt, execution_time)

        metrics = {}
        
        # Basic simulation info
        metrics['Iteration'] = self.iteration
        metrics['Time'] = t
        metrics['TimeStep'] = dt
        metrics['ExecutionTime'] = execution_time
        
        # Initialize arrays for calculations
        velocities = []
        sound_speeds = []
        densities = []
        pressures = []
        temperatures = []
        kinetic_energy = 0.0
        potential_energy = 0.0
        internal_energy = 0.0
        vorticity_values = []
        

        for i in range(self.nx):
            for j in range(self.ny):
                rho, u, v, p = self._conserved_to_primitive(self.U[i, j, :])
                
                # Velocity and sound speed
                vel_mag = np.sqrt(u**2 + v**2)
                velocities.append(vel_mag)
                c = np.sqrt(self.gamma * p / rho) if rho > 1e-9 else 0.0
                sound_speeds.append(c)
                
                # Density and pressure
                densities.append(rho)
                pressures.append(p)
                
                # Temperature (assuming ideal gas: T = p/(rho*R), using p/rho as proxy)
                temp = p / rho if rho > 1e-9 else 0.0
                temperatures.append(temp)
                
                # Energy calculations (per unit volume, then integrate)
                cell_volume = self.dx * self.dy
                kinetic_energy += 0.5 * rho * (u**2 + v**2) * cell_volume
                potential_energy += rho * abs(self.g) * self.y_centers[j] * cell_volume
                internal_energy += p / (self.gamma - 1.0) * cell_volume
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                _, u_right, v_right, _ = self._conserved_to_primitive(self.U[i+1, j, :])
                _, u_left, v_left, _ = self._conserved_to_primitive(self.U[i-1, j, :])
                _, u_up, v_up, _ = self._conserved_to_primitive(self.U[i, j+1, :])
                _, u_down, v_down, _ = self._conserved_to_primitive(self.U[i, j-1, :])
                
                dvdx = (v_right - v_left) / (2 * self.dx)
                dudy = (u_up - u_down) / (2 * self.dy)
                vorticity = dvdx - dudy
                vorticity_values.append(abs(vorticity))
        
        # Fill metrics dictionary
        metrics['MaxVelocity'] = max(velocities) if velocities else 0.0
        metrics['MaxSoundSpeed'] = max(sound_speeds) if sound_speeds else 0.0
        metrics['KineticEnergy'] = kinetic_energy
        metrics['PotentialEnergy'] = potential_energy
        metrics['TotalEnergy'] = kinetic_energy + potential_energy + internal_energy
        
        # Energy change calculations
        energy_change = metrics['TotalEnergy'] - self.previous_total_energy
        metrics['EnergyChange'] = energy_change
        if self.previous_total_energy != 0:
            metrics['EnergyChangePercent'] = (energy_change / self.previous_total_energy) * 100
        else:
            metrics['EnergyChangePercent'] = 0.0
        self.previous_total_energy = metrics['TotalEnergy']
        
        metrics['MaxDensity'] = max(densities) if densities else 0.0
        metrics['MinDensity'] = min(densities) if densities else 0.0
        metrics['MaxPressure'] = max(pressures) if pressures else 0.0
        metrics['MinPressure'] = min(pressures) if pressures else 0.0
        metrics['MaxVorticity'] = max(vorticity_values) if vorticity_values else 0.0
        metrics['AvgTemperature'] = np.mean(temperatures) if temperatures else 0.0
        
        # Calculate actual CFL number achieved
        max_signal_speed = max(
            max(velocities) + max(sound_speeds) if velocities and sound_speeds else 0.0,
            1e-9
        )
        metrics['ActualCFL'] = max_signal_speed * dt / min(self.dx, self.dy)
        
        # Calculate bubble and spike heights for Rayleigh-Taylor analysis
        bubble_height, spike_height, mixing_width = self.calculate_bubble_spike_heights()
        metrics['BubbleHeight'] = bubble_height
        metrics['SpikeHeight'] = spike_height
        metrics['MixingWidth'] = mixing_width
        
        return metrics

    def calculate_vorticity_field(self):
        """
        Calculate the full 2D vorticity field.
        
        Returns:
            np.ndarray: 2D array of vorticity values (nx, ny)
                       ω = ∂v/∂x - ∂u/∂y (positive = counterclockwise rotation)
        """
        if self.use_numba:
            return calculate_vorticity_field_numba(self.U, self.dx, self.dy, self.gamma)
        
        vorticity_field = np.zeros((self.nx, self.ny))
        
        # Use central differences for interior points
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                _, u_right, v_right, _ = self._conserved_to_primitive(self.U[i+1, j, :])
                _, u_left, v_left, _ = self._conserved_to_primitive(self.U[i-1, j, :])
                _, u_up, v_up, _ = self._conserved_to_primitive(self.U[i, j+1, :])
                _, u_down, v_down, _ = self._conserved_to_primitive(self.U[i, j-1, :])
                
                dvdx = (v_right - v_left) / (2 * self.dx)
                dudy = (u_up - u_down) / (2 * self.dy)
                vorticity_field[i, j] = dvdx - dudy
        

        for j in range(1, self.ny-1):
            _, u_center, v_center, _ = self._conserved_to_primitive(self.U[0, j, :])
            _, u_right, v_right, _ = self._conserved_to_primitive(self.U[1, j, :])
            _, u_up, v_up, _ = self._conserved_to_primitive(self.U[0, j+1, :])
            _, u_down, v_down, _ = self._conserved_to_primitive(self.U[0, j-1, :])
            
            dvdx = (v_right - v_center) / self.dx
            dudy = (u_up - u_down) / (2 * self.dy)
            vorticity_field[0, j] = dvdx - dudy
        

        for j in range(1, self.ny-1):
            _, u_center, v_center, _ = self._conserved_to_primitive(self.U[-1, j, :])
            _, u_left, v_left, _ = self._conserved_to_primitive(self.U[-2, j, :])
            _, u_up, v_up, _ = self._conserved_to_primitive(self.U[-1, j+1, :])
            _, u_down, v_down, _ = self._conserved_to_primitive(self.U[-1, j-1, :])
            
            dvdx = (v_center - v_left) / self.dx
            dudy = (u_up - u_down) / (2 * self.dy)
            vorticity_field[-1, j] = dvdx - dudy
        

        for i in range(1, self.nx-1):
            _, u_center, v_center, _ = self._conserved_to_primitive(self.U[i, 0, :])
            _, u_right, v_right, _ = self._conserved_to_primitive(self.U[i+1, 0, :])
            _, u_left, v_left, _ = self._conserved_to_primitive(self.U[i-1, 0, :])
            _, u_up, v_up, _ = self._conserved_to_primitive(self.U[i, 1, :])
            
            dvdx = (v_right - v_left) / (2 * self.dx)  # Central difference
            dudy = (u_up - u_center) / self.dy
            vorticity_field[i, 0] = dvdx - dudy
        

        for i in range(1, self.nx-1):
            _, u_center, v_center, _ = self._conserved_to_primitive(self.U[i, -1, :])
            _, u_right, v_right, _ = self._conserved_to_primitive(self.U[i+1, -1, :])
            _, u_left, v_left, _ = self._conserved_to_primitive(self.U[i-1, -1, :])
            _, u_down, v_down, _ = self._conserved_to_primitive(self.U[i, -2, :])
            
            dvdx = (v_right - v_left) / (2 * self.dx)  # Central difference
            dudy = (u_center - u_down) / self.dy
            vorticity_field[i, -1] = dvdx - dudy
        

        vorticity_field[0, 0] = vorticity_field[1, 1]      # Bottom-left
        vorticity_field[0, -1] = vorticity_field[1, -2]    # Top-left  
        vorticity_field[-1, 0] = vorticity_field[-2, 1]    # Bottom-right
        vorticity_field[-1, -1] = vorticity_field[-2, -2]  # Top-right
        
        return vorticity_field

    def calculate_bubble_spike_heights(self):
        """
        Calculate bubble height (light fluid penetration upward) and 
        spike height (heavy fluid penetration downward).
        
        Method: Interface tracking using density threshold
        - Bubble height: maximum y_phys where light fluid dominates
        - Spike height: minimum y_phys where heavy fluid dominates
        - Uses middle density as interface criterion
        
        Returns:
            tuple: (bubble_height, spike_height, mixing_width)
                   Heights in physical coordinates (centered around y=0)
        """
        # if self.use_numba:
        #     return calculate_bubble_spike_heights_numba(self.U, self.gamma, self.y_centers, self.Ly)

        # Get density field
        density_field = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                rho, _, _, _ = self._conserved_to_primitive(self.U[i, j, :])
                density_field[i, j] = rho
        
        # Determine density thresholds
        rho_max = np.max(density_field)  # Heavy fluid density
        rho_min = np.min(density_field)  # Light fluid density
        rho_interface = 0.5 * (rho_max + rho_min)  # Interface density
        
        # Convert y grid coordinates to physical coordinates (centered around y=0)
        y_phys = self.y_centers - self.Ly/2  # Map grid [0,Ly] to physical [-Ly/2, Ly/2]
        
        # Find bubble height (maximum y_phys where light fluid exists)
        bubble_height = -self.Ly/2  # Initialize to bottom of domain
        for j in range(self.ny):
            # Check if any cell at this y-level has light fluid (below interface density)
            if np.any(density_field[:, j] < rho_interface):
                bubble_height = max(bubble_height, y_phys[j])
        
        # Find spike height (minimum y_phys where heavy fluid exists)  
        spike_height = self.Ly/2   # Initialize to top of domain
        for j in range(self.ny):
            # Check if any cell at this y-level has heavy fluid (above interface density)
            if np.any(density_field[:, j] > rho_interface):
                spike_height = min(spike_height, y_phys[j])
        
        # Calculate mixing width (total extent of mixing zone)
        mixing_width = bubble_height - spike_height
        
        # Ensure reasonable values (handle edge cases)
        if bubble_height <= -self.Ly/2:
            bubble_height = 0.0  # No bubble penetration detected
        if spike_height >= self.Ly/2:
            spike_height = 0.0   # No spike penetration detected
        if mixing_width < 0:
            mixing_width = 0.0   # Ensure positive mixing width
            
        return bubble_height, spike_height, mixing_width

    def log_metrics(self, metrics):
        """Write metrics to CSV file."""
        if self.csv_writer:
            self.csv_writer.writerow(metrics)
            self.metrics_file.flush()

    def close_metrics_logging(self):
        """Close the metrics file."""
        if self.metrics_file:
            self.metrics_file.close()
            self.metrics_file = None
            self.csv_writer = None

    def initialize(self, initial_condition_func):
        """Initializes the grid from a function that returns primitive variables."""
        for i in range(self.nx):
            for j in range(self.ny):
                rho, u, v, p = initial_condition_func(self.x_centers[i], self.y_centers[j])
                self.U[i, j, :] = self._primitive_to_conserved(rho, u, v, p)

    def _apply_gravity(self, dt, g):
        """Applies the gravity source term."""
        if self.use_numba:
            apply_gravity_numba(self.U, dt, g, self.gamma)
        else:
            for i in range(self.nx):
                for j in range(self.ny):
                    rho, u, v, p = self._conserved_to_primitive(self.U[i, j, :])
                    # Update momentum in y-dir
                    v_new = v + dt * g
                    # Update total energy: dE/dt = rho * v * g (work done by gravity)
                    self.U[i, j, 2] = rho * v_new  # rho*v
                    self.U[i, j, 3] += dt * rho * v * g  # Energy: E += dt * rho * v * g
    
    def _calculate_dt(self, g):
        """Calculates the stable time step."""
        if self.use_numba:
            return calculate_dt_numba(self.U, self.dx, self.dy, self.cfl, self.gamma, g)
        
        max_s_x = 1e-9
        max_s_y = 1e-9
        for i in range(self.nx):
            for j in range(self.ny):
                rho, u, v, p = self._conserved_to_primitive(self.U[i, j, :])
                if rho > 1e-9:
                    c = np.sqrt(self.gamma * p / rho)
                    max_s_x = max(max_s_x, abs(u) + c)
                    max_s_y = max(max_s_y, abs(v) + c)
        
        dt_x = self.dx / max_s_x
        dt_y = self.dy / max_s_y
        
        # Gravity CFL condition
        dt_g = np.sqrt(2 * min(self.dx, self.dy) / abs(g)) if g != 0 else np.inf

        return self.cfl * min(dt_x, dt_y, dt_g)

    def _x_sweep(self, dt):
        """Performs the sweep in the x-direction with periodic boundary conditions."""
        for j in range(self.ny):
            U_1d = np.zeros((self.nx, 3))
            for i in range(self.nx):
                rho, u, v, p = self._conserved_to_primitive(self.U[i, j, :])
                U_1d[i, :] = np.array([rho, rho * u, p / (self.gamma - 1.0) + 0.5 * rho * u**2])

            # Use the 1D Godunov scheme
            godunov_1d = GodunovScheme(self.riemann_solver, self.nx, L=self.Lx, gamma=self.gamma, cfl=self.cfl, use_numba=self.use_numba)
            godunov_1d.U = U_1d
            
            godunov_1d.step(dt, bc_type='periodic')

            # Update the 2D state
            for i in range(self.nx):
                rho_new, u_new, p_new = godunov_1d._conserved_to_primitive(godunov_1d.U[i, :])
                _, _, v_old, _ = self._conserved_to_primitive(self.U[i, j, :])
                self.U[i, j, :] = self._primitive_to_conserved(rho_new, u_new, v_old, p_new)

    def _y_sweep(self, dt):
        """Performs the sweep in the y-direction with reflective boundary conditions."""
        for i in range(self.nx):

            U_1d = np.zeros((self.ny, 3))
            for j in range(self.ny):
                rho, u, v, p = self._conserved_to_primitive(self.U[i, j, :])
                U_1d[j, :] = np.array([rho, rho * v, p / (self.gamma - 1.0) + 0.5 * rho * v**2])

            # Use the 1D Godunov scheme
            godunov_1d = GodunovScheme(self.riemann_solver, self.ny, L=self.Ly, gamma=self.gamma, cfl=self.cfl, use_numba=self.use_numba)
            godunov_1d.U = U_1d
            
            godunov_1d.step(dt, bc_type='reflective')

            # Update the 2D state
            for j in range(self.ny):
                rho_new, v_new, p_new = godunov_1d._conserved_to_primitive(godunov_1d.U[j, :])
                _, u_old, _, _ = self._conserved_to_primitive(self.U[i, j, :])
                self.U[i, j, :] = self._primitive_to_conserved(rho_new, u_old, v_new, p_new)

    def step(self, dt, g):
        """Performs a full time step using Strang splitting."""
        # Strang splitting: 1/2 dt advection, 1 dt gravity, 1/2 dt advection
        # print("Debug: Starting x-sweep (1/4)")
        self._x_sweep(dt / 2.0)
        # print("Debug: Starting y-sweep (1/4)")
        self._y_sweep(dt / 2.0)
        
        # print("Debug: Starting gravity step")
        # Check state before gravity
        rho_min = np.min(self.U[:,:,0])
        rho_max = np.max(self.U[:,:,0])
        # print(f"Debug: Before gravity - rho range: [{rho_min:.6f}, {rho_max:.6f}]")
        
        self._apply_gravity(dt, g)
        
        # Check state after gravity
        rho_min = np.min(self.U[:,:,0])
        rho_max = np.max(self.U[:,:,0])
        # print(f"Debug: After gravity - rho range: [{rho_min:.6f}, {rho_max:.6f}]")
        
        # print("Debug: Starting y-sweep (3/4)")
        self._y_sweep(dt / 2.0)
        # print("Debug: Starting x-sweep (4/4)")
        self._x_sweep(dt / 2.0)
        # print("Debug: Time step complete")

    def solve(self, total_time, g=-0.1, output_interval=0.1, metrics_filename=None):
        """Main simulation loop with optional metrics logging."""
        if metrics_filename:
            self.initialize_metrics_logging(metrics_filename)
        
        t = 0.0
        output_num = 0
        self.iteration = 0
        
        # Save initial state
        self.save_data(f"output_{output_num:04d}.npz", t)
        
        # Log initial metrics
        if metrics_filename:
            self.step_start_time = time.time()
            initial_metrics = self.calculate_metrics(t, 0.0, 0.0)
            self.log_metrics(initial_metrics)
        
        output_num += 1
        next_output_time = output_interval

        while t < total_time:
            step_start = time.time()
            
            dt = self._calculate_dt(g)
            if t + dt > total_time:
                dt = total_time - t
            
            if dt <= 1e-12:
                print("Time step too small, stopping.")
                break
            
            self.step(dt, g)
            t += dt
            self.iteration += 1
            
            step_end = time.time()
            execution_time = step_end - step_start
            
            # Log metrics every step
            if metrics_filename:
                metrics = self.calculate_metrics(t, dt, execution_time)
                self.log_metrics(metrics)
            
            print(f"t = {t:.4f}, dt = {dt:.6f}, iteration = {self.iteration}")

            if t >= next_output_time:
                self.save_data(f"output_{output_num:04d}.npz", t)
                output_num += 1
                next_output_time += output_interval
        
        # Save final state
        if t not in [out[1] for out in self.load_data_list()]:
             self.save_data(f"output_{output_num:04d}.npz", t)

        print(f"Simulation finished at t = {t:.4f}")
        
        # Close metrics logging
        if metrics_filename:
            self.close_metrics_logging()
            print(f"Metrics saved to: {metrics_filename}")

    def save_data(self, filename, t):
        """Saves the simulation state to a file."""
        primitives = np.zeros((self.nx, self.ny, 4))
        for i in range(self.nx):
            for j in range(self.ny):
                primitives[i, j, :] = self._conserved_to_primitive(self.U[i, j, :])
        
        np.savez(filename, U=self.U, primitives=primitives, t=t,
                 nx=self.nx, ny=self.ny, Lx=self.Lx, Ly=self.Ly, gamma=self.gamma)

    def load_data_list(self):
        """Helper to get a list of output files."""
        import glob
        files = sorted(glob.glob("output_*.npz"))
        data_list = []
        for f in files:
            with np.load(f) as data:
                data_list.append((f, data['t']))
        return data_list
