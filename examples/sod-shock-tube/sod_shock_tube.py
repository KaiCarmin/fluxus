import numpy as np
import matplotlib.pyplot as plt
from fluxus.simulation import Simulation
from fluxus.core import BoundaryType, State

# --- CONFIGURATION ---
NX = 200
DT = 0.002  # Small time step
T_MAX = 0.2 # Standard end time for Sod
GAMMA = 1.4

# 1. Initialize Simulation
# We create a 1D domain (ny=1) from x=0 to x=1.0
sim = Simulation(nx=NX, ny=1, extent_x=1.0, extent_y=0.1, gamma=GAMMA)

# 2. Define Initial Conditions
# Left: High Pressure/Density | Right: Low Pressure/Density
def sod_ic(x, y):
    if x < 0.5:
        # Left State
        return State(rho=1.0, u=0.0, v=0.0, p=1.0)
    else:
        # Right State
        return State(rho=0.125, u=0.0, v=0.0, p=0.1)

sim.set_initial_condition(sod_ic)

# 3. Set Boundaries
# Transmissive (Outflow) lets waves leave the domain without reflecting
sim.set_boundaries(
    BoundaryType.Transmissive, BoundaryType.Transmissive, # X-Left, X-Right
    BoundaryType.Transmissive, BoundaryType.Transmissive  # Y-Top, Y-Bottom (Irrelevant for 1D)
)

# 4. Run Loop
print("Starting Sod Shock Tube Simulation...")
t = 0.0
step = 0

# Arrays to store history for plotting
while t < T_MAX:
    sim.step(DT)
    t += DT
    step += 1
    if step % 10 == 0:
        print(f"Time: {t:.3f} / {T_MAX}")

# 5. Extract Data
# sim.data is a flat 1D array of conserved variables. 
# We need to parse it back to primitives (rho, u, p) for plotting.
# (The Simulation class helper only gave density, let's grab everything manually)

# Reshape to (NY, NX, 5) and strip ghosts
raw = sim.data.reshape(sim.ny + 2*sim.ng, sim.nx + 2*sim.ng, 5)
internal = raw[sim.ng:-sim.ng, sim.ng:-sim.ng, :] # Shape (1, 200, 5)
data_1d = internal[0] # Shape (200, 5)

rho = data_1d[:, 0]
mom_x = data_1d[:, 1]
E = data_1d[:, 4]

# Compute Primitives
u = mom_x / rho
# p = (gamma - 1) * (E - 0.5 * rho * u^2)
kinetic = 0.5 * rho * u**2
p = (GAMMA - 1.0) * (E - kinetic)

x_axis = np.linspace(0, 1.0, NX)

# 6. Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Density
axes[0].plot(x_axis, rho, 'b-', label='Density')
axes[0].set_title("Density")
axes[0].set_ylim(0, 1.1)
axes[0].grid(True)

# Velocity
axes[1].plot(x_axis, u, 'g-', label='Velocity')
axes[1].set_title("Velocity")
axes[1].grid(True)

# Pressure
axes[2].plot(x_axis, p, 'r-', label='Pressure')
axes[2].set_title("Pressure")
axes[2].grid(True)

plt.suptitle(f"Sod Shock Tube (t={t:.2f})")
plt.tight_layout()
plt.show()