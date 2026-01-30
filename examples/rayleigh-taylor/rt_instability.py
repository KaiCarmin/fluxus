import numpy as np
import matplotlib.pyplot as plt
from fluxus.simulation import Simulation, BoundaryType
from fluxus.core import State

# --- CONFIGURATION ---
NX, NY = 100, 300   # Resolution
DT = 0.0025
STEPS = 1000
PLOT_EVERY = 20
GAMMA = 1.4

# 1. Setup Simulation (using your new class)
sim = Simulation(nx=NX, ny=NY, extent_x=0.25, extent_y=0.75, gamma=GAMMA)

# 2. Set Boundaries
# Reflective on top/bottom (walls), Transmissive on sides (or Periodic)
# For RT, Reflective on Top/Bottom is critical so the fluid doesn't fall out.
sim.set_boundaries(
    BoundaryType.Reflective, BoundaryType.Reflective, # X-Left, X-Right (Wall)
    BoundaryType.Reflective, BoundaryType.Reflective  # Y-Bottom, Y-Top (Wall)
)

# 3. Enable Gravity
sim.set_gravity(-0.1)

# 4. Initial Conditions (Using the helper function logic)
def rt_initial_condition(x, y):
    # Heavy fluid on top (y > 0.375)
    if y > 0.375:
        rho = 2.0
    else:
        rho = 1.0
        
    # Hydrostatic Pressure: P = P0 + rho * g * (y - interface)
    # P0 = 2.5 at interface (y=0.375)
    # g = -0.1 (gravity is negative, so P decreases as y increases)
    P0 = 2.5
    g = -0.1
    dist = y - 0.375
    
    # Fundamental Hydrostatic Balance: dP/dy = rho * g
    # P(y) = P0 + rho * g * dist
    # Note: Since g is negative, pressure drops as we go up. Correct.
    p = P0 + rho * g * dist
    
    # Velocity Perturbation (The Spark)
    v = 0.0
    if abs(dist) < 0.02:
        # Wiggle the velocity to trigger instability
        v = 0.01 * (1.0 + np.cos(4 * np.pi * x / 0.25))
        
    return State(rho, 0.0, v, p)

sim.set_initial_condition(rt_initial_condition)

# 5. Run & Visualize
print("Starting Rayleigh-Taylor Simulation...")
fig = plt.figure(figsize=(6, 10))
ax = fig.add_subplot(111)

# Initial Plot
img = ax.imshow(sim.density, origin='lower', cmap='RdBu_r', 
                extent=[0, 0.25, 0, 0.75], vmin=0.9, vmax=2.1)
plt.colorbar(img, label='Density')
ax.set_title("Init")

try:
    for step in range(STEPS):
        sim.step(DT)
        
        if step % PLOT_EVERY == 0:
            print(f"Step {step}/{STEPS}")
            img.set_data(sim.density)
            ax.set_title(f"Rayleigh-Taylor (Step {step})")
            plt.pause(0.001)

except KeyboardInterrupt:
    print("Simulation stopped by user.")

plt.show()