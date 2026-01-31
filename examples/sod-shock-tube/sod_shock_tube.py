import numpy as np
import matplotlib.pyplot as plt
from fluxus.simulation import Simulation
from fluxus.core import BoundaryType, State, HLLCSolver, GodunovIntegrator
from fluxus.utils import setup_logger

# Setup logger
logger = setup_logger(level="INFO")

# --- CONFIGURATION ---
NX = 200
DT = 0.002
T_MAX = 0.2
GAMMA = 1.4
CFL = 0.5

logger.info(f"Configuration: NX={NX}, DT={DT}, T_MAX={T_MAX}, GAMMA={GAMMA}, CFL={CFL}")

# 1. Build the Physics Stack (Dependency Injection)
#    We explicitly choose the HLLC Riemann Solver and Godunov Integrator.
logger.info("Building physics stack with HLLC solver and Godunov integrator")
solver = HLLCSolver(GAMMA)
integrator = GodunovIntegrator(solver)

# 2. Initialize Simulation
logger.info(f"Initializing simulation: nx={NX}, ny=1, extent_x=1.0, extent_y=0.1")
sim = Simulation(nx=NX, ny=1, integrator=integrator, extent_x=1.0, extent_y=0.1, cfl=CFL)

# 3. Define Initial Conditions (Sod Shock Tube)
#    Left: High Pressure | Right: Low Pressure
def sod_ic(x, y):
    if x < 0.5:
        return State(rho=1.0, u=0.0, v=0.0, p=1.0)
    else:
        return State(rho=0.125, u=0.0, v=0.0, p=0.1)

logger.info("Setting initial condition: Sod shock tube (discontinuity at x=0.5)")
sim.set_initial_condition(sod_ic)

# 4. Set Boundaries
#    Transmissive (Outflow) allows the waves to exit the domain cleanly.
logger.info("Setting boundaries: Transmissive on all sides")
sim.set_boundaries(
    BoundaryType.Transmissive, BoundaryType.Transmissive, # X-Left, X-Right
    BoundaryType.Transmissive, BoundaryType.Transmissive  # Y-Top, Y-Bottom
)

# 5. Run Loop
logger.info("Starting Sod Shock Tube Simulation...")
t = 0.0
step = 0

# Track history to ensure we hit T_MAX exactly
while t < T_MAX:
    # Use adaptive stepping if you prefer: dt_used = sim.step()
    # But for valid comparison to textbooks, we often force a fixed dt here.
    sim.step(DT)
    t += DT
    step += 1
    
    if step % 20 == 0:
        logger.info(f"Step {step}: t={t:.3f}")

logger.info(f"Simulation completed: {step} steps, final time t={t:.3f}")

# 6. Visualization
logger.info("Generating visualization...")
#    Extract primitive variables from the simulation data
#    (Using the helper property .density is easier, but here we want Velocity/Pressure too)

# Reshape to (NY, NX, 5) and strip ghosts
raw = sim.data.reshape(sim.ny + 2*sim.ng, sim.nx + 2*sim.ng, 5)
internal = raw[sim.ng:-sim.ng, sim.ng:-sim.ng, :] # Shape (1, 200, 5)
data_1d = internal[0] # Shape (200, 5)

rho = data_1d[:, 0]
mom_x = data_1d[:, 1]
E = data_1d[:, 4]

# Compute Primitives
u = mom_x / rho
kinetic = 0.5 * rho * u**2
p = (GAMMA - 1.0) * (E - kinetic)

x_axis = np.linspace(0, 1.0, NX)

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