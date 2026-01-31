import numpy as np
import matplotlib.pyplot as plt

from fluxus.simulation import Simulation, BoundaryType
from fluxus.core import (
    HLLCSolver, 
    GodunovIntegrator, 
    State,
    PiecewiseConstantReconstructor
)

from fluxus.utils import setup_logger

# Setup logger
logger = setup_logger(level="INFO")

# --- CONFIGURATION ---
NX, NY = 100, 300
EXTENT_X = 0.25
EXTENT_Y = 0.75
GAMMA = 1.4
CFL = 0.8
G_Y = -0.1
STEPS = 4000
PLOT_EVERY = 20

logger.info(f"Configuration: NX={NX}, NY={NY}, EXTENT_X={EXTENT_X}, EXTENT_Y={EXTENT_Y}")
logger.info(f"Physics: GAMMA={GAMMA}, G_Y={G_Y}, STEPS={STEPS}")

# 1. Build Physics Stack
logger.info("Building physics stack with HLLC solver and Godunov integrator")
solver = HLLCSolver(GAMMA)
reconstructor = PiecewiseConstantReconstructor()
integrator = GodunovIntegrator(solver, reconstructor)
integrator.set_gravity(G_Y)
logger.debug(f"Gravity set to {G_Y}")

# 2. Initialize Simulation
logger.info(f"Initializing simulation: nx={NX}, ny={NY}, cfl={CFL}")
sim = Simulation(nx=NX, ny=NY, integrator=integrator, 
                 extent_x=EXTENT_X, extent_y=EXTENT_Y, cfl=CFL)

# 3. Boundaries (Reflective Box)
logger.info("Setting boundaries: Reflective on all sides")
sim.set_boundaries(
    BoundaryType.Reflective, BoundaryType.Reflective,
    BoundaryType.Reflective, BoundaryType.Reflective
)

# 4. Initial Condition
def rt_ic(x, y):
    # Heavy fluid (rho=2) on top
    if y > 0.375:
        rho = 2.0
    else:
        rho = 1.0
    
    # Hydrostatic Pressure Equilibrium
    dist = y - 0.375
    p = 2.5 + rho * G_Y * dist
    
    # Velocity Perturbation
    v = 0.0
    if abs(dist) < 0.02:
        v = 0.01 * (1.0 + np.cos(4 * np.pi * x / EXTENT_X))
        
    return State(rho, 0.0, v, p)

logger.info("Setting initial condition: Rayleigh-Taylor instability setup")
sim.set_initial_condition(rt_ic)

# --- ENERGY TRACKING CALLBACK ---
metrics = {"time": [], "total_E": [], "ke": [], "pe": []}

def track_metrics(sim):
    # Run every 10 steps
    if sim.step_count % 10 != 0:
        return

    # 1. Get Data (Strip Ghosts)
    raw = sim.data.reshape(sim.ny + 2*sim.ng, sim.nx + 2*sim.ng, 5)
    inner = raw[sim.ng:-sim.ng, sim.ng:-sim.ng, :]
    
    rho   = inner[:, :, 0]
    mom_x = inner[:, :, 1]
    mom_y = inner[:, :, 2]
    E_tot = inner[:, :, 4] # Total Energy Density (includes Internal + Kinetic)

    # 2. Calculate Energies
    # Kinetic Energy Density = 0.5 * (mx^2 + my^2) / rho
    ke_dens = 0.5 * (mom_x**2 + mom_y**2) / rho
    
    # Potential Energy Density = rho * g * h
    # Height array (y coordinate for every cell)
    heights = np.linspace(sim.dy/2, sim.ny*sim.dy - sim.dy/2, sim.ny)
    # Broadcast (NY,) -> (NY, NX)
    pe_dens = rho * abs(G_Y) * heights[:, np.newaxis]

    # Integrate (Sum * Volume)
    dV = sim.dx * sim.dy
    total_KE = np.sum(ke_dens) * dV
    total_PE = np.sum(pe_dens) * dV
    total_E_sys = np.sum(E_tot) * dV + total_PE # System Energy = Internal + Kinetic + Potential

    # 3. Store
    metrics["time"].append(sim.sim_time)
    metrics["total_E"].append(total_E_sys)
    metrics["ke"].append(total_KE)
    metrics["pe"].append(total_PE)

logger.info("Registering energy tracking callback")
sim.add_callback(track_metrics)

# --- RUN LOOP ---
logger.info("Starting Simulation...")
plt.ion()
fig = plt.figure(figsize=(14, 6))

# Plot 1: Density Map
ax_map = fig.add_subplot(121)
img = ax_map.imshow(sim.density, origin='lower', cmap='RdBu_r', 
                    extent=[0, EXTENT_X, 0, EXTENT_Y], vmin=0.9, vmax=2.1)
fig.colorbar(img, ax=ax_map, label='Density')
ax_map.set_title("Density Field")

# Plot 2: Energy (Dual Axis)
ax_eng = fig.add_subplot(122)
ax_ke = ax_eng.twinx() # Create a second Y-axis for Kinetic Energy

# Lines
line_tot, = ax_eng.plot([], [], 'k-', label='Total Energy (L)', linewidth=2)
line_pe,  = ax_eng.plot([], [], 'g--', label='Potential (L)')
line_ke,  = ax_ke.plot([], [], 'b-', label='Kinetic (R)')

# Labels
ax_eng.set_xlabel("Time [s]")
ax_eng.set_ylabel("Total / Potential Energy [J]", color='k')
ax_ke.set_ylabel("Kinetic Energy [J]", color='b')

# Legend setup (combining both axes)
lines = [line_tot, line_pe, line_ke]
labels = [l.get_label() for l in lines]
ax_eng.legend(lines, labels, loc='center right')
ax_eng.grid(True)

try:
    for step in range(STEPS):
        dt = sim.step()
        
        if step % PLOT_EVERY == 0:
            t = sim.sim_time
            
            # Print explicit debug values to verify they aren't zero
            if len(metrics["pe"]) > 0:
                curr_pe = metrics["pe"][-1]
                curr_ke = metrics["ke"][-1]
                logger.info(f"Step {step}: t={t:.3f} | PE={curr_pe:.4f} | KE={curr_ke:.6f}")
            
            # Update Density
            img.set_data(sim.density)
            ax_map.set_title(f"Step {step} (t={t:.2f})")
            
            # Update Energy Plots
            times = metrics["time"]
            if len(times) > 1:
                line_tot.set_data(times, metrics["total_E"])
                line_pe.set_data(times, metrics["pe"])
                line_ke.set_data(times, metrics["ke"])
                
                # Rescale Views
                ax_eng.set_xlim(0, t)
                ax_eng.set_ylim(min(metrics["pe"])*0.95, max(metrics["total_E"])*1.05)
                ax_ke.set_xlim(0, t)
                ax_ke.set_ylim(0, max(metrics["ke"])*1.1)
                
            plt.pause(0.001)

except KeyboardInterrupt:
    logger.warning("Simulation stopped by user")

logger.info("Simulation completed, displaying final plots...")
plt.ioff()
plt.show()