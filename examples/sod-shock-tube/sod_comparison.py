import numpy as np
import matplotlib.pyplot as plt
from fluxus.simulation import Simulation, BoundaryType
from fluxus.core import (
    HLLCSolver, 
    GodunovIntegrator, MUSCLHancockIntegrator,
    PiecewiseConstantReconstructor, SuperbeeReconstructor,
    State
)

NX = 200
T_MAX = 0.2
GAMMA = 1.4
CFL = 0.5

def run_sod(setup_name, integrator_class, recon_class):
    print(f"Running {setup_name}...")
    
    # Setup Physics
    solver = HLLCSolver(GAMMA)
    if recon_class:
        reconstructor = recon_class()
        if integrator_class == MUSCLHancockIntegrator:
            integrator = integrator_class(solver, reconstructor)
        else:
            integrator = integrator_class(solver, reconstructor)
    else:
        integrator = integrator_class(solver)

    # Setup Simulation
    sim = Simulation(nx=NX, ny=1, integrator=integrator, extent_x=1.0, extent_y=0.1, cfl=CFL)
    
    # Transmissive boundaries for 1D shock tube
    sim.set_boundaries(BoundaryType.Transmissive, BoundaryType.Transmissive,
                       BoundaryType.Transmissive, BoundaryType.Transmissive)

    # Sod Initial Condition
    def sod_ic(x, y):
        if x < 0.5: return State(1.0, 0.0, 0.0, 1.0) # High Pressure
        else:       return State(0.125, 0.0, 0.0, 0.1) # Low Pressure

    sim.set_initial_condition(sod_ic)

    # Run with adaptive time stepping until T_MAX is reached
    while sim.sim_time < T_MAX:
        sim.step()  # Adaptive dt computed automatically
    
    print(f"  Completed in {sim.step_count} steps (t={sim.sim_time:.6f})")
    
    # Extract primitive variables
    raw = sim.data.reshape(sim.ny + 2*sim.ng, sim.nx + 2*sim.ng, 5)
    internal = raw[sim.ng:-sim.ng, sim.ng:-sim.ng, :]
    data_1d = internal[0]
    
    rho = data_1d[:, 0]
    mom_x = data_1d[:, 1]
    E = data_1d[:, 4]
    
    # Compute velocity and pressure
    u = mom_x / rho
    kinetic = 0.5 * rho * u**2
    p = (GAMMA - 1.0) * (E - kinetic)
        
    return rho, u, p

# --- EXECUTE COMPARISONS ---

# 1. First Order (Godunov + Piecewise Constant)
rho_1st, u_1st, p_1st = run_sod("1st Order (Godunov)", GodunovIntegrator, PiecewiseConstantReconstructor)

# 2. Second Order (MUSCL + Superbee)
rho_2nd, u_2nd, p_2nd = run_sod("2nd Order (MUSCL-Hancock + Superbee)", MUSCLHancockIntegrator, SuperbeeReconstructor)

# --- PLOT ---
x = np.linspace(0, 1.0, NX)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Density
axes[0].plot(x, rho_1st, 'o-', label='1st Order (Diffusive)', color='gray', markersize=3, alpha=0.6)
axes[0].plot(x, rho_2nd, 's-', label='2nd Order (Superbee)', color='red', markersize=3, linewidth=2)
axes[0].set_title("Density")
axes[0].set_xlabel("Position (x)")
axes[0].set_ylabel("Density")
axes[0].set_ylim(0, 1.1)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend()

# Velocity
axes[1].plot(x, u_1st, 'o-', label='1st Order (Diffusive)', color='gray', markersize=3, alpha=0.6)
axes[1].plot(x, u_2nd, 's-', label='2nd Order (Superbee)', color='red', markersize=3, linewidth=2)
axes[1].set_title("Velocity")
axes[1].set_xlabel("Position (x)")
axes[1].set_ylabel("Velocity")
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend()

# Pressure
axes[2].plot(x, p_1st, 'o-', label='1st Order (Diffusive)', color='gray', markersize=3, alpha=0.6)
axes[2].plot(x, p_2nd, 's-', label='2nd Order (Superbee)', color='red', markersize=3, linewidth=2)
axes[2].set_title("Pressure")
axes[2].set_xlabel("Position (x)")
axes[2].set_ylabel("Pressure")
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].legend()

plt.suptitle(f"Sod Shock Tube: Accuracy Comparison (t={T_MAX:.2f})")
plt.tight_layout()

# Save for README
dir_path = "examples/sod-shock-tube"
plt.savefig(f"{dir_path}/sod_comparison.png", dpi=150)
print("Saved sod_comparison.png")
plt.show()