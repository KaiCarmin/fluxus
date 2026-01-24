import pytest
from fluxus.core import State, HLLSolver

# 1. Define the Solvers we want to test
# Format: (SolverClass, name)
SOLVERS = [
    (HLLSolver, "HLL"),
    # (HLLCSolver, "HLLC") # Uncomment later when implemented
]

@pytest.fixture
def gamma():
    return 1.4

@pytest.mark.parametrize("SolverClass, solver_name", SOLVERS)
def test_supersonic_flow(SolverClass, solver_name, gamma):
    """
    Physical Check: Supersonic flow to the right.
    The Flux should be exactly exactly F(LeftState), because
    no information from the Right can travel upstream against the flow.
    """
    solver = SolverClass(gamma)
    
    # Left: High pressure, Moving FAST to the right (Supersonic)
    # Sound speed ~ 1.18. Velocity = 10.0. Mach >> 1.
    L = State(rho=1.0, u=10.0, v=0.0, p=1.0)
    R = State(rho=0.125, u=0.0, v=0.0, p=0.1) # Standard Sod Right state

    flux = solver.solve(L, R)
    
    # Calculate expected physical flux from L manually
    # Flux_mass = rho * u = 1.0 * 10.0 = 10.0
    expected_mass_flux = L.rho * L.u
    
    # Allow small floating point tolerance
    assert flux.rho == pytest.approx(expected_mass_flux, rel=1e-5)
    assert flux.mom_x == pytest.approx(L.rho * L.u**2 + L.p, rel=1e-5)
    
    print(f"\n[{solver_name}] Supersonic Test Passed: Mass Flux {flux.rho}")

@pytest.mark.parametrize("SolverClass, solver_name", SOLVERS)
def test_symmetry(SolverClass, solver_name, gamma):
    """
    Physical Check: Symmetry.
    If L and R are identical, flux should be F(L) (or F(R)).
    """
    solver = SolverClass(gamma)
    state = State(1.0, 0.0, 0.0, 1.0)
    
    flux = solver.solve(state, state)
    
    # Velocity is 0, so mass flux should be 0
    assert flux.rho == pytest.approx(0.0)
    # Momentum flux is just Pressure (since u=0)
    assert flux.mom_x == pytest.approx(1.0)