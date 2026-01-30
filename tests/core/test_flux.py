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
    Both states must be moving faster than their sound speed to the right.
    """
    solver = SolverClass(gamma)
    
    # Left: Mach ~8.5 (10 / 1.18)
    L = State(rho=1.0, u=10.0, v=0.0, p=1.0)
    
    # Right: Mach ~9.4 (10 / 1.06) <-- CHANGED u from 0.0 to 10.0
    R = State(rho=0.125, u=10.0, v=0.0, p=0.1) 

    flux = solver.solve(L, R)
    
    # Now S_L = min(L.u - a_L, R.u - a_R)
    # S_L = min(8.8, 8.9) > 0
    # Solver should detect S_L > 0 and immediately return F(L)
    
    expected_mass_flux = L.rho * L.u # 1.0 * 10.0 = 10.0
    
    assert flux.rho == pytest.approx(expected_mass_flux, rel=1e-5)
    assert flux.mom_x == pytest.approx(L.rho * L.u**2 + L.p, rel=1e-5)

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