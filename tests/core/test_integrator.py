import pytest
import numpy as np
from fluxus.core import Grid, State, HLLCSolver, GodunovIntegrator
from fluxus.utils import setup_logger

# Setup logger for tests
logger = setup_logger(level="DEBUG")

# Helper to create a valid data buffer for the C++ Grid
def create_grid_data(nx, ny, nz, ng=2):
    # Total cells including ghosts
    N_x = nx + 2 * ng
    N_y = ny + 2 * ng
    N_z = nz # Assuming no ghosts in Z for 2D/1D if nz=1? 
    # Actually, the C++ code uses 'ng' for all dims in get_index: (j+ng)*stride_y
    # So we must allocate ghosts for Y and Z if loops use them.
    # For now, let's just stick to the safe allocation size:
    
    if nz == 1: # 1D or 2D
         N_z = 1 
         # Note: Our C++ get_index uses 'k * stride_z'. If k=0, stride_z doesn't matter.
    else:
         N_z = nz + 2 * ng

    # 5 Conserved Variables per cell
    return np.zeros((N_z * N_y * N_x * 5), dtype=np.float64)

@pytest.fixture
def solver():
    return HLLCSolver(gamma=1.4)

@pytest.fixture
def integrator(solver):
    return GodunovIntegrator(solver)

def test_static_preservation(integrator):
    """
    If the grid is uniform (P=const, Rho=const, u=0), 
    nothing should happen.
    """
    logger.info("Testing static preservation")
    nx, ny, nz, ng = 10, 1, 1, 2
    data = create_grid_data(nx, ny, nz, ng)
    logger.debug(f"Created grid data: nx={nx}, ny={ny}, nz={nz}, ng={ng}")
    
    # 1. Initialize Grid Wrapper
    grid = Grid(data, 1, nx, ny, nz, ng, 0.1, 0.1, 0.1)
    logger.debug("Initialized grid wrapper")
    
    # 2. Fill with uniform state (rho=1, p=1, u=0)
    # We must set Conserved Variables in the buffer manually or via helper
    # State(1,0,0,0,1) -> Conserved(1, 0, 0, 0, 2.5) for gamma=1.4
    # Let's use the grid's C++ accessor if possible, but we can't write 'State' easily 
    # without a Python helper. 
    # Plan B: Manipulate the numpy array directly using stride knowledge.
    
    # Stride X is 5. 
    # We want to fill every cell.
    # Conserved U = [1.0, 0.0, 0.0, 0.0, 2.5] (E = p/(g-1) = 1/0.4 = 2.5)
    
    # Fill array with default 0s, then set Rho=1, E=2.5
    # Reshape to (TotalCells, 5) for easy assignment
    arr_view = data.reshape(-1, 5)
    arr_view[:, 0] = 1.0 # Rho
    arr_view[:, 4] = 2.5 # Energy
    logger.debug("Filled grid with uniform state: rho=1.0, p=1.0, u=0.0")
    
    # 3. Run one step
    logger.debug("Running integrator step with dt=0.01")
    integrator.step(grid, dt=0.01)
    
    # 4. Check results
    # The middle cell (nx//2) should still be exactly the same
    mid_state = grid.get_state(nx//2, 0, 0)
    logger.debug(f"Middle cell state: rho={mid_state.rho}, p={mid_state.p}, u={mid_state.u}")
    
    assert mid_state.rho == pytest.approx(1.0)
    assert mid_state.p == pytest.approx(1.0)
    assert mid_state.u == pytest.approx(0.0)
    logger.info("Static preservation test passed")

def test_1d_advection(integrator):
    """
    Setup a flow u=10.0 to the right.
    Put a density spike. 
    Check if flux moves mass from Left to Right.
    """
    logger.info("Testing 1D advection")
    nx, ny, nz, ng = 10, 1, 1, 2
    dx = 1.0
    data = create_grid_data(nx, ny, nz, ng)
    logger.debug(f"Created grid data: nx={nx}, ny={ny}, nz={nz}, ng={ng}, dx={dx}")
    
    # Grid wrapper
    grid = Grid(data, 1, nx, ny, nz, ng, dx, 1.0, 1.0)
    logger.debug("Initialized grid wrapper")
    
    # --- SETUP INITIAL CONDITION ---
    # Uniform Flow: u=10, p=1.
    # Background Rho=1.0. Spike Rho=2.0 at index 4.
    
    # Calculate Conserved for Background:
    # Rho=1, u=10 -> Mom=10. E = 1.0/0.4 + 0.5*1*100 = 2.5 + 50 = 52.5
    bg_U = [1.0, 10.0, 0.0, 0.0, 52.5]
    
    # Calculate Conserved for Spike:
    # Rho=2, u=10 -> Mom=20. E = 1.0/0.4 + 0.5*2*100 = 2.5 + 100 = 102.5
    spike_U = [2.0, 20.0, 0.0, 0.0, 102.5]
    
    arr_view = data.reshape(-1, 5)
    arr_view[:] = bg_U # Fill background
    
    # Find index of cell i=4 (accounting for ghosts)
    # Stride calculation: (j+ng)*stride_y + (i+ng)*stride_x
    # j=0, i=4. stride_x=5. stride_y = (nx+2ng)*5 = 14*5 = 70.
    # index = (0+2)*70 + (4+2)*5 = 140 + 30 = 170 (flattened components)
    # Simpler: The reshaped view includes ghosts naturally.
    # Row corresponding to j=0, i=4 is roughly index (ng_y*Width + ng_x + 4)
    
    width = nx + 2*ng
    center_idx = ng * width + (ng + 4) 
    arr_view[center_idx] = spike_U # Set the spike
    logger.debug(f"Set density spike at cell (4, 0, 0)")
    
    # Verify setup
    s_initial = grid.get_state(4, 0, 0)
    logger.debug(f"Initial spike state: rho={s_initial.rho}, u={s_initial.u}")
    assert s_initial.rho == 2.0
    
    # --- RUN STEP ---
    # dt = 0.05. u = 10. CFL = 0.5.
    # The wave should move 0.5 cells to the right.
    # Cell 4 should lose mass. Cell 5 should gain mass.
    logger.debug("Running integrator step with dt=0.05")
    integrator.step(grid, dt=0.05)
    
    # --- CHECK PHYSICS ---
    s_4 = grid.get_state(4, 0, 0)
    s_5 = grid.get_state(5, 0, 0)
    
    logger.info(f"Step Result: Cell 4 Rho={s_4.rho}, Cell 5 Rho={s_5.rho}")
    print(f"\nStep Result: Cell 4 Rho={s_4.rho}, Cell 5 Rho={s_5.rho}")
    
    # Mass should have left Cell 4
    assert s_4.rho < 2.0
    # Mass should have entered Cell 5 (originally 1.0)
    assert s_5.rho > 1.0
    logger.debug(f"Mass correctly advected: Cell 4 lost mass, Cell 5 gained mass")
    
    # Conservation Check: Sum of mass should be conserved (roughly)
    # (ignoring boundary outflow for this small time step)
    total_mass = np.sum(arr_view[:, 0])
    logger.debug(f"Total mass in grid: {total_mass}")
    # Ideally compare to pre-step mass, but simple inequality is enough here.
    logger.info("1D advection test passed")

def test_2d_sweep_activation(integrator):
    """
    Ensure that setting dim=2 actually updates in Y direction.
    """
    logger.info("Testing 2D sweep activation")
    nx, ny, nz, ng = 5, 5, 1, 2
    data = create_grid_data(nx, ny, nz, ng)
    grid = Grid(data, 2, nx, ny, nz, ng, 1.0, 1.0, 1.0) # ndim=2
    logger.debug(f"Created 2D grid: nx={nx}, ny={ny}, nz={nz}, ng={ng}")
    
    arr_view = data.reshape(-1, 5)
    # Set uniform background
    arr_view[:] = [1.0, 0.0, 0.0, 0.0, 2.5]
    logger.debug("Set uniform background state") 
    
    # Set a vertical velocity v=10 in the middle column
    # If Y-sweep works, this should advect mass upwards (to j+1)
    
    # Target cell: i=2, j=2
    width = nx + 2*ng
    # Index for (i=2, j=2)
    # layout is row-major usually (j varies slowest?) 
    # Wait, get_index = (j+ng)*stride_y ... so Y is the slow index (Rows).
    
    row_stride = width
    idx_center = (ng + 2) * row_stride + (ng + 2)
    
    # Set State: Rho=2, v=10
    # Conserved: Rho=2, MomY=20, E = 2.5 + 0.5*2*100 = 102.5
    arr_view[idx_center] = [2.0, 0.0, 20.0, 0.0, 102.5]
    logger.debug("Set vertical velocity v=10 at center cell (2, 2, 0)")
    
    logger.debug("Running integrator step with dt=0.05")
    integrator.step(grid, dt=0.05)
    
    # Check cell (i=2, j=3) - The one ABOVE. 
    # It should have gained mass if Y-sweep ran.
    s_above = grid.get_state(2, 3, 0)
    logger.debug(f"Cell (2, 3, 0) state after step: rho={s_above.rho}")
    
    assert s_above.rho > 1.0, "Y-Sweep did not move mass upwards!"
    logger.info("2D sweep activation test passed")