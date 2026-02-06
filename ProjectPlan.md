# Fluxus Project Plan

## Overview

The visualization module provides reusable, composable components for real-time and post-processing visualization of CFD simulations. The design emphasizes simplicity for common use cases while maintaining extensibility for advanced scenarios.

**Module Separation Strategy:**

- `fluxus.core`: C++ engine with the core calculations, State conversions (already exists) and analysis heavy calculations
- `fluxus.analysis`: Physics calculations, derived quantities, data extraction, **metrics tracking** (new standalone module)
- `fluxus.io`: Data import/export for checkpointing, visualization, restarts (new standalone module)
- `fluxus.visualisation`: Plotting and rendering only - consumes data from analysis (visualization-specific)

This separation ensures:

- Analysis functions reusable in user scripts
- I/O operations independent of visualization workflow
- Visualization stays lightweight (matplotlib only)
- Heavy computations can be optimized in C++ if needed

### Core Architecture

(add core extensions - RK integrators, etc...)

### Python modules

#### 1. Analysis Module (`fluxus.analysis`)

**Purpose:** Physics-based calculations and data extraction. Standalone module used by visualization and user analysis scripts.

**Data Extraction (`fluxus.analysis.extract`):**

- **Leverages core's `State::from_conserved()`** - no reimplementation needed
- `get_primitives(sim, gamma)`: Extract primitive variables using core conversions
  - Returns dict: `{'rho': array, 'u': array, 'v': array, 'w': array, 'p': array}`
  - Automatically strips ghost cells
  - Supports 1D, 2D, 3D (returns appropriate shapes)
  - Zero-copy views where possible
- `get_conserved(sim)`: Returns conservative variables (direct grid access)
- `get_slice(sim, axis, position, gamma)`: Extract 2D slice from 3D data

**Derived Quantities (`fluxus.analysis.derived`):**

- `compute_velocity_magnitude(sim, gamma)`: |**v**| = √(u² + v² + w²)
- `compute_mach_number(sim, gamma)`: M = |**v**| / c_s
- `compute_temperature(sim, gamma, R)`: T from ideal gas (if applicable)
- `compute_vorticity(sim, gamma)`: ∇ × **v** (2D: scalar, 3D: vector)
  - **Candidate for C++ implementation** (requires spatial derivatives)
- `compute_divergence(sim, gamma)`: ∇ · **v**
  - **Candidate for C++ implementation**
- `compute_q_criterion(sim, gamma)`: Vortex identification
  - **Candidate for C++ implementation** (expensive tensor operations)

**Conservation Checks (`fluxus.analysis.conservation`):**

- `compute_total_mass(sim)`: ∫ ρ dV
- `compute_total_momentum(sim)`: ∫ ρ**v** dV
- `compute_total_energy(sim)`: ∫ E dV
  - **Candidate for C++ implementation** (simple sum, but large arrays)
- `compute_kinetic_energy(sim, gamma)`: ∫ ½ρ|**v**|² dV
  - **Candidate for C++ implementation**
- `compute_potential_energy(sim, gx, gy, gz)`: ∫ ρ**g**·**r** dV
  - **Candidate for C++ implementation**

**Coordinate Generation (`fluxus.analysis.grid`):**

- `get_coordinates_1d(sim)`: Returns x array
- `get_coordinates_2d(sim)`: Returns X, Y meshgrids
- `get_coordinates_3d(sim)`: Returns X, Y, Z meshgrids
- `get_cell_centers(sim, axis)`: Cell center positions along axis

**Why separate from visualization:**

- Analysis functions useful for users writing custom scripts
- Can be tested independently
- Clear dependency: visualisation imports analysis, not vice versa
- Performance-critical functions identified for future C++ implementation

**Metrics Tracking (`fluxus.analysis.metrics`):**

**MetricsCollector:**

- Generic callback system that plugs into `Simulation.add_callback()`
- Uses conservation functions from `fluxus.analysis.conservation`
- Pre-built metrics:
  - `'total_energy'` → `conservation.compute_total_energy()`
  - `'kinetic'` → `conservation.compute_kinetic_energy()`
  - `'potential'` → `conservation.compute_potential_energy()`
  - `'mass'` → `conservation.compute_total_mass()`
  - `'momentum'` → `conservation.compute_total_momentum()`
- Configurable sampling rate (collect every N steps)
- Export to dict/DataFrame/CSV
- Metadata storage (simulation parameters, timestamps)
- **Independent of visualization** - can track metrics without plotting

**Usage Pattern:**

```python
from fluxus.analysis import MetricsCollector

metrics = MetricsCollector(track=['total_energy', 'kinetic', 'potential'])
metrics.configure_potential_energy(gx=0.0, gy=-0.1, gz=0.0)
metrics.set_sampling_rate(10)  # Every 10 steps

sim.add_callback(metrics.collect)

# Run simulation
for step in range(1000):
    sim.step()

# Export metrics
data = metrics.get_data()  # Returns dict
metrics.to_csv('metrics.csv')  # Or save to file
```

#### 2. I/O Module (`fluxus.io`)

**Purpose:** Data import/export for checkpointing, visualization formats, and restarts. Standalone module.

**Checkpoint System (`fluxus.io.checkpoint`):**

**HDF5Writer:**

- Save full simulation state to HDF5 format
- `save_checkpoint(sim, filename, compression='gzip')`: Save current state
- `load_checkpoint(filename)`: Returns data dict for restart
- Metadata: time, step, grid parameters, gamma
- Efficient compression options
- Support for time series (multiple timesteps in one file)

**Usage:**

```python
from fluxus.io import save_checkpoint, load_checkpoint

# Save
save_checkpoint(sim, 'checkpoint_t0.5.h5', compression='gzip', compression_level=4)

# Load for restart
data = load_checkpoint('checkpoint_t0.5.h5')
sim.data[:] = data['grid_data']
sim.sim_time = data['metadata']['time']
```

**Visualization Export (`fluxus.io.export`):**

**VTKWriter:**

- Export to VTK formats for ParaView/VisIt
- `write_vtk(sim, filename, gamma)`: Write structured grid (.vti)
- `write_pvtk(sim, filename, gamma)`: Parallel VTK for MPI runs
- Automatically converts to primitives
- Includes derived quantities (optional: vorticity, Mach, etc.)

**Usage:**

```python
from fluxus.io import write_vtk

# Export for ParaView
write_vtk(sim, 'output_t0.5.vti', gamma=1.4, 
          include=['vorticity', 'mach_number'])
```

**XDMFWriter:**

- Combined HDF5 + XDMF for efficient ParaView workflow
- `write_xdmf(sim, filename, gamma)`: Write both .h5 and .xmf
- Time series support (append to existing file)

**Time Series Manager (`fluxus.io.timeseries`):**

**TimeSeriesWriter:**

- Manages multiple timesteps efficiently
- Single HDF5 file with multiple groups
- Or individual files with naming convention
- `append_timestep(sim, writer, time)`: Add new timestep
- Memory-efficient streaming

**Usage:**

```python
from fluxus.io import TimeSeriesWriter

with TimeSeriesWriter('simulation.h5', mode='w') as writer:
    for step in range(1000):
        sim.step()
        if step % 100 == 0:
            writer.append_timestep(sim, sim.sim_time, gamma=1.4)
```

**Data Formats Supported:**

- **HDF5**: Primary format (checkpoints, analysis, time series)
- **VTK/VTI**: Structured grids for ParaView
- **XDMF**: HDF5 + XML for ParaView (efficient)
- **NPZ**: NumPy format (simple, for testing)

#### 3. Visualization Module (`fluxus.visualisation`)

**Purpose:** Rendering and visual output only. Consumes data from `fluxus.analysis` and optionally displays metrics tracked by `fluxus.analysis.metrics`.

**Real-Time Plotting Classes (`fluxus.visualisation.live`):**

**LivePlot1D:**

- Multi-panel layouts (density, velocity, pressure, energy)
- Auto-scaling or fixed axis limits
- Efficient line updates without full redraws (blitting)
- Configurable update frequency
- Time/step annotations
- **Uses `fluxus.analysis.extract.get_primitives()` internally**
- **Can display metrics from `MetricsCollector`** (optional panel)

**LivePlot2D:**

- Heatmap visualization with imshow
- Colorbar management
- Multiple variables in subplots
- Variable scaling (linear/log)
- Extent-aware (physical coordinates via `fluxus.analysis.grid`)

**MultiPanelPlot:**

- Flexible layout system (rows × cols)
- Mix plot types (heatmaps + time series)
- Dual-axis support for different scales
- Synchronized time updates across panels
- **Can add metrics time series panels** via `add_metrics_panel(metrics, variables)`

**Common Features:**

- `plt.ion()` integration for non-blocking updates
- `update_every` parameter to control performance
- Manual or automatic axis rescaling
- Save snapshot capability

**Displaying Tracked Metrics (`fluxus.visualisation.live`):**

**Integration with MetricsCollector:**

```python
from fluxus.analysis import MetricsCollector
from fluxus.visualisation import MultiPanelPlot

# Track metrics (analysis module)
metrics = MetricsCollector(track=['total_energy', 'kinetic', 'potential'])
sim.add_callback(metrics.collect)

# Visualize with metrics panel
vis = MultiPanelPlot(layout=(1, 2))
vis.add_heatmap(0, variable='density')
vis.add_metrics_panel(1, metrics, variables=['total_energy', 'kinetic', 'potential'])

for step in range(1000):
    sim.step()
    vis.update(sim)  # Automatically updates metrics panel
```

**Snapshot Management (`fluxus.visualisation.snapshots`):**

**SnapshotManager:**

- Auto-incrementing filename generation with zero-padding
- Multiple format support (PNG, PDF, SVG)
- Metadata embedding in filenames (time, step, resolution)
- Directory organization
- Cleanup old snapshots

**Plot Styling (`fluxus.visualisation.styles`):**

- Pre-configured plot styles for common problems
- `style_shock_tube()`: 3-panel layout with appropriate limits
- `style_instability()`: Density map + vertical profile
- `style_vortex()`: Vorticity colormap presets
- Consistent fonts, colors, grid aesthetics

### C++ Core Integration Strategy

**Already Available in Core (types.hpp):**

- `State::from_conserved()` - Primitive extraction (efficient C++)
- `State::to_conserved()` - Conservative conversion
- `State::to_flux()` - Flux calculation
- **These are already exposed to Python via bindings**

**Analysis Module Usage:**

```python
# Python analysis module uses core's State conversions
def get_primitives(sim, gamma):
    raw = sim.data.reshape(...)
    primitives = {'rho': [], 'u': [], 'v': [], 'w': [], 'p': []}
  
    for cell in inner_cells:
        # Uses core.State.from_conserved (already C++!)
        state = State.from_conserved(
            cell[0], cell[1], cell[2], cell[3], cell[4], gamma
        )
        primitives['rho'].append(state.rho)
        primitives['u'].append(state.u)
        # ... etc
  
    return primitives
```

**Candidates for Future C++ Implementation:**

**New C++ Module (Separate from Grid):**

**`Analysis.hpp` - All Analysis Functions:**

- Field computations (spatial derivatives → arrays)
- Scalar reductions (integration → doubles)
- Pure functions that take `const Grid&` reference
- Unified namespace for all diagnostic calculations

**High Priority (Performance-Critical):**

1. **Vorticity Calculation** (`analysis::compute_vorticity()`)

   - Requires spatial derivatives (∂v/∂x - ∂u/∂y)
   - Stencil operations over large arrays
   - Called frequently for instability visualization
   - Return numpy array via pybind11
2. **Energy Integration** (`analysis::compute_kinetic_energy()`, `analysis::compute_potential_energy()`)

   - Simple sums, but over millions of cells
   - Can be parallelized (OpenMP)
   - Return scalar double
3. **Q-Criterion** (`analysis::compute_q_criterion()`)

   - Expensive: requires velocity gradient tensor eigenvalues
   - Used for 3D vortex identification
   - Significant speedup potential

**Medium Priority:**

4. **Divergence** (`analysis::compute_divergence()`)

   - Similar to vorticity (spatial derivatives)
   - Less frequently used
5. **Gradient Operators** (`analysis::compute_gradient()`)

   - General-purpose spatial derivatives
   - Foundation for many derived quantities
6. **Mass/Momentum Integration** (`analysis::compute_total_mass()`, `analysis::compute_total_momentum()`)

   - Conservation checks
   - Simple parallel reductions

**Python Usage:**

```python
# Fast path: use C++ implementation if available
from fluxus import core

try:
    # Use C++ analysis module (unified namespace)
    vorticity = core.analysis.compute_vorticity(sim.grid, gamma=1.4)
    ke = core.analysis.compute_kinetic_energy(sim.grid, gamma=1.4)
    pe = core.analysis.compute_potential_energy(sim.grid, 0.0, -0.1, 0.0)
except AttributeError:
    # Fallback: pure Python implementation
    from fluxus.analysis.derived import _compute_vorticity_python
    vorticity = _compute_vorticity_python(sim, gamma=1.4)
```

**Benefits:**

- 10-100× speedup for derivative calculations
- Reduced Python overhead for large grids
- Enables real-time advanced visualizations
- Fallback ensures compatibility
- **Clean separation**: Grid stays focused, analysis is modular
- **Easy testing**: Analysis functions are pure functions of Grid state
- **Unified API**: Single namespace for all diagnostics

**Directory Structure:**

```
src/core/
├── include/
│   ├── Grid.hpp
│   ├── types.hpp
│   ├── analysis/
│   │   └── Analysis.hpp         # All analysis functions
│   ├── flux/
│   ├── integrator/
│   └── ...
├── lib/
│   ├── Grid.cpp
│   ├── analysis/
│   │   ├── fields.cpp           # Vorticity, divergence, Q-criterion
│   │   ├── reductions.cpp       # Energy, mass, momentum integrals
│   │   └── gradients.cpp        # Shared gradient operators
│   ├── flux/
│   └── ...
└── bindings.cpp                 # Expose core.analysis submodule
```

**Notes:**

- Single header (`Analysis.hpp`) keeps API unified
- Implementation can be split for organization (fields.cpp, reductions.cpp)
- All exposed under `fluxus.core.analysis` namespace in Python
- If module grows very large (>1000 lines), can reconsider splitting header

### Example User Usage

### Example User Usage

#### Example 1: Simple 1D Visualization

```python
from fluxus import Simulation
from fluxus.core import HLLCSolver, GodunovIntegrator
from fluxus.visualisation import LivePlot1D

# Setup simulation
solver = HLLCSolver(gamma=1.4)
integrator = GodunovIntegrator(solver)
sim = Simulation(nx=200, ny=1, integrator=integrator)
sim.set_initial_condition(sod_ic)

# Create live plotter (uses fluxus.analysis internally)
vis = LivePlot1D(
    variables=['density', 'velocity', 'pressure'],
    gamma=1.4,
    update_every=20
)
vis.start_interactive()

# Run with automatic visualization
for step in range(1000):
    sim.step()
    vis.update(sim)  # Calls analysis.get_primitives() internally

vis.show_final()
```

#### Example 2: 2D Dashboard with Metrics Tracking

```python
from fluxus import Simulation
from fluxus.analysis import MetricsCollector
from fluxus.visualisation import MultiPanelPlot

# Setup 2D simulation (Rayleigh-Taylor)
sim = Simulation(nx=100, ny=300, integrator=muscl_integrator)
sim.set_initial_condition(rt_ic)

# Track metrics (analysis module - independent of visualization)
metrics = MetricsCollector(track=['total_energy', 'kinetic', 'potential'])
metrics.configure_potential_energy(gx=0.0, gy=-0.1, gz=0.0)
sim.add_callback(metrics.collect)

# Create visualization with metrics panel
vis = MultiPanelPlot(layout=(1, 2), figsize=(14, 6))
vis.add_heatmap(0, variable='density', cmap='RdBu_r', title='Density Field')
vis.add_metrics_panel(1, metrics, 
                     left_axis=['total_energy', 'potential'],
                     right_axis=['kinetic'])

# Run simulation
vis.start_interactive()
for step in range(4000):
    sim.step()
    vis.update(sim)  # Automatically updates both density and metrics

# Save metrics to file and show final state
metrics.to_csv('energy_evolution.csv')
vis.show_final()
```

#### Example 3: Full Workflow (Analysis + I/O + Visualization)

```python
from fluxus import Simulation
from fluxus.analysis import MetricsCollector, compute_vorticity
from fluxus.io import save_checkpoint, TimeSeriesWriter
from fluxus.visualisation import LivePlot2D
import matplotlib.pyplot as plt

# Setup simulation
sim = Simulation(nx=200, ny=200, integrator=integrator)
sim.set_initial_condition(vortex_ic)

# Track conservation
metrics = MetricsCollector(track=['mass', 'total_energy'])
sim.add_callback(metrics.collect)

# Real-time visualization
vis = LivePlot2D(variable='density', update_every=50)

# Time series export for ParaView
with TimeSeriesWriter('output.h5', mode='w') as ts_writer:
    vis.start_interactive()
  
    for step in range(10000):
        sim.step()
        vis.update(sim)
      
        # Checkpoint every 1000 steps
        if step % 1000 == 0:
            save_checkpoint(sim, f'checkpoint_step{step}.h5')
            print(f"Checkpoint saved at step {step}")
      
        # Export for ParaView every 100 steps
        if step % 100 == 0:
            ts_writer.append_timestep(sim, sim.sim_time, gamma=1.4)

vis.show_final()

# Post-processing: analyze final state
vorticity = compute_vorticity(sim, gamma=1.4)  # Uses C++ if available
primitives = get_primitives(sim, gamma=1.4)

fig, ax = plt.subplots()
im = ax.contourf(vorticity, levels=20, cmap='RdBu_r')
plt.colorbar(im, label='Vorticity')
plt.savefig('vorticity_final.pdf')

# Export metrics and check conservation
metrics.to_csv('conservation.csv')
mass_drift = (metrics.get_data()['mass'][-1] / metrics.get_data()['mass'][0] - 1) * 100
print(f"Mass conservation: {mass_drift:.2e}% drift")
```

```python
from fluxus.visualisation import MultiPanelPlot

# Create 1x2 layout: density map + energy plot
vis = MultiPanelPlot(layout=(1, 2), figsize=(14, 6))

# Left panel: density heatmap
vis.add_heatmap(
    position=0,
    variable='density',
    cmap='RdBu_r',
    title='Density Field'
)

# Right panel: dual-axis energy plot
vis.add_line_plot(
    position=1,
    xlabel='Time [s]',
    ylabel_left='Total / Potential Energy [J]',
    ylabel_right='Kinetic Energy [J]'
)

# Metrics tracking
metrics = MetricsCollector(track=['total_energy', 'kinetic', 'potential'])
sim.add_callback(metrics.collect)

# Update callback
def update_callback(sim_instance):
    if sim_instance.step_count % 20 == 0:
        # Update density heatmap
        vis.update_heatmap(0, sim_instance.density, 
                          title=f"t={sim_instance.sim_time:.2f}")
  
        # Update energy lines
        data = metrics.get_data()
        vis.update_line(1, 
                       x_data=data['time'],
                       left_lines={'Total': data['total_energy'],
                                  'Potential': data['potential']},
                       right_lines={'Kinetic': data['kinetic']})

sim.add_callback(update_callback)

# Run
vis.start_interactive()
for step in range(4000):
    sim.step()

vis.show_final()
```

### Development Phases

#### Phase 1: Analysis Module Foundation

**Goal:** Standalone analysis module that leverages core's State conversions

**Components:**

1. **`fluxus.analysis.extract` module**

   - `get_primitives(sim, gamma)` - uses `State.from_conserved()` from core
   - `get_conserved(sim)` - direct grid access with ghost stripping
   - Ghost cell handling utilities
   - Unit tests with known states
2. **`fluxus.analysis.grid` module**

   - `get_coordinates_1d(sim)`
   - `get_coordinates_2d(sim)`
   - Cell center calculation
3. **`fluxus.analysis.conservation` module**

   - `compute_total_mass(sim)` - pure Python (simple sum)
   - `compute_total_energy(sim)` - pure Python
   - `compute_kinetic_energy(sim, gamma)` - pure Python
   - `compute_potential_energy(sim, gx, gy, gz)` - pure Python
4. **`fluxus.analysis.metrics` module**

   - `MetricsCollector` class
   - Uses conservation functions internally
   - Time-series storage and export (CSV/dict/DataFrame)
5. **Documentation & Tests**

   - API reference for analysis module
   - Unit tests for all extraction/calculation functions
   - Example: Manual data extraction and plotting

**Deliverable:** Independent `fluxus.analysis` module usable without visualization

#### Phase 2: I/O Module

**Goal:** Data persistence for checkpointing, restart, and ParaView export

**Components:**

1. **`fluxus.io.checkpoint` module**

   - `save_checkpoint()` - HDF5 checkpointing with compression
   - `load_checkpoint()` - Restart from checkpoint
   - Metadata storage (time, step, parameters)
2. **`fluxus.io.export` module**

   - `write_vtk()` - VTK export for ParaView/VisIt
   - `write_xdmf()` - HDF5 + XDMF for efficient ParaView workflow
3. **`fluxus.io.timeseries` module**

   - `TimeSeriesWriter` context manager
   - Efficient multi-timestep storage in single HDF5
4. **Documentation & Tests**

   - Checkpoint/restart workflow example
   - ParaView visualization guide
   - Unit tests for save/load operations

**Deliverable:** Working `fluxus.io` module with HDF5/VTK support

#### Phase 3: Basic Visualization (1D)

**Goal:** Simple real-time visualization for 1D problems

**Components:**

1. **`fluxus.visualisation.live.LivePlot1D` class**

   - Uses `analysis.extract.get_primitives()` internally
   - 3-panel layout (ρ, u, p)
   - `update()` method with blitting for performance
   - Auto-scaling or fixed limits
2. **Documentation**

   - Update `sod_shock_tube.py` to use new modules
   - Tutorial: "Simple 1D Visualization" (Example 1)

**Deliverable:** Replace manual plotting in `sod_shock_tube.py` with `LivePlot1D`

#### Phase 4: 2D Visualization & Multi-Panel

**Goal:** Support complex 2D simulations with dashboard layouts

**Components:**

1. **`fluxus.visualisation.live.LivePlot2D` class**

   - Heatmap with colorbar
   - Uses `analysis.grid.get_coordinates_2d()` for physical extents
   - Multiple variables in subplots
2. **`fluxus.visualisation.live.MultiPanelPlot` class**

   - Flexible layout system (rows × cols)
   - `add_heatmap()` - add field visualization
   - `add_metrics_panel()` - display tracked metrics from `MetricsCollector`
   - Dual-axis support for different scales
   - Automatic updates from metrics data
3. **Documentation**

   - Update `rt_instability.py` to use new modules
   - Tutorial: "2D Dashboard with Metrics" (Example 2)

**Deliverable:** Clean RT example with dashboard showing density + energy evolution

#### Phase 5: Advanced Analysis (Derived Quantities)

**Goal:** Add vorticity, Mach number, and other physics-based diagnostics

**Components:**

1. **`fluxus.analysis.derived` module**

   - `compute_velocity_magnitude(sim, gamma)` - pure Python/NumPy
   - `compute_mach_number(sim, gamma)` - pure Python
   - `compute_vorticity(sim, gamma)` - pure Python with finite differences
   - `compute_divergence(sim, gamma)` - pure Python
2. **Gradient operators (internal)**

   - Finite difference stencils (central differences)
   - Boundary handling
3. **Documentation**

   - Tutorial: "Full Workflow" (Example 3)
   - Advanced analysis example with vorticity visualization

**Deliverable:** Complete analysis module with derived quantities

#### Phase 6: C++ Acceleration (Optional Performance Layer)

**Goal:** Accelerate performance-critical analysis functions

**Components:**

1. **`Analysis.hpp` C++ module**

   - `analysis::compute_vorticity()` - C++ with OpenMP
   - `analysis::compute_kinetic_energy()` - parallel reduction
   - `analysis::compute_potential_energy()` - parallel reduction
   - Exposed via `fluxus.core.analysis` submodule
2. **Python fallback pattern**

   - Try C++ implementation first
   - Fallback to pure Python if not available
   - Transparent to user
3. **Performance profiling**

   - Benchmark Python vs C++ implementations
   - Document speedup (target: 10-100×)
4. **Documentation**

   - C++ integration guide
   - Performance comparison table

**Deliverable:** Optional C++ acceleration with fallback

**Estimated Effort:** 3-4 days (can be deferred)

---

#### Phase 7: Polish, Snapshots & Documentation

**Goal:** Production-ready modules with complete documentation

**Components:**

1. **`fluxus.visualisation.snapshots.SnapshotManager`**

   - Auto-incrementing filenames with zero-padding
   - Multiple formats (PNG, PDF, SVG)
   - Metadata embedding
2. **`fluxus.visualisation.styles` module**

   - `style_shock_tube()` preset layout
   - `style_instability()` preset layout
   - Consistent aesthetics
3. **Complete documentation**

   - Full API reference (analysis, io, visualisation)
   - Gallery with all 3 examples
   - Best practices guide
   - Troubleshooting guide
4. **Comprehensive testing**

   - Unit tests for all modules
   - Integration tests (full workflows)
   - Performance regression tests
   - CI/CD setup

**Deliverable:** Polished, production-ready modules

### Scaling Considerations for 3D

#### Architectural Preparation

To facilitate future 3D support, the current design includes:

1. **Dimensionality Abstraction**

   - `extract_slice(axis, position)` method already in API
   - Coordinate generation functions accept dimensionality parameter
   - Grid access doesn't assume 2D
2. **Backend Abstraction (Future)**

   - Current: Direct matplotlib usage
   - Future: `VisualizationBackend` abstract class
     - `MatplotlibBackend` (1D/2D, simple 3D slices)
     - `PyVistaBackend` (volumetric rendering, iso-surfaces)
     - `ParaViewBackend` (in-situ via Catalyst)
3. **Data Export Pipeline**

   - Phase 4 includes `DataWriter` class:
     - `write_vtk(sim, filename)` for ParaView/VisIt
     - `write_hdf5(sim, filename)` for efficient storage
     - `write_xdmf(sim, filename)` for parallel I/O
   - Visualization then happens in external tools
4. **3D Slice Visualization**

   - `LivePlot3DSlice` class (future):
     - Extract XY, XZ, YZ planes
     - Reuses `LivePlot2D` for rendering
     - Slice position slider
   - `LineProfile` class:
     - Extract 1D line through 3D domain
     - Reuses `LivePlot1D` for rendering

#### Limitations Acknowledged

- **Real-time 3D volumetric rendering not in scope** for matplotlib-based module
- Large 3D simulations (>100³) should use:
  - Offline visualization (ParaView, VisIt)
  - In-situ processing (Catalyst)
  - Cloud-based rendering (e.g., ParaView Web)
- The module provides **data export and slice extraction**, not full 3D rendering

#### Optional Dependencies Strategy

```
# Base installation (matplotlib only)
pip install fluxus

# With 3D visualization support
pip install fluxus[viz3d]  # Adds PyVista, VTK

# With HPC I/O support  
pip install fluxus[hpc]  # Adds h5py, mpi4py
```

#### Migration Path

When 3D becomes priority:

1. Add `fluxus.visualisation.backends` submodule
2. Implement `PyVistaBackend` alongside `MatplotlibBackend`
3. User selects backend: `LivePlot3D(backend='pyvista')`
4. Existing 1D/2D code unchanged (matplotlib remains default)
5. Advanced users opt-in to 3D rendering

### Design Principles

1. **Simplicity First:** Common cases (1D shock, 2D instability) should be 5-10 lines
2. **Composability:** Mix and match components (plots, metrics, snapshots)
3. **Performance Awareness:** Update frequency controls, blitting, lazy evaluation
4. **Zero Coupling:** No dependencies on specific solvers/integrators
5. **Extensibility:** Users can subclass or register custom derived quantities
6. **Future-Proof:** API designed to accommodate 3D without breaking changes
