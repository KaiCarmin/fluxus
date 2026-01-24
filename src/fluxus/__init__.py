"""
Fluxus: High-performance scientific simulation engine
"""
import sys

# 1. Package Version
__version__ = "0.1.0"

# 2. Bridge the C++ Core
# We import the compiled binary (_core) and alias it to a clean name (core).
# This solves your "Import fluxus.core could not be resolved" error.
try:
    from . import _core as core
except ImportError as e:
    # Helpful error if someone tries to run without compiling first
    raise ImportError(
        f"Could not import the compiled C++ extension 'fluxus._core'. "
        f"Did you run 'pip install .'? Original error: {e}"
    ) from e

# Make core accessible as both fluxus.core and fluxus._core
# This allows "from fluxus.core import State" to work
sys.modules['fluxus.core'] = core

# 3. Flatten the Namespace (The Facade Pattern)
# These imports allow users to access classes directly from the top level.
# e.g. 'from fluxus import SimulationConfig' instead of 'from fluxus.config import SimulationConfig'

# Note: Only uncomment these if you have actually created these files!
from .config import SimulationConfig
from .utils import setup_logger
from .simulation import Simulation

# 4. Define Public API
# This controls what gets imported if someone types 'from fluxus import *'
__all__ = ["core", "Simulation", "SimulationConfig", "setup_logger"]

__all__ = [
    "__version__",
    "Simulation",
    "SimulationConfig",
    "setup_logger",
]
