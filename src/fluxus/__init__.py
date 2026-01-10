"""
Fluxus: High-performance scientific simulation engine
"""

__version__ = "0.1.0"

# Import C++ extension module
from . import _core

# Import Python API
from .simulation import Simulation, SimulationConfig
from .runner import Runner
from .utils import load_config, save_results

__all__ = [
    "__version__",
    "Simulation",
    "SimulationConfig",
    "Runner",
    "load_config",
    "save_results",
]
