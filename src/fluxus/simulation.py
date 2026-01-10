from .utils import setup_logger
from .config import SimulationConfig

logger = setup_logger()

class Simulation:
    def __init__(self, config: SimulationConfig):
        logger.info(f"Initializing simulation with grid {config.nx}x{config.ny}")
        # ...