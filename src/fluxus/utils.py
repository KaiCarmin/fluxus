import logging
import sys

def setup_logger(level: str = "INFO"):
    logger = logging.getLogger("fluxus")
    logger.setLevel(level)
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Avoid duplicate handlers if setup_logger is called twice
    if not logger.handlers:
        logger.addHandler(ch)
        
    return logger