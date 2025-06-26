##..........................LOGGER MODULE..........................##
import logging
import os 
from datetime import datetime

## Creating LOG DIR: 
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

## Creating LOG FILE:
LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

## Configuring the logger:

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

## Creating a logger function:
def get_logger(name):
    """
    Returns a logger with the specified name.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Prevent adding handlers multiple times
        handler = logging.FileHandler(LOG_FILE)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
        
    return logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")
    logger.debug("This is a debug message.")
    logger.critical("This is a critical message.")