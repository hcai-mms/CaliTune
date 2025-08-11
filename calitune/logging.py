import logging
import os
import sys
from datetime import datetime

def setup_logger():
    log_dir = "./log_calitune"
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a log file name with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_file = os.path.join(log_dir, f'Calitune_{timestamp}.log')

    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger

def log_config(logger, config):
    for key, value in config.items():
        globals()[key] = value
        logger.info(f"{key} : {value}")
    logger.info(64 * "-")