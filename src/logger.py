import logging
import os
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def get_logger():
    logger = logging.getLogger("main_logger")
    if logger.handlers:
        return logger

    os.makedirs(f"{project_root}/outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{project_root}/outputs/logs/app_{timestamp}.log"

    file_handler = logging.FileHandler(
        log_filename, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
