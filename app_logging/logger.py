import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "app", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s"))

    fh_path = os.path.join(log_dir, "app.log")
    fh = RotatingFileHandler(fh_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
