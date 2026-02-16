import os
from pathlib import Path
import time

from loguru import logger


def LOGGERS():
    from justatom.configuring.prime import Config

    loggers_filename_or_path = Config.loguru["LOG_FILE_NAME"] or Path(os.getcwd()) / f"loggers_{time.time()}.log"
    fp = Path(loggers_filename_or_path)
    logger.add(
        str(fp),
        rotation=Config.loguru["LOG_ROTATION"] or "10 MB",
        retention=Config.loguru["LOG_RETENTION"] or "10 days",
    )


__all__ = ["LOGGERS"]
