from pathlib import Path

from loguru import logger


def LOGGERS():
    from justatom.configuring.prime import Config

    loggers_filename_or_path = Config.loguru["LOG_FILE_NAME"]
    fp = Path(loggers_filename_or_path)
    logger.add(
        str(fp),
        rotation=Config.loguru["LOG_ROTATION"],
        retention=Config.loguru["LOG_RETENTION"],
    )


__all__ = ["LOGGERS"]
