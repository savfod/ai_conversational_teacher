import loguru
import time
import sys
from typing import Literal


def setup_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO") -> None:
    """Configure application logging.

    This installs a console logger and a daily-rotated file logger using
    loguru. The default level is "INFO".

    Args:
        level: Logging level to set for handlers.

    Returns:
        None
    """
    loguru.logger.remove()
    loguru.logger.add(sink=sys.stdout, level=level)
    loguru.logger.add(sink=f"logs/{time.strftime('%Y-%m-%d')}.log", level=level, rotation="00:00")