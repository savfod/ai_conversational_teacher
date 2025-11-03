import loguru
import time
import sys

def setup_logging(level="INFO"):
    loguru.logger.remove()
    loguru.logger.add(sink=sys.stdout, level=level)
    loguru.logger.add(sink=f"logs/{time.strftime('%Y-%m-%d')}.log", level=level, rotation="00:00")