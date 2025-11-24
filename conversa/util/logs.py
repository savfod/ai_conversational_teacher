import sys
import time
from functools import wraps
from typing import Any, Literal

import loguru


def get_logger(name: str) -> Any:
    """Get a logger instance for the given module name.

    Args:
        name: Name of the module to get the logger for.
    Returns:
        Logger instance.
    """
    return loguru.logger.bind(module=name)


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
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
    loguru.logger.add(
        sink=f"logs/{time.strftime('%Y-%m-%d')}.log", level=level, rotation="00:00"
    )


def log_function_duration(name: str | None = None):
    """
    Decorator to log the duration of a function call.

    Args:
        name (str, optional): Name to use in the log message.
            If not provided, the function's name will be used.

    Example usage:
        @log_function_duration(name="MyFunction")
        def my_function():
            # Function implementation
            pass

        @log_function_duration()
        def my_function():
            # Function implementation
            pass

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the module where the decorated function is defined
            func_logger = get_logger(func.__module__)
            if name is None:
                func_name = func.__name__
            else:
                func_name = name

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            func_logger.debug(
                f"Function {func_name} completed in: {duration:.2f} seconds"
            )
            return result

        return wrapper

    return decorator
