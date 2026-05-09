"""Utilities for logging:

* `get_logger` creates and returns a logger with colored output
* `init_logger` initializes and returns a customized logger
* `log_execution` is a decorator to log entry into and ext from a function.
"""

import functools
import logging
import sys
from pathlib import Path
from typing import Callable

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

COLOR_MAP = {
    "DEBUG": Fore.CYAN,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Mirror the standard Formatter logic so we still get message, asctime, and exception text
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        color = COLOR_MAP.get(record.levelname, "")
        level = f"{color}{record.levelname}{Style.RESET_ALL}"
        output = f"{record.asctime} [{record.name}] {level}: {record.message}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            output = f"{output}\n{record.exc_text}"

        return output


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
) -> logging.Logger:
    """Create and return a logger with colored output.

    Args:
        name: Name of the logger.
        level: Minimum logging level.
        log_to_file: If True, also log to a file named '{name}.log'.

    Returns:
        Configured logger instance.

    Examples:
        >>> logger = get_logger("my_logger", level=logging.DEBUG, log_to_file=True)
        >>> logger.debug("This is a debug message.")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured
    logger.setLevel(level)
    formatter = ColorFormatter("%(asctime)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_to_file:
        file_handler = logging.FileHandler(f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def init_logger(
    logger_name: str,
    log_level_for_console: str = "info",
    log_level_for_file: str = "debug",
    save_dir: str | None = None,
) -> logging.Logger:
    """
    Initialize a logger

    Args:
        logger_name: name for the logger
        log_level_for_console: minimum level of messages logged to the
            console logging
        log_level_for_file:
        save_dir:

    Returns:
        the logger

    Examples:
        >>> logger_dir = "logs"
        >>> logger_name = "check_log"
        >>> logger = init_logger(logger_name, save_dir=logger_dir)

        This will create two logs:

        * one printed to console where we run the code (the
          `StreamHandler`),
        * and one that will be saved to file `save_dir/logger_name.txt`
          (the `FileHandler`).

        `'logger.propagate = False'` makes sure that the logs sent to file
        will not be printed to console.

        We use the `Formatter` class to define the format of the logs.
        Here:

        * The time of the log in a human-readable format, `asctime`
        * `levelname` is the level of the log, one out of `INFO, DEBUG, WARNING, ERROR, CRITICAL`.
        * The name of the file, `filename`, from which the log was generated,
        and the line number, `lineno`.
        * Lastly,  the message itself — `message`.

        The default has only `INFO` logs and above (i.e., also
        `WARNING, ERROR` and `CRITICAL`)
        displayed in the console; the file will also include `DEBUG` logs.
    """
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s %(lineno)d - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(log_level_for_console.upper())
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        fh = logging.FileHandler(save_dir + f"/{logger_name}.txt")
        fh.setLevel(log_level_for_file.upper())
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_execution(func: Callable) -> Callable:
    """Decorator to log the execution of a function.
    Only records entry to and exit from the function, to the console.
    """
    loglevel = logging.info

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loglevel(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        loglevel(f"Finished executing {func.__name__}")
        return result

    return wrapper

    return wrapper
    return wrapper
