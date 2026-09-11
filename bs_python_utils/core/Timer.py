"""
Utilities to time code:

* a `Timer` class that can be used as a context manager
* a `timeit` decorator for functions.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar


def timeit(func: Callable) -> Callable:
    """
    Decorator to time a function
    """

    @wraps(func)
    def wrapper(*args: Iterable, **kwargs: dict) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.3f} seconds")
        return result

    return wrapper


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    """A flexible timer class for measuring execution time.

    This class can be used in three ways:

    1. **As a context manager**: automatically starts and stops timing
    2. **With manual calls**: call start() and stop() explicitly
    3. **As a named timer**: accumulate timing across multiple calls using the `name` attribute

    Attributes:
        name: Optional name for the timer. If provided, cumulative elapsed time is stored
            in the class-level `timers` dict. Use None for one-off timings.
        text: Format string for displaying elapsed time. Defaults to "Elapsed time: {:0.4f} seconds".
            The string is formatted with a single float representing elapsed seconds.
        logger: Callable to handle output. Defaults to print(). Set to None to suppress output.
        timers: Class-level dictionary storing cumulative times for all named timers.

    Raises:
        TimerError: If stop() is called without start(), or start() is called on a running timer.

    Example:
        >>> # As a context manager (recommended for simple use cases)
        >>> with Timer():
        ...     time.sleep(0.1)  # doctest: +SKIP
        Elapsed time: 0.1000 seconds

        >>> # With a custom message
        >>> with Timer(text="My code took {:0.2f}s"):
        ...     time.sleep(0.05)  # doctest: +SKIP
        My code took 0.05s

        >>> # Suppress output
        >>> with Timer(logger=None):
        ...     time.sleep(0.1)  # doctest: +SKIP

        >>> # Suppress output and capture elapsed time
        >>> with Timer(logger=None) as t:
        ...     time.sleep(0.1)  # doctest: +SKIP
        >>> elapsed = t.stop()

        >>> # Manual start/stop
        >>> timer = Timer()
        >>> timer.start()  # doctest: +SKIP
        >>> time.sleep(0.1)  # doctest: +SKIP
        >>> elapsed = timer.stop()  # doctest: +SKIP
        Elapsed time: 0.1000 seconds

        >>> # Named timer for cumulative timing
        >>> Timer(name="db_queries").start()  # doctest: +SKIP
        >>> time.sleep(0.05)  # doctest: +SKIP
        >>> Timer(name="db_queries").stop()  # doctest: +SKIP
        Elapsed time: 0.0500 seconds
        >>> Timer.timers  # doctest: +SKIP
        {'db_queries': 0.05}
    """

    timers: ClassVar = {}
    name: Any = None
    text: Any = "Elapsed time: {:0.4f} seconds"
    logger: Any = print
    _start_time: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize the timer and register it if named.

        For named timers, creates an entry in the class-level `timers` dict
        to accumulate elapsed time across multiple start/stop cycles.
        """
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self):
        """Start the timer.

        Records the current time using perf_counter() for high precision.

        Raises:
            TimerError: If the timer is already running.
        """
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and report elapsed time.

        Calculates elapsed time since start() was called, sends output to the logger
        if configured, and accumulates the time if this is a named timer.

        Returns:
            Elapsed time in seconds as a float.

        Raises:
            TimerError: If the timer is not running.
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self):
        """Enter context manager: start the timer.

        Returns:
            self: The Timer instance.
        """
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Exit context manager: stop the timer.

        Stops the timer regardless of whether an exception occurred in the with block.
        """
        self.stop()
