"""
Utilities to time code:

* a `Timer` class that can be used as a context manager
* a `timeit` decorator for functions.
"""

import time
from functools import wraps
from typing import Any, Callable, Iterable


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


class Timer:
    """
    A timer that can be started, stopped, and reset as needed by the user.
    It keeps track of the total elapsed time in the `elapsed` attribute::

    Examples:
        >>> with Timer() as t:
        >>>  ....
        >>> print(f"... took {t.elapsed} seconds")

    use `Timer(time.process_time)` to get only CPU time.

    can also do:

    Examples:
        >>> t = Timer()
        >>> t.start()
        >>> t.stop()
        >>> t.start()   # will add to the same counter
        >>> t.stop()
        >>> print(f"{t.elapsed} seconds total")
    """

    def __init__(self, func: Callable = time.perf_counter) -> None:
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self) -> None:
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self) -> None:
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self) -> None:
        self.elapsed = 0.0

    @property
    def running(self) -> bool:
        return self._start is not None

    def __enter__(self) -> Any:
        self.start()
        return self

    def __exit__(self, *args: Iterable) -> None:
        self.stop()
