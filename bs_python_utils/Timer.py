"""
utilities to time code
"""

import time
from functools import wraps
from typing import Callable


def timeit(func: Callable) -> Callable:
    """
    Decorator to time a function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result

    return wrapper


class Timer:
    """
    A timer that can be started, stopped, and reset as needed by the user.
    It keeps track of the total elapsed time in the `elapsed` attribute::

       with Timer() as t:
         ....
       print(f"... took {t.elapsed} seconds")

    use `Timer(time.process_time)` to get only CPU time.

    can also do::

       t = Timer()
       t.start()
       t.stop()
       t.start()   # will add to the same counter
       t.stop()
       print(f"{t.elapsed} seconds total")
    """

    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError("Already started")
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError("Not started")
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
