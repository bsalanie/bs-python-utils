"""
Contains various utilities programs:

* `printargs`: a decorator that reports the arguments of the function it decorates
* `bs_name_func`: returns the name of the current function or its callers
* `bs_error_abort`: reports on error and aborts execution
* `final_s`: whether a word should have a final 's'
* `bs_switch`: an improved `switch` statement
* `find_first`: returns the index of and the first item in an iterable that satisfies a condition
* `print_stars`: prints a title within lines of stars
* `file_print_stars`: does the same, to a file
* `fstring_integer_with_significant_digits`: rounds an integer and returns an f-string
* `mkdir_if_needed`: creates a directory if it does not exist
* `bscomb`: a combination ${n \\choose k}$ operator
* `bslog, bsxlogx, bsexp`: $C^2$ extensions of $\\log(x), x\\log x, \\exp(x)$, and their first two derivatives
* `bs_projection_point`: projects a point on a line in the plane.

Note:
    if the math looks strange in the documentation, just reload the page.
"""

import sys
import traceback
from functools import wraps
from io import TextIOBase
from math import exp, factorial, log, sqrt
from pathlib import Path
from typing import Any, Callable, Iterable, cast

TwoFloats = tuple[float, float]
ThreeFloats = tuple[float, float, float]


def printargs(func: Callable) -> Callable:
    """
    Decorator that reports the arguments of the function

    Args:
      func: the decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Function {func.__name__} called with args = {args} and kwargs = {kwargs}"
        )
        return func(*args, **kwargs)

    return wrapper


def bs_name_func(back: int = 2) -> str:
    """
    get the name of the current function, or further back in stack

    Args:
        back: 2 is current function, 3 the function that called it etc

    Returns:
        the name of the function
    """
    stack = traceback.extract_stack()
    *_, func_name, _ = stack[-back]
    return cast(str, func_name)


def bs_error_abort(msg: str = "error, aborting") -> None:
    """
    report error and abort

    Args:
        msg: a message

    Returns:
        prints the message and exits with code 1
    """
    print_stars(f"{bs_name_func(3)}: {msg}")
    sys.exit(1)


def final_s(n: int, word: str) -> str:
    """
    pluralizes word if n > 1

    Args:
        n: how many times
        word: to be pluralized, maybe

    Returns:
        `1 word` or `n words`
    """
    return f"{n} {word}{'s' if n > 1 else ''}"


def fstring_integer_with_significant_digits(number: int, m: int) -> str:
    """returns an f-string with `number` rounded to `m` significant digits

    Args:
        number: the integer we want to round
        m:  how many digits we keep; the rest is filled with zeroes

    Return:
        a string with the rounded integer

    Examples:
    >>> fstring_integer_with_significant_digits(12345, 0)
    ' 0'
    >>> fstring_integer_with_significant_digits(12345, 3)
    '12,300'
    >>> fstring_integer_with_significant_digits(12345, 7)
    '12345'
    """
    if (not isinstance(number, int)) or (not isinstance(m, int)):
        bs_error_abort("Both arguments should be integers.")
    if number == 0:
        return f"{number:d}"
    else:
        digits = len(str(abs(int(number))))
        if digits <= m:
            return f"{int(number):d}"
        else:
            power = digits - m
            rounded = round(number / 10**power) * 10**power
            return f"{int(rounded): ,d}"


def bs_switch(
    match: str, dico: dict, strict: bool = True, default: Any = "no match"
) -> Any:
    """
    a switch statement that allows for partial matches if strict is False

    Args:
        match: what we look for in the keys
        dico: a dictionary with string keys
        strict: if `False`, we accept a partial match
        default: what we return if no match is found

    Returns:
        the value for the match, or `default`

    Examples:
        >>> calc_dict = {
        "plus": lambda x, y: x + y,
        "minus": lambda x, y: x - y
        }
        >>> plus = bs_switch('plus', calc_dict, default="unintended function")
        >>> minus = bs_switch('min', calc_dict, strict=False, default="unintended function")
        >>> plus(6, 4)
        10
        >>> minus(6, 4)
        2
        >>> bs_switch('plu', calc_dict)
        "no match"
    """
    if strict:
        return dico.get(match, default)
    else:
        try:
            key = next(k for k in dico if match in k)
            return dico[key]
        except StopIteration:
            return default


def find_first(iterable: Iterable, condition: Callable = lambda x: True) -> Any:
    """
    Returns the index of and the first item in the `iterable` that
    satisfies the `condition`.

    Args:
        iterable: where to look
        condition: must return a boolean

    Returns:
        If the condition is not given, returns 0 and the first item of
        the iterable.

        Raises `StopIteration` if no item satisfyng the condition is found.

    Examples:
        >>> find_first( (1,2,3), condition=lambda x: x % 2 == 0)
        (1, 2)
        >>> find_first(range(3, 100))
        (0, 3)
        >>> find_first( () )
        Traceback (most recent call last):
        ...
        StopIteration


    """
    return next((i, x) for i, x in enumerate(iterable) if condition(x))


def print_stars(title: str | None = None, n: int = 70) -> None:
    """Print a horizontal bar of ``*`` characters, optionally framing a title.

    Args:
        title: Optional text centred between two star lines.
        n: Length of the star line.
    """
    line_stars = "*" * n
    print()
    print(line_stars)
    if title:
        print(title.center(n))
        print(line_stars)
    print()


def file_print_stars(
    file_handle: TextIOBase, title: str | None = None, n: int = 70
) -> None:
    """Write the ``print_stars`` output to a file-like object."""
    line_stars = "*" * n
    file_handle.write("\n")
    file_handle.write(line_stars)
    file_handle.write("\n")
    if title:
        file_handle.write(title.center(n))
        file_handle.write("\n")
        file_handle.write(line_stars)
        file_handle.write("\n")
    file_handle.write("\n")


def mkdir_if_needed(p: Path | str) -> Path:
    """
    create the directory if it does not exist

    Args:
        p: a path

    Returns:
        the directory Path

    """
    try:
        q = Path(p)
    except OSError:
        bs_error_abort(f"{p} is not a path")
    if not q.exists():
        q.mkdir(parents=True)
    return q


def bscomb(n: int, k: int) -> int:
    """
    number of combinations of k among n `{n \\choose k}`

    Args:
        n:
        k: should be smaller than n

    Returns:
        `{n \\choose k}`.
    """
    if not isinstance(n, int):
        bs_error_abort(f"n should be an integer, not {n}")
    if not isinstance(k, int):
        bs_error_abort(f"k should be an integer, not {k}")
    if n < k:
        bs_error_abort(f"k={k} should not be larger than n={n}")
    return factorial(n) // (factorial(k) * factorial(n - k))


def bslog(
    x: float, eps: float = 1e-30, deriv: int = 0
) -> float | TwoFloats | ThreeFloats:
    """
    extends the logarithm below `eps` by taking a second-order approximation
    perhaps with derivatives

    Args:
        x: argument
        eps: lower bound
        deriv: if 1, also return first derivative; if 2, the first two derivatives

    Returns:
        `\\ln(x)` `C^2`-extended below `eps`, perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv can only be 0, 1, or 2; not {deriv}")
    if x > eps:
        logx = log(x)
        if deriv == 0:
            return logx
        dlogx = 1.0 / x
        if deriv == 1:
            return logx, dlogx
        d2logx = -dlogx * dlogx
        return logx, dlogx, d2logx
    else:
        dx = 1.0 - x / eps
        log_smaller = log(eps) - dx - dx * dx / 2.0
        if deriv == 0:
            return log_smaller
        dlog_smaller = (1.0 + dx) / eps
        if deriv == 1:
            return log_smaller, dlog_smaller
        d2log_smaller = -1.0 / eps / eps
        return log_smaller, dlog_smaller, d2log_smaller


def bsxlogx(
    x: float, eps: float = 1e-30, deriv: int = 0
) -> float | TwoFloats | ThreeFloats:
    """
    extends `x \\ln(x)` below `eps` by making it go to zero in a `C^2` extension
    perhaps with derivatives

    Args:
        x: argument
        eps: lower bound
        deriv: if 1, also return first derivative; if 2, the first two derivatives

    Returns:
        `x \\ln(x)`  `C^2`-extended below `eps`, perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv can only be 0, 1, or 2; not {deriv}")
    if x > eps:
        logx = log(x)
        if deriv == 0:
            return x * logx
        if deriv == 1:
            return x * logx, 1.0 + logx
        return x * logx, 1.0 + logx, 1.0 / x
    else:
        logeps = log(eps)
        dx = x / eps
        log_smaller = x * logeps - eps / 2.0 + x * dx / 2.0
        if deriv == 0:
            return log_smaller
        if deriv == 1:
            return log_smaller, logeps + dx
        return log_smaller, logeps + dx, 1.0 / eps


def _bsexp_extend(x: float, deriv: int, limx: float) -> float | TwoFloats | ThreeFloats:
    """extends the exponential C^2-wise beyond limx"""
    elimx = exp(limx)
    dx = x - limx
    exp_extend = elimx * (1.0 + dx * (1.0 + 0.5 * dx))
    if deriv == 0:
        return exp_extend
    dexp_extend = elimx * (1.0 + dx)
    if deriv == 1:
        return exp_extend, dexp_extend
    # deriv = 2
    return exp_extend, dexp_extend, elimx


def bsexp(
    x: float,
    bigx: float = 50.0,
    lowx: float = -50.0,
    deriv: int = 0,
) -> float | TwoFloats | ThreeFloats:
    """
    `C^2`-extends the exponential above `bigx` and below `lowx`
    perhaps with derivatives

    Args:
        x: argument
        bigx: upper bound
        lowx: lower bound
        deriv: if 1, also return first derivative; if 2, the first two derivatives

    Returns:
        the exponential `C^2`-extended above `bigx` and below `lowx`
        perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv can only be 0, 1, or 2; not {deriv}")
    if lowx < x < bigx:
        expx = exp(x)
        if deriv == 0:
            return expx
        if deriv == 1:
            return expx, expx
        return expx, expx, expx
    elif x < lowx:
        return _bsexp_extend(x, deriv, lowx)
    else:
        return _bsexp_extend(x, deriv, bigx)


def bs_projection_point(
    x: float, y: float, a: float, b: float, c: float
) -> tuple[float, float, float]:
    """
    projection of point (x,y) on line ax+by+c=0

    Args:
        x: y: coordinates
        a: b: c: line parameters (as in ax+by+c=0)

    Returns:
        x_proj: y_proj: coordinates of projection point
        dist: distance of point from line
    """
    a2b2 = a * a + b * b
    denom = sqrt(a2b2)
    value = a * x + b * y + c
    x_proj = x - a * value / a2b2
    y_proj = y - b * value / a2b2
    dist = abs(value) / denom
    return x_proj, y_proj, dist
