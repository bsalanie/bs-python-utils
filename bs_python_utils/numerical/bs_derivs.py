from collections.abc import Callable


def richardson_derivative(
    f: Callable[[float, list], float], x: float, args: list, h: float
) -> float:
    """Compute the first derivative of $f(x, args)$ at $x$ via Richardson extrapolation.

    Uses a 5-point central-difference stencil (step `h` and `2h`) that is
    fourth-order accurate.

    Args:
        f: The function to differentiate, called as `f(x, args)`.
        x: The point at which to differentiate.
        args: Other arguments passed through to `f`.
        h: The step size.

    Returns:
        An approximation of the first derivative of `f` at `x`.
    """
    f1 = f(x + h, args)
    f2 = f(x - h, args)
    f3 = f(x + 2.0 * h, args)
    f4 = f(x - 2.0 * h, args)
    return (8.0 * (f1 - f2) - (f3 - f4)) / (12.0 * h)


def richardson_second_derivative(
    f: Callable[[float, list], float], x: float, args: list, h: float
) -> float:
    """Compute the second derivative of $f(x, args)$ at $x$ via Richardson extrapolation.

    Applies `richardson_derivative` to the (Richardson-extrapolated) first
    derivative of `f`.

    Args:
        f: The function to differentiate, called as `f(x, args)`.
        x: The point at which to differentiate.
        args: Other arguments passed through to `f`.
        h: The step size.

    Returns:
        An approximation of the second derivative of `f` at `x`.
    """

    def fprime(y, a):
        return richardson_derivative(f, y, a, h)

    return richardson_derivative(fprime, x, args, h)
