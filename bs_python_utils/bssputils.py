"""
Contains `scipy` utility programs:

* `describe_array`: report descriptive statistics on a vectorized array
* `spline_reg`: spline interpolation in one dimension.
"""

from math import sqrt
from typing import Any, cast

import numpy as np
import scipy.stats as sts
from scipy.interpolate import UnivariateSpline

from bs_python_utils.bsnputils import check_vector, rice_stderr
from bs_python_utils.bsutils import bs_error_abort, print_stars


def describe_array(v: np.ndarray, name: str | None = "v") -> Any:
    """Print a summary of descriptive statistics for a 1-D array.

    Args:
        v: Array to summarise (flattened to one dimension).
        name: Optional label printed alongside the summary.

    Returns:
        The ``scipy.stats.describe`` result for ``v``.
    """
    print_stars(f"{name} has:")
    d = sts.describe(v, None)
    print(f"Number of elements: {d.nobs}")
    print(f"Minimum: {d.minmax[0]}")
    print(f"Maximum: {d.minmax[1]}")
    print(f"Mean: {d.mean}")
    print(f"Std deviation: {sqrt(d.variance)}")
    return d


def spline_reg(
    y: np.ndarray,
    x: np.ndarray,
    x_new: np.ndarray | None = None,
    is_sorted: bool | None = False,
    smooth: bool | None = True,
) -> np.ndarray:
    """Interpolate ``y`` as a function of ``x`` using univariate splines.

    Args:
        y: Observed response values.
        x: Covariate locations.
        x_new: Evaluation points; defaults to ``x``.
        is_sorted: Set to ``True`` when ``x`` is already sorted in ascending order.
        smooth: When ``True`` estimate weights via ``rice_stderr`` and fit a smoothing spline;
            otherwise enforce an interpolating spline with ``s=0``.

    Returns:
        The interpolated values evaluated at ``x_new``.
    """
    n = check_vector(x)
    ny = check_vector(y)
    if ny != n:
        bs_error_abort("x and y should have the same size")

    if not is_sorted:
        # need to sort by increasing value of x
        order_rhs = np.argsort(x)
        rhs = x[order_rhs]
        lhs = y[order_rhs]
    else:
        rhs, lhs = x, y

    if smooth:
        # we compute a local estimator of the stderr of (y | x) and we use it to enter weights
        sigyx = rice_stderr(lhs, rhs)
        w = 1 / sigyx
        spl = UnivariateSpline(rhs, lhs, w=w)
    else:
        spl = UnivariateSpline(rhs, lhs, s=0)

    xeval = x if x_new is None else x_new
    y_pred = spl(xeval)
    return cast(np.ndarray, y_pred)
