"""Bivariate vector quantiles and ranks.

This module implements the bivariate case of the vector quantiles and vector
ranks construction of Chernozhukov, Galichon, Hallin, and Henry (2017).

The main workflow is:

1. Solve for the dual weights ``v`` with :func:`_solve_for_v`.
2. Evaluate quantiles with :func:`bivariate_quantiles_v` or
   :func:`bivariate_quantiles`.
3. Read off barycentric ranks with :func:`bivariate_ranks`.

References:
    Chernozhukov, Galichon, Hallin, and Henry. "Monge-Kantorovich Depth,
    Quantiles, Ranks and Signs." *Annals of Statistics* 45(1), 2017.
"""

from typing import cast

import numpy as np

from bs_python_utils.core.bsutils import bs_error_abort
from bs_python_utils.numerical.bsnputils import TwoArrays, npmaxabs
from bs_python_utils.numerical.chebyshev import Interval, cheb_get_nodes_1d
from bs_python_utils.opt.bs_opt import minimize_free, print_optimization_results


def _compute_ad(y: np.ndarray) -> TwoArrays:
    """Compute matrices used in the dual optimization.

    Args:
        y: Observations with shape ``(n, 2)``.

    Returns:
        A tuple ``(a_mat, dy2)`` where ``a_mat`` contains the pairwise slope
        ratios and ``dy2`` contains pairwise differences in the second
        coordinate.
    """
    y1 = y[:, 0]
    dy1 = np.subtract.outer(y1, y1)
    y2 = y[:, 1]
    dy2 = np.subtract.outer(y2, y2)
    np.fill_diagonal(dy2, 1.0)
    dy2 = dy2.T
    a_mat = np.divide(dy1, dy2)
    return a_mat, dy2


def _compute_m_M(
    v: np.ndarray, a_mat: np.ndarray, dy2: np.ndarray, tau1_nodes: np.ndarray
) -> TwoArrays:
    """Compute the lower and upper bounds used in the dual optimization.

    Args:
        v: Dual weights of length ``n``.
        a_mat: Pairwise slope matrix returned by :func:`_compute_ad`.
        dy2: Pairwise differences in the second coordinate.
        tau1_nodes: Quadrature nodes for the first coordinate.

    Returns:
        A tuple ``(m, M)`` with arrays of shape ``(n, len(tau1_nodes))``.

    Warning:
        The (rare) ``dy2 == 0`` case currently raises warnings in computing ``b_mat``.
        That logic works for the present implementation but should be refactored into a cleaner, more explicit
        branch.
    """
    dv = np.subtract.outer(v, v)
    b_mat = dv / dy2
    np.fill_diagonal(dy2, 0.0)
    EPS = 1e-12
    maskp = dy2 < EPS
    maskm = dy2 > -EPS
    n, n_nodes = v.size, tau1_nodes.size
    m_low = np.empty((n, n_nodes))
    m_high = np.empty((n, n_nodes))
    for i, tau1 in enumerate(tau1_nodes):
        f_mat = tau1 * a_mat - b_mat
        f_matp = f_mat.copy()
        f_matm = f_mat.copy()
        f_matp[maskp] = 1
        f_matm[maskm] = 0
        m_low[:, i] = np.max(f_matm, axis=1)
        m_high[:, i] = np.min(f_matp, axis=1)
    return np.clip(m_low, 0.0, 1.0), np.clip(m_high, 0.0, 1.0)


def bivariate_quantiles_v(y: np.ndarray, tau: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate vector quantiles for fixed dual weights.

    Args:
        y: Observations with shape ``(n, 2)``.
        tau: Evaluation points in ``[0, 1]^2`` with shape ``(m, 2)``.
        v: Dual weights solving the optimal transport problem, with length
            ``n``.

    Returns:
        Array of quantile locations with shape ``(m, 2)``.

    Raises:
        SystemExit: If ``tau`` does not have exactly two columns.
    """
    if tau.shape[1] != 2:
        bs_error_abort("tau must have two columns")
    q = y[np.argmax(tau @ y.T - v, axis=1), :]
    return cast(np.ndarray, q)


def _objgrad(
    v: np.ndarray, args: list, gr: bool = False
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the objective and optionally its gradient.

    Args:
        v: Free dual variables of length ``n - 1``.
        args: Positional arguments ``[y, a_mat, dy2, tau1_nodes, tau1_weights,
            verbose]``.
        gr: Whether to also return the gradient and the barycentric ranks.

    Returns:
        If ``gr`` is ``False``, returns the scalar objective value.

        If ``gr`` is ``True``, returns a tuple ``(obj_val, grad_val,
        bivrank)``.
    """
    y = args[0]
    y1 = y[:, 0]
    y2 = y[:, 1]
    n = y.shape[0]
    a_mat, dy2 = args[1], args[2]
    tau1_nodes = args[3]
    tau1_weights = args[4]
    vs1 = np.append(v, -np.sum(v))
    m, M = _compute_m_M(vs1, a_mat, dy2, tau1_nodes)
    # print(f"m is {m}")
    # print(f"M  is {M}")
    # import sys

    # sys.exit(1)

    EPS = 1e-12
    bivrank = np.zeros((n, 2))

    pos_diffs = np.maximum(M - m, 0.0)
    pos_diffs_sq = np.maximum(M * M - m * m, 0.0)
    probs = pos_diffs @ tau1_weights
    factor1 = (pos_diffs * tau1_nodes) @ tau1_weights
    factor2 = (pos_diffs_sq @ tau1_weights) / 2.0
    obj_val = y1 @ factor1 + y2 @ factor2 - vs1 @ probs

    if np.min(probs) > EPS:
        bivrank[:, 0] = factor1 / probs
        bivrank[:, 1] = factor2 / probs

    if gr:
        grad_val = probs[-1] - probs[:-1]
        return obj_val, grad_val, bivrank
    else:
        return cast(float, obj_val)


def _obj(v: np.ndarray, args: list):
    """Return only the objective value.

    Args:
        v: Free dual variables.
        args: Positional arguments passed to :func:`_objgrad`.

    Returns:
        Scalar objective value.
    """
    return _objgrad(v, args)


def _grad(v: np.ndarray, args: list):
    """Return only the objective gradient.

    Args:
        v: Free dual variables.
        args: Positional arguments passed to :func:`_objgrad`.

    Returns:
        Objective gradient.
    """
    res_objg = cast(tuple[float, np.ndarray], _objgrad(v, args, gr=True))
    grad_val = res_objg[1]
    verbose = args[-1]
    if verbose:
        print(f"The error on the gradient is {npmaxabs(grad_val)}")
    return grad_val


def _solve_for_v(y: np.ndarray, n_nodes: int = 32, verbose: bool = False) -> TwoArrays:
    """Solve the dual optimization problem for the sample.

    Args:
        y: Observations with shape ``(n, 2)``.
        n_nodes: Number of Chebyshev nodes used for quadrature.
        verbose: Print optimisation diagnostics when ``True``.

    Returns:
        A tuple ``(vstar, bivranks)`` where ``vstar`` has length ``n`` and
        ``bivranks`` has shape ``(n, 2)``.

    Raises:
        SystemExit: If the optimization fails or if ``y`` is not
            two-dimensional.
    """
    d = y.shape[1]

    if d != 2:
        bs_error_abort(f"only works for 2-dimensional y, not for {d}")

    v0 = np.mean(y[:-1, :], 1)

    interval01 = Interval(0.0, 1.0)
    tau1_nodes, tau1_weights = cheb_get_nodes_1d(interval01, n_nodes)

    a_mat, dy2 = _compute_ad(y)

    argsog = [y, a_mat, dy2, tau1_nodes, tau1_weights, verbose]

    res = minimize_free(_obj, _grad, v0, args=argsog)
    if verbose:
        print_optimization_results(res, "Minimizing over v")

    if not res.success:
        bs_error_abort("Problem! the optimization failed.")
    vstar = res.x
    if verbose:
        print(f"The final gradient over v is close to 0: error {npmaxabs(res.jac)}")
    _, _, bivranks = cast(tuple, _objgrad(vstar, argsog, gr=True))
    vstar = np.append(vstar, -np.sum(vstar))
    return cast(np.ndarray, vstar), cast(np.ndarray, bivranks)


def bivariate_ranks(
    y: np.ndarray,
    n_nodes: int = 32,
    verbose: bool = False,
) -> np.ndarray:
    """Compute barycentric ranks for each observation.

    Args:
        y: Observations with shape ``(n, 2)``.
        n_nodes: Number of Chebyshev nodes used in the quadrature.
        verbose: Print diagnostics when ``True``.

    Returns:
        Array of average ranks (shape ``(n, 2)``) with ``nan`` for zero-mass cells.
    """
    d = y.shape[1]

    if d != 2:
        bs_error_abort(f"only works for 2-dimensional y, not for {d}")

    _, bivranks = _solve_for_v(y, n_nodes, verbose)
    return cast(np.ndarray, bivranks)


def bivariate_quantiles(
    y: np.ndarray, tau: np.ndarray, n_nodes: int = 32, verbose: bool = False
) -> np.ndarray:
    """Solve for the dual weights and evaluate bivariate quantiles.

    Args:
        y: Observations with shape ``(n, 2)``.
        tau: Query points in ``[0, 1]^2`` with shape ``(m, 2)``.
        n_nodes: Number of Chebyshev nodes for the quadrature.
        verbose: Print optimisation diagnostics when ``True``.

    Returns:
        Bivariate quantiles evaluated at ``tau``.
    """
    v, _ = _solve_for_v(y, n_nodes, verbose)
    return bivariate_quantiles_v(y, tau, v)
