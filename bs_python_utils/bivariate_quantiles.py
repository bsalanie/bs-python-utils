"""This takes in observations of a bivariate random variable `y`
and computes vector quantiles and vector ranks à la
[Chernozhukov-Galichon-Hallin-Henry (*Ann. Stats.* 2017)](
https://projecteuclid.org/journals/annals-of-statistics/volume-45/
issue-1/MongeKantorovich-depth-quantiles-ranks-and-signs/10.1214/16-AOS1450.full).


Note:
    if the math looks strange in the documentation, just reload the page.

The sequence of steps is as follows:

* choose a  number of Chebyshev nodes for numerical integration and optimize
  the weights: `v = solve_for_v(y, n_nodes)`
* to obtain the $(u_1,u_2)$ quantiles for $(u_1, u_2)\\in [0,1]$, run
`qtiles_y = bivariate_quantiles_v(y, v, u1, u2)`
* to compute the vector ranks for all points in the sample (the barycenters
  of the cells in the power diagram):
`ranks_y = bivariate_ranks_v(y, v, n_nodes)`

Steps 1 and 2 can be combined: `qtiles_y = bivariate_quantiles(y, v, u1, u2, n_nodes)`

Steps 1 and 3 can be combined: `ranks_y = bivariate_ranks(y, n_nodes)`
"""

from typing import cast

import numpy as np

from bs_python_utils.bs_opt import minimize_free, print_optimization_results
from bs_python_utils.bsnputils import TwoArrays, npmaxabs
from bs_python_utils.bsutils import bs_error_abort
from bs_python_utils.chebyshev import Interval, cheb_get_nodes_1d


def _compute_ab(y_sorted: np.ndarray, v_sorted: np.ndarray) -> TwoArrays:
    """Build the `A` and `B` matrices used in the dual optimisation."""
    y1 = y_sorted[:, 0]
    dy1 = np.subtract.outer(y1, y1)
    y2 = y_sorted[:, 1]
    dy2 = np.subtract.outer(y2, y2)
    np.fill_diagonal(dy2, 1.0)
    dv = np.subtract.outer(v_sorted, v_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_mat = np.divide(dy1.T, dy2, where=np.abs(dy2) > 1e-12)
        b_mat = np.divide(dv.T, dy2, where=np.abs(dy2) > 1e-12)
    a_mat = np.nan_to_num(a_mat, nan=0.0, posinf=0.0, neginf=0.0)
    b_mat = np.nan_to_num(b_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return a_mat, b_mat


def _compute_u2_bounds(
    k: int, u1: np.ndarray, a_mat: np.ndarray, b_mat: np.ndarray
) -> TwoArrays:
    """Return the admissible interval of ``u2`` that selects index ``k``."""
    n = a_mat.shape[0]
    m = u1.size
    if k == 0:
        left_bound = np.zeros(m)
        a_right = a_mat[0, 1:]
        b_right = b_mat[0, 1:]
        if a_right.size:
            right_bound = np.min(np.outer(u1, a_right) - b_right, 1)
        else:
            right_bound = np.ones(m)
    elif 1 <= k < n - 1:
        a_left = a_mat[k, :k]
        b_left = b_mat[k, :k]
        if a_left.size:
            left_bound = np.max(np.outer(u1, a_left) - b_left, 1)
        else:
            left_bound = np.zeros(m)
        a_right = a_mat[k, (k + 1) :]
        b_right = b_mat[k, (k + 1) :]
        if a_right.size:
            right_bound = np.min(np.outer(u1, a_right) - b_right, 1)
        else:
            right_bound = np.ones(m)
    elif k == n - 1:
        a_left = a_mat[-1, :-1]
        b_left = b_mat[-1, :-1]
        if a_left.size:
            left_bound = np.max(np.outer(u1, a_left) - b_left, 1)
        else:
            left_bound = np.zeros(m)
        right_bound = np.ones(m)
    else:
        bs_error_abort(f"{k=} is not compatible with {n=}")
    left_bound = np.clip(left_bound, 0.0, 1.0)
    right_bound = np.clip(right_bound, 0.0, 1.0)

    return left_bound, right_bound


def bivariate_quantiles_v(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate vector quantiles for a given set of dual weights.

    Args:
        y: Observations with shape ``(n, 2)``.
        u: Evaluation points in ``[0, 1]^2`` (shape ``(m, 2)``).
        v: Dual weights solving the optimal transport problem (length ``n``).

    Returns:
        Array of quantile locations with shape ``(m, 2)``.
    """
    u = np.atleast_2d(u)
    if u.shape[1] != 2:
        bs_error_abort("u must have two columns")
    m = u.shape[0]
    q = np.empty((m, 2))
    block = max(1, min(m, 5_000))
    for start in range(0, m, block):
        stop = min(start + block, m)
        chunk = u[start:stop]
        net_val = chunk @ y.T - v
        k_max = np.argmax(net_val, axis=1)
        q[start:stop] = y[k_max]
    return cast(np.ndarray, q)


def bivariate_ranks_v(
    y: np.ndarray, v: np.ndarray, n_nodes: int = 32, presorted: bool = False
) -> np.ndarray:
    """Compute the barycentric ranks of each observation given optimal weights.

    Args:
        y: Observations with shape ``(n, 2)``.
        v: Dual weights returned by ``solve_for_v_``.
        n_nodes: Number of Chebyshev nodes used in the quadrature.
        presorted: Set to ``True`` when ``y``/``v`` are pre-sorted by the
            second coordinate.

    Returns:
        Array of average ranks (shape ``(n, 2)``) with ``nan`` for zero-mass cells.
    """
    n, d = y.shape

    if d != 2:
        bs_error_abort(f"only works for 2-dimensional y, not for {d}")

    interval01 = Interval(0.0, 1.0)
    u1_nodes, u1_weights = cheb_get_nodes_1d(interval01, n_nodes)

    if presorted:
        sort_order = np.arange(n)
        y_sorted = y
        v_sorted = v
    else:
        sort_order = np.argsort(y[:, 1])
        y_sorted = y[sort_order, :]
        v_sorted = v[sort_order]

    a_mat, b_mat = _compute_ab(y_sorted, v_sorted)

    average_ranks = np.zeros((n, 2))

    for k in range(n):
        left_bounds, right_bounds = _compute_u2_bounds(k, u1_nodes, a_mat, b_mat)
        pos_diffs = np.maximum(right_bounds - left_bounds, 0.0)
        pos_diffs_sq = np.maximum(
            right_bounds * right_bounds - left_bounds * left_bounds, 0.0
        )
        prob_k = pos_diffs @ u1_weights
        if prob_k <= 1e-12:
            average_ranks[sort_order[k], :] = np.array([np.nan, np.nan])
            continue
        average_ranks[sort_order[k], 0] = ((u1_nodes * pos_diffs) @ u1_weights) / prob_k
        average_ranks[sort_order[k], 1] = ((pos_diffs_sq @ u1_weights) / 2.0) / prob_k

    return average_ranks


def _objgrad(
    v_sorted: np.ndarray, args: list, gr: bool = False
) -> float | tuple[float, np.ndarray]:
    """computes the expectation of $\\psi(U, v)$ and perhaps its gradient wrt `v`

    Args:
        v_sorted: an `n`-vector of weights, sorted by increasing `y[:, 1]`
        args: a list of other arguments `[y_sorted, u1_nodes, u1_weights, verbose]`
        gr: if `True`, we also evaluate the gradient

    Returns:
        the value of the expectation and perhaps its gradient
    """
    y_sorted = args[0]
    n = y_sorted.shape[0]
    u1_nodes = args[1]
    u1_weights = args[2]
    vs1 = np.append(v_sorted, -np.sum(v_sorted))
    a_mat, b_mat = _compute_ab(y_sorted, vs1)

    obj_val = 0.0
    probs = np.zeros(n)
    for k in range(n):
        left_bounds, right_bounds = _compute_u2_bounds(k, u1_nodes, a_mat, b_mat)
        pos_diffs = np.maximum(right_bounds - left_bounds, 0.0)
        pos_diffs_sq = np.maximum(
            right_bounds * right_bounds - left_bounds * left_bounds, 0.0
        )
        obj_val += (
            y_sorted[k, 0] * ((u1_nodes * pos_diffs) @ u1_weights)
            + y_sorted[k, 1] * (pos_diffs_sq @ u1_weights) / 2.0
        )
        probs[k] = pos_diffs @ u1_weights
        obj_val -= vs1[k] * probs[k]

    if gr:
        grad_val = probs[-1] - probs[:-1]
        return obj_val, grad_val
    else:
        return cast(float, obj_val)


def _obj(v_sorted: np.ndarray, args: list):
    return _objgrad(v_sorted, args)


def _grad(v_sorted: np.ndarray, args: list):
    res_objg = cast(tuple[float, np.ndarray], _objgrad(v_sorted, args, gr=True))
    grad_val = res_objg[1]
    verbose = args[3]
    if verbose:
        print(f"The error on the gradient is {npmaxabs(grad_val)}")
    return grad_val


def solve_for_v_(y: np.ndarray, n_nodes: int = 32, verbose: bool = False) -> np.ndarray:
    """Solve the dual optimisation to obtain the optimal weights ``v``.

    Args:
        y: Observations with shape ``(n, 2)``.
        n_nodes: Number of Chebyshev nodes for the quadrature.
        verbose: Print optimisation diagnostics when ``True``.

    Returns:
        Array of length ``n`` containing the optimal weights (including the
            residual term).
    """
    n, d = y.shape

    if d != 2:
        bs_error_abort(f"only works for 2-dimensional y, not for {d}")

    # sort by increasing y[:, 1]
    sort_order = np.argsort(y[:, 1])
    y_sorted = y[sort_order, :]

    v0 = np.mean(y_sorted[:-1, :], 1)

    interval01 = Interval(0.0, 1.0)
    u1_nodes, u1_weights = cheb_get_nodes_1d(interval01, n_nodes)

    argsog = [y_sorted, u1_nodes, u1_weights, verbose]

    res = minimize_free(_obj, _grad, v0, args=argsog)
    if verbose:
        print_optimization_results(res, "Minimizing over v")

    if not res.success:
        bs_error_abort("Problem! the optimization failed.")
    vstar = res.x
    if verbose:
        print(f"The final gradient over v is close to 0: error {npmaxabs(res.jac)}")
    vstar1_sorted = np.append(vstar, -np.sum(vstar))

    # revert to original order
    vstar1 = np.zeros_like(vstar1_sorted)
    vstar1[sort_order] = vstar1_sorted

    return vstar1


def bivariate_quantiles(
    y: np.ndarray, u: np.ndarray, n_nodes: int = 32, verbose: bool = False
) -> np.ndarray:
    """Solve for the dual weights then evaluate bivariate quantiles.

    Args:
        y: Observations, shape ``(n, 2)``.
        u: Query points in ``[0, 1]^2`` (shape ``(m, 2)``).
        n_nodes: Number of Chebyshev nodes for the quadrature.
        verbose: Print optimisation diagnostics when ``True``.

    Returns:
        Bivariate quantiles at ``u``.
    """
    v = solve_for_v_(y, n_nodes, verbose)
    return bivariate_quantiles_v(y, u, v)


def bivariate_ranks(
    y: np.ndarray, n_nodes: int = 32, verbose: bool = False
) -> np.ndarray:
    """Compute ranks by first solving for the optimal weights ``v``.

    Args:
        y: Observations, shape ``(n, 2)``.
        n_nodes: Number of Chebyshev nodes for the quadrature.
        verbose: Print optimisation diagnostics when ``True``.

    Returns:
        Average ranks with shape ``(n, 2)``.
    """
    v = solve_for_v_(y, n_nodes, verbose)
    return bivariate_ranks_v(y, v, n_nodes)
