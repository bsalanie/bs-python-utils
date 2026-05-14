"""Interface to `scipy.optimize`:

* `ScalarFunctionAndGradient`, `ProximalFunction` type aliases
* an `OptimizeParams` class
* `check_gradient_scalar_function` checks whether an analytical gradient is correct
  (supports ``mode='jax'`` for exact AD-based checking when JAX is installed)
* `acc_grad_descent`: accelerated gradient descent for convex, possibly
  non-smooth functions
* `acc_grad_descent_jax`: JAX-accelerated variant using ``lax.while_loop``
  (available when JAX is installed)
* `minimize_some_fixed`: minimizes a function with some parameter values
  possibly fixed and some possibly within bounds, using L-BFGS-B
* `minimize_free`: minimizes a function with some parameter values possibly
  within bounds
* `dfp_update, bfgs_update`: compute updates to the inverse Hessian
* `armijo_alpha, barzilai_borwein_alpha`: two ways of computing the step
  length
* `print_optimization_results`, `print_constrained_optimization_results`
  format the results.
"""

from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import Any, Callable, Iterable, Optional, Union, cast

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    _JAX = True
except ImportError:
    _JAX = False

from bs_python_utils.numerical.bsnputils import TwoArrays, check_vector, npmaxabs
from bs_python_utils.core.bsutils import bs_error_abort, print_stars
from bs_python_utils.core.Timer import timeit

ScalarFunctionAndGradient = Callable[
    [np.ndarray, Iterable, Optional[bool]], Union[float, tuple[float, np.ndarray]]
]
"""Type of `f(v, args, gr)` that returns a scalar value and also a gradient if
`gr` is `True`.
"""


ProximalFunction = Callable[[np.ndarray, float, Iterable], np.ndarray]
"""Type of `h(x, t, pars)` that returns a vector value."""


@dataclass
class OptimizeParams:
    """used for optimization; combines values, bounds and initial values for a
    parameter vector
    """

    params_values: np.ndarray | None
    params_bounds: list[tuple] | None
    params_init: np.ndarray | None


def print_optimization_results(
    resus: spopt.OptimizeResult, title: str = "Minimizing"
) -> None:
    """print results from unconstrained optimization.

    Args:
        resus: results from optimization
        title: a title

    Returns:
        just prints.
    """
    print_stars(title)
    print(resus.message)
    if resus.success:
        print(f"Successful! in {resus.nit} iterations")
        print(f" evaluated {resus.nfev} function calls and {resus.njev} gradients")
        jac = getattr(resus, "jac", None)
        if jac is not None:
            print("\nMinimizer and grad_f:")
            x_arr = np.atleast_1d(resus.x)
            jac_arr = np.atleast_1d(jac)
            if x_arr.shape == jac_arr.shape:
                print(np.column_stack((x_arr, jac_arr)))
            else:
                print("x:", x_arr)
                print("grad_f:", jac_arr)
        print(f"Minimized value is {resus.fun}")
    else:
        print_stars("Minimization failed!")
    return


def print_constrained_optimization_results(
    resus: spopt.OptimizeResult,
    title: str = "Minimizing",
    print_constr: bool = False,
    print_multipliers: bool = False,
) -> None:
    """print results from constrained optimization.

    Args:
        resus: results from optimization
        title: a title
        print_constr: if `True`, print the values of the constraints
        print_multipliers: if `True`, print the values of the multipliers

    Returns:
        just prints.
    """
    print_stars(title)
    print(resus.message)
    if resus.success:
        print(f"Successful! in {resus.nit} iterations")
        print(f" evaluated {resus.nfev} functions and {resus.njev} gradients")
        print(f"Minimized value is {resus.fun}")
        if hasattr(resus, "optimality"):
            print(f"The Lagrangian norm is {resus.optimality}")
        if hasattr(resus, "constr_violation"):
            print(f"The largest constraint violation is {resus.constr_violation}")
        if print_multipliers and hasattr(resus, "v"):
            print(f"The multipliers are {resus.v}")
        if print_constr and hasattr(resus, "constr"):
            print(f"The values of the constraints are {resus.constr}")
    else:
        print_stars("Constrained minimization failed!")
    return


def armijo_alpha(
    f: Callable,
    grad_f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    args: Iterable,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 100,
    c1: float = 1e-4,
) -> float:
    """Compute an Armijo backtracking step for a descent direction.

    Args:
        f: Objective function.
        grad_f: Gradient of the objective.
        x: Current iterate.
        d: Candidate descent direction.
        args: Extra arguments passed to ``f``/``grad_f``.
        alpha_init: Initial step size to try.
        beta: Multiplicative shrinkage factor; must lie in ``(0, 1)``.
        max_iter: Maximum number of backtracking steps.
        c1: Armijo sufficient-decrease constant.

    Returns:
        Step size satisfying the Armijo condition; returns ``0.0`` when the
            supplied direction
        is numerically orthogonal to the gradient.
    """
    f0 = f(x, args)
    g0 = grad_f(x, args)
    directional_derivative = float(np.dot(g0, d))
    eps = 1e-12
    if directional_derivative >= 0.0:
        if abs(directional_derivative) <= eps:
            return 0.0
        bs_error_abort("Direction is not a descent direction for Armijo step.")
        return 0.0
    if not 0.0 < beta < 1.0:
        bs_error_abort("beta must lie in (0, 1) for Armijo step.")
        return 0.0
    if c1 <= 0.0 or c1 >= 1.0:
        bs_error_abort("c1 must lie in (0, 1) for Armijo step.")
        return 0.0
    alpha = alpha_init
    for _ in range(max_iter):
        x1 = x + alpha * d
        f1 = f(x1, args)
        if f1 <= f0 + c1 * alpha * directional_derivative:
            return alpha
        alpha *= beta
    bs_error_abort("Too many iterations in Armijo line search.")
    return 0.0


def barzilai_borwein_alpha(
    grad_f: Callable, x: np.ndarray, args: Iterable
) -> tuple[float, np.ndarray]:
    """Estimate a Barzilai–Borwein step size and return the gradient at ``x``.

    Args:
        grad_f: Gradient function.
        x: Current iterate.
        args: Extra arguments passed to the gradient.

    Returns:
        A tuple ``(alpha, gradient)`` where ``alpha`` is the safeguarded BB
            step size.
    """
    eps = 1e-12
    g = grad_f(x, args)
    norm_g = spla.norm(g)
    if norm_g < eps:
        return 1.0, g
    alpha = 1.0 / max(norm_g, eps)
    x_hat = x - alpha * g
    g_hat = grad_f(x_hat, args)
    diff_g = g - g_hat
    norm_dg = spla.norm(diff_g)
    if norm_dg < eps:
        return alpha, g
    numerator = np.abs(np.dot(x - x_hat, diff_g))
    denominator = max(norm_dg * norm_dg, eps)
    alpha = max(numerator / denominator, eps)
    return alpha, g


def check_gradient_scalar_function(
    fg: ScalarFunctionAndGradient,
    p: np.ndarray,
    args: Iterable,
    mode: str = "central",
    EPS: float = 1e-6,
) -> TwoArrays:
    """Checks the gradient of a scalar function.

    Args:
        fg: should return the scalar value, and the gradient if its `gr` argument is `True`
        p: where we are checking the gradient
        args: other arguments passed to `fg`
        mode: ``"central"``, ``"forward"``, or ``"jax"`` (exact AD via
            ``jax.grad``; requires JAX and a JAX-differentiable ``fg`` — the
            scalar returned by ``fg(..., gr=False)`` must be a JAX array, not
            wrapped in ``float()``)
        EPS: the step for forward or central finite differences (ignored for
            ``mode='jax'``)

    Returns:
        the analytic and numeric (or AD) gradients.
    """
    f0, f_grad = fg(p, args, gr=True)  # type: ignore
    f0 = cast(float, f0)

    print_stars("checking the gradient: analytic, numeric")

    g = np.zeros_like(p)
    if mode == "central":
        for i, x in enumerate(p):
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = cast(float, fg(p1, args, gr=False))  # type: ignore
            p1[i] -= 2.0 * EPS
            f_minus = cast(float, fg(p1, args, gr=False))  # type: ignore
            g[i] = (f_plus - f_minus) / (2.0 * EPS)
            print(f"{i}: {f_grad[i]}, {g[i]}")
    elif mode == "forward":
        for i, x in enumerate(p):
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = cast(float, fg(p1, args, gr=False))  # type: ignore
            g[i] = (f_plus - f0) / EPS
            print(f"{i}: {f_grad[i]}, {g[i]}")
    elif mode == "jax":
        if not _JAX:
            bs_error_abort("mode='jax' requires JAX; install bs-python-utils[jax]")
            return f_grad, g
        # fg must be JAX-differentiable (uses jnp ops internally)
        jax_grad_fn = jax.grad(lambda p_arr: fg(p_arr, args, gr=False))
        g = np.asarray(jax_grad_fn(jnp.array(p)))
        for i in range(len(p)):
            print(f"{i}: {f_grad[i]}, {g[i]}")
    else:
        bs_error_abort("mode must be 'central', 'forward', or 'jax'")

    return f_grad, g


@timeit
def acc_grad_descent(
    grad_f: Callable,
    x_init: np.ndarray,
    other_params: Iterable,
    prox_h: ProximalFunction | None = None,
    print_result: bool = False,
    verbose: bool = False,
    tol: float = 1e-9,
    alpha: float = 1.01,
    beta: float = 0.5,
    maxiter: int = 10000,
) -> tuple[np.ndarray, int]:
    """Accelerated gradient descent with optional proximal operator.

    Args:
        grad_f: Gradient of the smooth component; called as ``grad_f(x, other_params)``.
        x_init: Initial iterate.
        other_params: Extra arguments forwarded to ``grad_f`` and ``prox_h``.
        prox_h: Proximal operator for the nonsmooth term ``h``. Defaults to identity.
        print_result: Print a convergence message on exit.
        verbose: When ``True`` report gradient errors every 10 iterations.
        tol: Infinity-norm tolerance on the gradient for declaring convergence.
        alpha: Upper bound used when adapting the step size.
        beta: Lower bound used when adapting the step size.
        maxiter: Maximum number of iterations.

    Returns:
        A pair ``(x_star, status)`` where ``status`` is ``0`` on convergence and ``1`` otherwise.
    """

    # no proximal projection if no h
    local_prox_h: ProximalFunction = prox_h if prox_h else lambda x, t, p: x

    x = x_init.copy()
    y = x_init.copy()

    #  for stepsize we use Barzilai-Borwein
    t, g = barzilai_borwein_alpha(grad_f, y, other_params)

    grad_err_init = npmaxabs(g)

    if verbose:
        print(f"agd: grad_err_init={grad_err_init}")

    n_iter = 0
    theta = 1.0

    eps = 1e-12

    while n_iter < maxiter:
        grad_err = npmaxabs(g)
        if grad_err < tol:
            break
        xi = x
        yi = y
        x = y - t * g
        x = local_prox_h(x, t, other_params)

        theta = 2.0 / (1.0 + sqrt(1.0 + 4.0 / theta / theta))

        if np.dot(y - x, x - xi) > 0:  # wrong direction, we restart
            x = xi
            y = x
            theta = 1.0
        else:
            y = x + (1.0 - theta) * (x - xi)

        gi = g
        g = grad_f(y, other_params)
        diff_y = y - yi
        ndy = spla.norm(diff_y)
        denom = np.dot(diff_y, gi - g)
        if abs(denom) > eps:
            t_hat = 0.5 * ndy * ndy / abs(denom)
            t = min(alpha * t, max(beta * t, t_hat))
        else:
            t = min(alpha * t, max(beta * t, t))

        n_iter += 1

        if verbose and n_iter % 10 == 0:
            print(f" AGD with grad_err = {grad_err} after {n_iter} iterations")

    x_conv = y

    ret_code = 0 if grad_err < tol else 1

    if verbose or print_result:
        if ret_code == 0:
            print_stars(
                f" AGD converged with grad_err = {grad_err} after {n_iter} iterations"
            )
        else:
            print_stars(
                f" Problem in AGD: grad_err = {grad_err} after {n_iter} iterations"
            )

    return x_conv, ret_code


if _JAX:
    from functools import partial as _partial

    @_partial(jax.jit, static_argnums=(0, 1, 5, 6, 7, 8))
    def _agd_jit_loop(
        grad_f: Callable,
        local_prox_h: Callable,
        x0: "jnp.ndarray",
        g0: "jnp.ndarray",
        t0: "jnp.ndarray",
        tol: float,
        alpha: float,
        beta: float,
        maxiter: int,
        other_params: Any,
    ) -> tuple:
        """Inner JIT-compiled while loop for :func:`acc_grad_descent_jax`.

        ``grad_f``, ``local_prox_h``, ``tol``, ``alpha``, ``beta``, and
        ``maxiter`` are static: XLA compiles one program per distinct
        combination of these values and reuses it for all subsequent calls with
        the same shapes.
        """
        eps = 1e-12

        def cond_fn(state: tuple) -> "jnp.ndarray":
            _, _, g, _, _, n_iter = state
            return jnp.logical_and(n_iter < maxiter, jnp.max(jnp.abs(g)) >= tol)

        def body_fn(state: tuple) -> tuple:
            x, y, g, t, theta, n_iter = state
            xi, yi = x, y

            x_new = local_prox_h(y - t * g, t, other_params)
            new_theta = 2.0 / (1.0 + jnp.sqrt(1.0 + 4.0 / (theta * theta)))

            # Restart if the update step is in the wrong direction
            restart = jnp.dot(y - x_new, x_new - xi) > 0
            x_out = jnp.where(restart, xi, x_new)
            y_out = jnp.where(
                restart,
                xi,
                x_new + (1.0 - new_theta) * (x_new - xi),
            )
            theta_out = jnp.where(restart, jnp.array(1.0), new_theta)

            g_new = grad_f(y_out, other_params)

            # Barzilai–Borwein step size update
            diff_y = y_out - yi
            ndy_sq = jnp.dot(diff_y, diff_y)
            denom = jnp.dot(diff_y, g - g_new)
            t_hat = jnp.where(jnp.abs(denom) > eps, 0.5 * ndy_sq / jnp.abs(denom), t)
            t_out = jnp.clip(t_hat, beta * t, alpha * t)

            return x_out, y_out, g_new, t_out, theta_out, n_iter + 1

        return jax.lax.while_loop(
            cond_fn,
            body_fn,
            (x0, x0, g0, t0, jnp.array(1.0), jnp.array(0)),
        )

    def acc_grad_descent_jax(
        grad_f: Callable,
        x_init: np.ndarray,
        other_params: Any,
        prox_h: ProximalFunction | None = None,
        tol: float = 1e-9,
        alpha: float = 1.01,
        beta: float = 0.5,
        maxiter: int = 10000,
    ) -> tuple[np.ndarray, int]:
        """JAX-accelerated accelerated gradient descent via ``lax.while_loop``.

        Equivalent to :func:`acc_grad_descent` but compiles the entire iteration
        loop with XLA via ``jax.jit``, eliminating Python overhead per step.
        The speedup is most pronounced when ``grad_f`` is already JIT-compiled
        (e.g. with ``@jax.jit``) and the problem is large enough for XLA to
        amortise its dispatch overhead.  On CPU, expect a cross-over around
        **n ≈ 2000**; on GPU the JAX path wins for all practical n.  For small
        n, :func:`acc_grad_descent` with a plain NumPy ``grad_f`` is faster.

        ``other_params`` must be a pytree of JAX arrays (or Python scalars).
        Pass plain NumPy arrays through ``jnp.array(...)`` first.

        Verbose printing and the ``@timeit`` wrapper are intentionally absent:
        the loop body cannot contain Python side-effects.

        Args:
            grad_f: Gradient of the smooth component; must be JAX-traceable
                (uses JAX ops or decorated with ``@jax.jit``).
            x_init: Initial iterate (NumPy or JAX array).
            other_params: Extra arguments forwarded to ``grad_f`` and ``prox_h``;
                must be a pytree of JAX arrays.
            prox_h: Proximal operator for the non-smooth term; must be
                JAX-traceable if provided.  Defaults to identity.
            tol: Infinity-norm tolerance on the gradient for convergence.
            alpha: Upper-bound multiplier for the adaptive step size.
            beta: Lower-bound multiplier for the adaptive step size.
            maxiter: Maximum iterations (static — changing this value triggers
                XLA recompilation).

        Returns:
            A pair ``(x_star, status)`` where ``status`` is ``0`` on convergence
            and ``1`` otherwise.
        """
        local_prox_h: Callable = prox_h if prox_h else lambda x, t, p: x

        x0 = jnp.array(x_init, dtype=jnp.float64)
        eps = 1e-12

        # Barzilai–Borwein initial step (eager, feeds the JIT loop)
        g0 = grad_f(x0, other_params)
        norm_g = jnp.linalg.norm(g0)
        t0 = 1.0 / jnp.maximum(norm_g, eps)
        x_hat = x0 - t0 * g0
        g_hat = grad_f(x_hat, other_params)
        diff_g0 = g0 - g_hat
        norm_dg = jnp.linalg.norm(diff_g0)
        numerator = jnp.abs(jnp.dot(x0 - x_hat, diff_g0))
        t_bb = jnp.maximum(numerator / jnp.maximum(norm_dg * norm_dg, eps), eps)
        t0 = jnp.where(norm_dg > eps, t_bb, t0)

        _, y_final, g_final, _, _, _ = _agd_jit_loop(
            grad_f, local_prox_h, x0, g0, t0, tol, alpha, beta, maxiter, other_params
        )

        grad_err = float(jnp.max(jnp.abs(g_final)))
        return np.asarray(y_final), 0 if grad_err < tol else 1


def _fix_some(
    obj: Callable, grad_obj: Callable, fixed_vars: list[int], fixed_vals: np.ndarray
) -> tuple[Callable, Callable]:
    """Freeze selected variables in ``obj`` and ``grad_obj`` at fixed values."""

    order = np.argsort(fixed_vars)
    fixed_vars_sorted = [fixed_vars[i] for i in order]
    fixed_vals_sorted = fixed_vals[order]

    def fixed_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars_sorted):
            t_full.insert(i_coef, fixed_vals_sorted[i])
        arr_full = np.array(t_full)
        return obj(arr_full, other_args)

    def fixed_grad_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars_sorted):
            t_full.insert(i_coef, fixed_vals_sorted[i])
        arr_full = np.array(t_full)
        grad_full = grad_obj(arr_full, other_args)
        return np.delete(grad_full, fixed_vars_sorted)

    return fixed_obj, fixed_grad_obj


def minimize_some_fixed(
    obj: Callable,
    grad_obj: Callable,
    x_init: np.ndarray,
    args: Iterable,
    fixed_vars: list[int] | None,
    fixed_vals: np.ndarray | None,
    options: dict | None = None,
    bounds: list[tuple[float, float]] | None = None,
    time_execution: bool = False,
) -> Any:
    """
    Minimize a function with some variables fixed, using L-BFGS-B.

    Args:
        obj: the original function
        grad_obj: its gradient function
        fixed_vars: a list if the indices of variables whose values are fixed
        fixed_vals: their fixed values
        x_init: the initial values of all variables (those on fixed variables
            are not used)
        args: other parameters
        options: any options passed on to `scipy.optimize.minimize`
        bounds: the bounds on all variables (those on fixed variables are
            not used)
        time_execution: if `True`, time the execution and print the result

    Returns:
        the result of optimization, on all variables.
    """
    if time_execution:
        time_start = perf_counter()
    if fixed_vars is None:
        resopt = spopt.minimize(
            obj,
            x_init,
            method="L-BFGS-B",
            args=args,
            options=options,
            jac=grad_obj,
            bounds=bounds,
        )
    else:
        fixed_vars = cast(list, fixed_vars)
        n_fixed = check_vector(fixed_vals)
        fixed_vals = cast(np.ndarray, fixed_vals)
        if len(fixed_vars) != n_fixed:
            bs_error_abort(
                f"fixed_vars has {len(fixed_vars)} indices but fixed_vals has"
                f" {fixed_vals.size} elements."
            )
        order = np.argsort(fixed_vars)
        fixed_vars = [fixed_vars[i] for i in order]
        fixed_vals = fixed_vals[order]

        fixed_obj, fixed_grad_obj = _fix_some(obj, grad_obj, fixed_vars, fixed_vals)

        # drop fixed variables and the corresponding bounds
        n = len(x_init)
        not_fixed = np.ones(n, dtype=bool)
        not_fixed[fixed_vars] = False
        t_init = x_init[not_fixed]
        t_bounds = (
            None if bounds is None else [bounds[i] for i in range(n) if not_fixed[i]]
        )

        resopt = spopt.minimize(
            fixed_obj,
            t_init,
            method="L-BFGS-B",
            args=args,
            options=options,
            jac=fixed_grad_obj,
            bounds=t_bounds,
        )

        # now re-fill the values of the variables
        t = resopt.x
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        resopt.x = np.array(t_full)

        # and re-fill the values of the gradients
        g = grad_obj(resopt.x, args)
        resopt.jac = g

        if time_execution:
            time_end = perf_counter()
            print(
                "\nTime elapsed in minimization:"
                f" {time_end - time_start: >.3f} seconds.\n"
            )

    return resopt


# @timeit
def minimize_free(
    obj: Callable,
    grad_obj: Callable,
    x_init: np.ndarray,
    args: Iterable,
    options: dict | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> Any:
    """
    Minimize a function on all of its variables, using BFGS or L-BFGS-B.

    Args:
        obj: the original function
        grad_obj: its gradient function
        x_init: the initial values of all variables
        args: other parameters
        options: any options passed on to `scipy.optimize.minimize`
        bounds: the bounds on all variables, if any

    Returns:
        the result of optimization, on all variables.
    """
    if bounds is None:
        resopt = spopt.minimize(
            obj,
            x_init,
            method="BFGS",
            args=args,
            options=options,
            jac=grad_obj,
        )
    else:
        resopt = spopt.minimize(
            obj,
            x_init,
            method="L-BFGS-B",
            args=args,
            options=options,
            jac=grad_obj,
            bounds=bounds,
        )

    return resopt


def dfp_update(
    hess_inv: np.ndarray, gradient_diff: np.ndarray, x_diff: np.ndarray
) -> np.ndarray:
    """Runs a DFP update for the inverse Hessian.

    Args:
        hess_inv: the current inverse Hessian
        gradient_diff: the update in the gradient
        x_diff: the update in x

    Returns:
        the updated inverse Hessian.
    """
    xdt = x_diff.T
    xxp = x_diff @ xdt
    xpg = xdt @ gradient_diff
    hdg = hess_inv @ gradient_diff
    dgp_hdg = gradient_diff.T @ hdg
    eps = 1e-12
    if abs(xpg) <= eps or abs(dgp_hdg) <= eps:
        return cast(np.ndarray, hess_inv)
    hess_inv_new = hess_inv + xxp / xpg - (hdg @ hdg.T) / dgp_hdg
    return cast(np.ndarray, hess_inv_new)


def bfgs_update(
    hess_inv: np.ndarray, gradient_diff: np.ndarray, x_diff: np.ndarray
) -> np.ndarray:
    """Runs a BFGS update for the inverse Hessian.

    Args:
        hess_inv: the current inverse Hessian
        gradient_diff: the update in the gradient
        x_diff: the update in x

    Returns:
        the updated inverse Hessian.
    """
    xdt = x_diff.T
    xpg = xdt @ gradient_diff
    hdg = hess_inv @ gradient_diff
    dgp_hdg = gradient_diff.T @ hdg
    eps = 1e-12
    if abs(xpg) <= eps or abs(dgp_hdg) <= eps:
        return cast(np.ndarray, hess_inv)
    u = x_diff / xpg - hdg / dgp_hdg
    hess_inv_new = dfp_update(hess_inv, gradient_diff, x_diff) + dgp_hdg * (u @ u.T)
    return cast(np.ndarray, hess_inv_new)
