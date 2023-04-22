""" interface to scipy.optimize """
from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import scipy.linalg as spla
import scipy.optimize as spopt

from bs_python_utils.bsnputils import npmaxabs
from bs_python_utils.bssputils import describe_array
from bs_python_utils.bsutils import bs_error_abort, print_stars, time_this

ScalarFunctionAndGradient = Callable[
    [np.ndarray, list, Optional[bool]], Union[float, Tuple[float, np.ndarray]]
]
"""Type of f(v, args, gr) that returns a scalar value and also a gradient if gr is True"""


@dataclass
class OptimizeParams:
    """
    used for optimization;
    combines values, bounds and initial values for a parameter vector
    """

    params_values: Union[np.ndarray, None]
    params_bounds: Union[List[Tuple], None]
    params_init: Union[np.ndarray, None]


def print_optimization_results(
    resus: object, title: Optional[str] = "Minimizing"
) -> None:
    """
    print results from unconstrained optimization

    Args:
        resus: results from optimization
        str title: a title

    Returns:
        just prints
    """
    print_stars(title)
    print(resus.message)
    if resus.success:
        print(f"Successful! in {resus.nit} iterations")
        print(f" evaluated {resus.nfev} functions functions and {resus.njev} gradients")
        print("\nMinimizer and grad_f:")
        print(np.column_stack((resus.x, resus.jac)))
        print(f"Minimized value is {resus.fun}")
    else:
        print_stars("Minimization failed!")
    return


def print_constrained_optimization_results(
    resus: object,
    title: Optional[str] = "Minimizing",
    print_constr: Optional[bool] = False,
    print_multipliers: Optional[bool] = False,
) -> None:
    """
    print results from constrained optimization

       Args:
        resus: results from optimization
        str title: a title
        print_constr: if `True`, print the values of the constraints
        print_multipliers: if `True`, print the values of the multipliers

    Returns:
        just prints
    """
    print_stars(title)
    print(resus.message)
    if resus.success:
        print(f"Successful! in {resus.nit} iterations")
        print(f" evaluated {resus.nfev} functions and {resus.njev} gradients")
        print(f"Minimized value is {resus.fun}")
        print(f"The Lagrangian norm is {resus.optimality}")
        print(f"The largest constraint violation is {resus.constr_violation}")
        if print_multipliers:
            print(f"The multipliers are {resus.v}")
        if print_constr:
            print(f"The values of the constraints are {resus.constr}")
    else:
        print_stars("Constrained minimization failed!")
    return


def armijo_alpha(
    f: Callable,
    x: np.ndarray,
    d: np.ndarray,
    args: list,
    alpha_init: Optional[float] = 1.0,
    beta: Optional[float] = 0.5,
    max_iter: Optional[int] = 100,
    tol: Optional[float] = 0.0,
) -> float:
    """Given a function `f` we are minimizing, computes the step size `alpha`
    to take in the direction `d` using the Armijo rule

    Args:
        f: the function
        x: the current point
        d: the direction we are taking
        args: other arguments passed to `f`
        alpha_init: the initial step size
        beta: the step size reduction factor
        max_iter: the maximum number of iterations
        tol: a tolerance

    Returns:
        the step size alpha
    """
    f0 = f(x, args)
    alpha = alpha_init
    for _ in range(max_iter):
        x1 = x + alpha * d
        f1 = f(x1, args)
        if f1 < f0 + tol:
            return alpha
        alpha *= beta
    else:
        bs_error_abort("Too many iterations")
    return alpha


def barzilai_borwein_alpha(
    grad_f: Callable, x: np.ndarray, args: list
) -> Tuple[float, np.ndarray]:
    """Given a function `f` we are minimizing, computes the step size `alpha`
    to take in the opposite direction of the gradient using the Barzilai-Borwein rule

    Args:
        grad_f: the gradient of the function
        x: the current point
        args: other arguments passed to `f`

    Returns:
        the step size `alpha` and the gradient `g` at the point `x`
    """
    g = grad_f(x, args)
    alpha = 1.0 / spla.norm(g)
    x_hat = x - alpha * g
    g_hat = grad_f(x_hat, args)
    norm_dg = spla.norm(g - g_hat)
    norm_dg2 = norm_dg * norm_dg
    alpha = np.abs(np.dot(x - x_hat, g - g_hat)) / norm_dg2
    return alpha, g


def check_gradient_scalar_function(
    fg: ScalarFunctionAndGradient,
    p: np.ndarray,
    args: list,
    mode: Optional[str] = "central",
    EPS: Optional[float] = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Checks the gradient of a scalar function

    Args:
        fg: should return the scalar value, and the gradient if its `gr` argument is `True`
        p: where we are checking the gradient
        args: other arguments passed to `fg`
        mode: "central" or "forward" derivatives
        EPS: the step for forward or central derivatives

    Returns:
        the analytic and numeric gradients
    """
    f0, f_grad = fg(p, args, gr=True)

    print_stars("checking the gradient: analytic, numeric")

    g = np.zeros_like(p)
    if mode == "central":
        for i, x in enumerate(p):
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = fg(p1, args, gr=False)
            p1[i] -= 2.0 * EPS
            f_minus = fg(p1, args, gr=False)
            g[i] = (f_plus - f_minus) / (2.0 * EPS)
            print(f"{i}: {f_grad[i]}, {g[i]}")
    elif mode == "forward":
        for i, x in enumerate(p):
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = fg(p1, args, gr=False)
            g[i] = (f_plus - f0) / EPS
            print(f"{i}: {f_grad[i]}, {g[i]}")
    else:
        bs_error_abort("mode must be 'central' or 'forward'")

    return f_grad, g


@time_this
def acc_grad_descent(
    grad_f: Callable,
    x_init: np.ndarray,
    prox_h: Optional[Callable] = None,
    other_params: Optional[Any] = None,
    print_result: Optional[bool] = False,
    verbose: Optional[bool] = False,
    tol: Optional[float] = 1e-9,
    alpha: Optional[float] = 1.01,
    beta: Optional[float] = 0.5,
    maxiter: Optional[int] = 10000,
) -> Tuple[np.ndarray, bool]:
    """
    minimizes `(f+h)` by Accelerated Gradient Descent
     where `f` is smooth and convex  and `h` is convex.

    By default `h` is zero.

    Args:
        grad_f: grad_f of `f`; should return an `(n)` array from an `(n)` array \
        and the `other_ params` object
        x_init: initial guess, shape `(n)`
        prox_h: proximal projector of `h`, if any; should return an `(n)` array from \
        an `(n)` array, a float, and an `(n)` array
        other_params: an object with additional parameters
        verbose: if `True`, print diagnosis
        tol: convergence criterion on absolute grad_f
        alpha: ceiling on step multiplier
        beta: floor on step multiplier
        maxiter: max number of iterations

    Returns: 
        the candidate solution, and 1 if converged/0 if not
    """

    # no proximal projection if no h
    local_prox_h = prox_h if prox_h else lambda x, t, p: x

    x = x_init.copy()
    y = x_init.copy()

    #  for stepsize we use Barzilai-Borwein
    t, g = barzilai_borwein_alpha(grad_f, y, other_params)

    grad_err_init = npmaxabs(g)

    if verbose:
        print(f"agd: grad_err_init={grad_err_init}")

    iter = 0
    theta = 1.0

    while iter < maxiter:
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
        ndy = spla.norm(y - yi)
        t_hat = 0.5 * ndy * ndy / abs(np.dot(y - yi, gi - g))
        t = min(alpha * t, max(beta * t, t_hat))

        iter += 1

        if verbose:
            print(f" AGD with grad_err = {grad_err} after {iter} iterations")

    x_conv = y

    ret_code = 0 if grad_err < tol else 1

    if verbose or print_result:
        if ret_code == 0:
            print_stars(
                f" AGD converged with grad_err = {grad_err} after {iter} iterations"
            )
        else:
            print_stars(
                f" Problem in AGD: grad_err = {grad_err} after {iter} iterations"
            )

    return (x_conv, ret_code)


def _fix_some(
    obj: Callable, grad_obj: Callable, fixed_vars: List[int], fixed_vals: np.ndarray
) -> Tuple[Callable, Callable]:
    """
    Takes in a function and its gradient, fixes the variables
    whose indices are `fixed_vars` to the values in `fixed_vals`,
    and returns the modified function and its gradient

    Args:
        obj: the original function
        grad_obj: its gradient function
        fixed_vars: a list if the indices of variables whose values are fixed
        fixed_vals: their fixed values

    Returns:
        the modified function and its modified gradient function
    """

    def fixed_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        arr_full = np.array(t_full)
        return obj(arr_full, other_args)

    def fixed_grad_obj(t, other_args):
        t_full = list(t)
        for i, i_coef in enumerate(fixed_vars):
            t_full.insert(i_coef, fixed_vals[i])
        arr_full = np.array(t_full)
        grad_full = grad_obj(arr_full, other_args)
        return np.delete(grad_full, fixed_vars)

    return fixed_obj, fixed_grad_obj


def minimize_some_fixed(
    obj: Callable,
    grad_obj: Callable,
    x_init: np.ndarray,
    args: List,
    fixed_vars: Union[List[int], None],
    fixed_vals: Union[np.ndarray, None],
    options: Optional[Dict] = None,
    bounds: Optional[List[Tuple]] = None,
):
    """
    minimize a function with some variables fixed, using L-BFGS-B

    Args:
        obj: the original function
        grad_obj: its gradient function
        fixed_vars: a list if the indices of variables whose values are fixed
        fixed_vals: their fixed values
        x_init: the initial values of all variables (those on fixed variables are not used)
        args: a list of other parameters
        options: any options passed on to scipy.optimize.minimize
        bounds: the bounds on all variables (those on fixed variables are not used)

    Returns:
        the result of optimization, on all variables
    """
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
        if len(fixed_vars) != fixed_vals.size:
            bs_error_abort(
                f"fixed_vars has {len(fixed_vars)} indices but fixed_vals has"
                f" {fixed_vals.size} elements."
            )
        fixed_obj, fixed_grad_obj = _fix_some(obj, grad_obj, fixed_vars, fixed_vals)

        # drop fixed variables and the corresponding bounds
        n = len(x_init)
        not_fixed = np.ones(n, dtype=bool)
        not_fixed[fixed_vars] = False
        t_init = x_init[not_fixed]
        t_bounds = [bounds[i] for i in range(n) if not_fixed[i]]

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
        resopt.x = t_full

        # and re-fill the values of the gradients
        g = grad_obj(np.array(t_full), args)
        resopt.jac = g

    return resopt


def dfp_update(
    hess_inv: np.ndarray, gradient_diff: np.ndarray, x_diff: np.ndarray
) -> np.ndarray:
    """runs a DFP update for the inverse Hessian

    Args:
        hess_inv: the current inverse Hessian
        gradient_diff: the update in the gradient
        x_diff: the update in x

    Returns:
        the updated inverse Hessian
    """
    xdt = x_diff.T
    xxp = x_diff @ xdt
    xpg = xdt @ gradient_diff
    hdg = hess_inv @ gradient_diff
    dgp_hdg = gradient_diff.T @ hdg
    hess_inv_new = hess_inv + xxp / xpg - (hdg @ hdg.T) / dgp_hdg
    return hess_inv_new


def bfgs_update(
    hess_inv: np.ndarray, gradient_diff: np.ndarray, x_diff: np.ndarray
) -> np.ndarray:
    """runs a DFP update for the inverse Hessian

    Args:
        hess_inv: the current inverse Hessian
        gradient_diff: the update in the gradient
        x_diff: the update in x

    Returns:
        the updated inverse Hessian
    """
    xdt = x_diff.T
    xpg = xdt @ gradient_diff
    hdg = hess_inv @ gradient_diff
    dgp_hdg = gradient_diff.T @ hdg
    u = x_diff / xpg - hdg / dgp_hdg
    hess_inv_new = dfp_update(hess_inv, gradient_diff, x_diff) + dgp_hdg * (u @ u.T)
    return hess_inv_new


# def bs_minimize_unconstrained(
#     f: Callable,
#     x0: np.ndarray,
#     args: List,
#     hessian_update: Optional[str] = "BFGS",
#     max_iters: Optional[int] = 10000,
#     tol: Optional[float] = 1e-6,
#     initial_hessian_scale: Optional[float] = 1.0,
#     verbose: Optional[bool] = False,
# ) -> Tuple[np.ndarray, float, int]:
#     """ minimizes the function `f` with quasi-Newton

#     Args:
#         f: a function of (x, args) that computes the objective function, and the gradient if grad=True
#         x0: the initial guess
#         args: other parameters for `f`
#         hessian_update: the type of Hessian update to use
#         max_iters: the maximum number of iterations
#         tol: the tolerance for the stopping criterion
#         initial_hessian_scale: an initial guess for the scale of the Hessian
#         verbose: whether to print the iteration number and the norm of the gradient

#     Returns:
#         the solution, the objective value, and the number of iterations
#     """
#     nx = x0.size
#     inverse_hessian_update = dfp_update if hessian_update == "DFP" else bfgs_update

#     hess_inv = np.eye(nx) / initial_hessian_scale
#     x = x0
#     for k in range(max_iters):
#         fv, fp = f(x, args, grad=True)
#         dx = -hess_inv @ fp
#         alpha = armijo_alpha(f, x, dx, args)
#         dx *= alpha
#         x += dx
#         fv_new, fp_new = f(x, args, grad=True)
#         error = max(abs(fv_new - fv), spla.norm(fp_new))
#         print(f"{error=}")
#         if error < tol:
#             break
#         dgrad = fp_new - fp
#         hess_inv = inverse_hessian_update(hess_inv, dgrad, dx)

#         if verbose:
#             print(f"Iteration {k}: f={f(x, args)}")
#             print(f"   at x={x}")
#             print(f" with {hess_inv=}")

#     return x, f(x, args), k


if __name__ == "__main__":
    print_stars("Testing acc_grad_descent")

    def grad_f(x, p):
        xp = x - p[0]
        return 4.0 * xp * xp * xp

    x_init = np.random.normal(size=10000)

    p = 1.0
    x_conv, ret_code = acc_grad_descent(
        grad_f, x_init, other_params=np.array([p]), tol=1e-12, verbose=False
    )

    describe_array(x_conv - p, "x-p should be close to zero")

    def obj(x, args):
        res = x - args
        return np.sum(res * res)

    def grad_obj(x, args):
        res = x - args
        return 2.0 * res

    n = 5
    x_init = np.full(n, 0.5)
    args = np.arange(n)
    bounds = [(-10.0, 10.0) for _ in range(n)]

    fixed_vars = [1, 3]
    fixed_vals = -np.ones(2)

    resopt = minimize_some_fixed(
        obj,
        grad_obj,
        x_init,
        args,
        fixed_vars=fixed_vars,
        fixed_vals=fixed_vals,
        bounds=bounds,
    )

    print(resopt)

    # test the step routines
    g = grad_obj(x_init, args)
    alpha_a = armijo_alpha(obj, x_init, -g, args)
    print(f"\nArmijo alpha={alpha_a}")

    alpha_b, g_b = barzilai_borwein_alpha(grad_obj, x_init, args)
    print(f"\nBarzilai-Borwein alpha={alpha_a}")
    print("g and g_b:")
    print(np.column_stack((g, g_b)))

    # test bs_minimize_unconstrained

    # def fg(x: np.ndarray, args: List, grad: bool=False) -> float | Tuple[float, np.ndarray]:
    #     fval = np.sum(x * x)
    #     if grad:
    #         fp = 2.0 * x
    #         return fval, fp
    #     else:
    #         return fval

    # def obj_and_grad(x, args, grad=False):
    #     if grad:
    #         return obj(x, args), grad_obj(x, args)
    #     else:
    #         return obj(x, args)

    # n = 3
    # x_init = np.full(n, 0.5)
    # args = None
    # x, fval, n_iters = bs_minimize_unconstrained(fg, x_init, args, verbose=True)
    # print(f"\n{x=}, should be zeroes; with {fval=} in {n_iters=} iterations.\n\n")

    # args = np.array([-0.2, 0.3, 0.1])
    # x, fval, n_iters = bs_minimize_unconstrained(obj_and_grad, x_init, args, verbose=True)
    # print(f"\n{x=}, should be {args}; with {fval=} in {n_iters=} iterations.\n\n")