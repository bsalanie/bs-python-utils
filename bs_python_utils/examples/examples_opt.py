"""examples of optimization"""

import numpy as np

from bs_python_utils.bs_opt import (
    acc_grad_descent,
    armijo_alpha,
    barzilai_borwein_alpha,
    minimize_some_fixed,
)
from bs_python_utils.bssputils import describe_array
from bs_python_utils.bsutils import print_stars

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
    time_execution=True,
)

print(resopt)

# test the step routines
g = grad_obj(x_init, args)
alpha_a = armijo_alpha(obj, grad_obj, x_init, -g, args)
print(f"\nArmijo alpha={alpha_a}")

alpha_b, g_b = barzilai_borwein_alpha(grad_obj, x_init, args)
print(f"\nBarzilai-Borwein alpha={alpha_b}")
print("g and g_b:")
print(np.column_stack((g, g_b)))
