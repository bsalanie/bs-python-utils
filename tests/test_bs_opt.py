import numpy as np

from bs_python_utils import bs_opt


def quadratic_objective(x, args):
    target = args
    residual = x - target
    return 0.5 * np.dot(residual, residual)


def quadratic_gradient(x, args):
    target = args
    return x - target


def test_armijo_alpha_decreases_quadratic():
    x = np.array([1.0, -1.0, 2.0])
    args = np.zeros_like(x)
    direction = -quadratic_gradient(x, args)

    alpha = bs_opt.armijo_alpha(
        quadratic_objective, quadratic_gradient, x, direction, args
    )

    assert alpha > 0.0
    new_x = x + alpha * direction
    assert quadratic_objective(new_x, args) <= quadratic_objective(x, args)


def test_barzilai_borwein_alpha_handles_zero_gradient():
    x = np.zeros(3)
    args = np.zeros_like(x)

    alpha, grad = bs_opt.barzilai_borwein_alpha(quadratic_gradient, x, args)

    assert alpha == 1.0
    np.testing.assert_allclose(grad, np.zeros_like(x))


def test_minimize_some_fixed_reinstates_fixed_variables():
    n = 4
    args = np.arange(n, dtype=float)
    x_init = np.zeros(n)
    fixed_vars = [2, 0]
    fixed_vals = np.array([10.0, -5.0])

    result = bs_opt.minimize_some_fixed(
        quadratic_objective,
        quadratic_gradient,
        x_init,
        args,
        fixed_vars=fixed_vars,
        fixed_vals=fixed_vals,
        bounds=[(-20.0, 20.0)] * n,
    )

    expected = args.copy()
    expected[fixed_vars[0]] = fixed_vals[0]
    expected[fixed_vars[1]] = fixed_vals[1]
    np.testing.assert_allclose(result.x, expected)


def test_dfp_update_returns_hessian_when_singular_direction():
    hess_inv = np.eye(2)
    gradient_diff = np.zeros((2, 1))
    x_diff = np.array([[0.0], [0.0]])

    updated = bs_opt.dfp_update(hess_inv, gradient_diff, x_diff)

    np.testing.assert_allclose(updated, hess_inv)


def test_bfgs_update_returns_hessian_when_singular_direction():
    hess_inv = np.eye(2)
    gradient_diff = np.zeros((2, 1))
    x_diff = np.array([[0.0], [0.0]])

    updated = bs_opt.bfgs_update(hess_inv, gradient_diff, x_diff)

    np.testing.assert_allclose(updated, hess_inv)
