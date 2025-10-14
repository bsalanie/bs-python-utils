from math import exp, isclose, sqrt

import numpy as np
import pytest

from bs_python_utils.bsnputils import FloatOrArray, bsgrid
from bs_python_utils.bsutils import bs_error_abort
from bs_python_utils.chebyshev import (
    Interval,
    Rectangle,
    cheb_eval_fun_at_nodes_1d,
    cheb_eval_fun_at_nodes_2d,
    cheb_find_root,
    cheb_get_coefficients_1d,
    cheb_get_coefficients_2d,
    cheb_get_nodes_1d,
    cheb_get_nodes_2d,
    cheb_integrate_from_coeffs_1d,
    cheb_integrate_from_coeffs_2d,
    cheb_integrate_from_nodes_1d,
    cheb_integrate_from_nodes_2d,
    cheb_integrate_from_nodes_4d,
    cheb_interp_1d,
    cheb_interp_1d_from_nodes,
    cheb_interp_2d,
    cheb_interp_2d_from_nodes,
)


def fun1d(x: FloatOrArray) -> FloatOrArray:
    return np.exp(-x)


def fun1d2(x: FloatOrArray) -> FloatOrArray:
    # return (x-0.3)*(x-0.5)*(x-0.7)
    return 8.0 * x * x - 8.0 * x + 1.0


def fun2d(xy: np.ndarray) -> np.ndarray | None:
    if xy.ndim == 1:
        return xy[0] * np.exp(-xy[1])
    elif xy.ndim == 2:
        return xy[:, 0] * np.exp(-xy[:, 1])
    else:
        bs_error_abort(f"xy cannot have {xy.ndim} dimensions.")
        return None


def fun2d_2d(xy: np.ndarray, zt: np.ndarray) -> np.ndarray:
    return np.outer(xy[:, 0] * np.exp(-xy[:, 1]), zt[:, 0] + 3.0 * zt[:, 1])


def test_cheb_get_nodes_1d():
    deg, x0, x1 = 8, 0.0, 1.0
    nodes, weights = cheb_get_nodes_1d(Interval(x0=x0, x1=x1), deg)
    nodes1m1 = np.cos(np.arange(1, 2.0 * deg, 2) * np.pi / (2.0 * deg))
    nodes_th = (nodes1m1 * (x1 - x0) + (x0 + x1)) / 2.0
    assert np.allclose(nodes, nodes_th)
    weights_th = (np.pi / deg) * np.sqrt(1.0 - nodes1m1 * nodes1m1) * (x1 - x0) / 2.0
    assert np.allclose(weights, weights_th)


def test_cheb_eval_fun_at_nodes_1d():
    deg, x0, x1 = 8, 0.0, 1.0
    interval = Interval(x0=x0, x1=x1)
    nodes, _ = cheb_get_nodes_1d(interval, deg)
    vals = cheb_eval_fun_at_nodes_1d(fun1d, nodes=nodes)
    vals_th = np.exp(-nodes)
    assert np.allclose(vals, vals_th)
    vals2 = cheb_eval_fun_at_nodes_1d(fun1d, interval=interval, degree=deg)
    assert np.allclose(vals2, vals_th)


def test_cheb_interp_1d():
    deg, x0, x1 = 8, -1.0, 1.0
    interval = Interval(x0=x0, x1=x1)
    x_vals = -0.9 + np.arange(4) / 2.0
    c = cheb_get_coefficients_1d(fun1d, interval, deg)
    y_vals, c2 = cheb_interp_1d(x_vals, interval, c=c)
    assert np.allclose(c2, c)
    y_vals_th = fun1d(x_vals)
    assert np.allclose(y_vals, y_vals_th)
    y_vals2, c3 = cheb_interp_1d(x_vals, interval, fun=fun1d, degree=deg)
    assert np.allclose(c3, c)
    assert np.allclose(y_vals2, y_vals_th)


def test_interp_1d_from_nodes():
    deg, x0, x1 = 8, 0.0, 1.0
    interval = Interval(x0=x0, x1=x1)
    nodes, _ = cheb_get_nodes_1d(interval, deg)
    m_vals = set(range(deg))
    f_vals = np.zeros(deg)
    for m in m_vals:
        f_vals[m] = fun1d(nodes[m])
    x = 0.6
    val = cheb_interp_1d_from_nodes(f_vals, x)
    val_th = fun1d(x)
    print(f"{val=} and {val_th=}")
    assert isclose(val, val_th)


def test_cheb_integrate_from_coeffs_1d():
    deg, x0, x1 = 8, 0.0, 1.0
    interval = Interval(x0=x0, x1=x1)
    c = cheb_get_coefficients_1d(fun1d, interval, deg)
    integ_val = cheb_integrate_from_coeffs_1d(c, interval)
    integ_val_th = exp(-x0) - exp(-x1)
    assert isclose(integ_val, integ_val_th)


def test_cheb_integrate_from_nodes_1d():
    deg, x0, x1 = 64, 0.0, 1.0
    interval = Interval(x0=x0, x1=x1)
    nodes, weights = cheb_get_nodes_1d(interval, deg)
    vals_at_nodes = cheb_eval_fun_at_nodes_1d(fun1d, nodes=nodes)
    integ_val = cheb_integrate_from_nodes_1d(vals_at_nodes, weights)
    integ_val_th = exp(-x0) - exp(-x1)
    assert isclose(integ_val, integ_val_th, abs_tol=1e-4)


def test_cheb_get_nodes_2d():
    deg, x0, x1, y0, y1 = 8, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    nodes_x, weights_x = cheb_get_nodes_1d(x_interval, deg)
    nodes_y, weights_y = cheb_get_nodes_1d(y_interval, deg)
    nodes_xy, weights_xy = cheb_get_nodes_2d(rectangle, deg)
    assert np.allclose(nodes_xy[3 * deg + 5, :], np.array([nodes_x[3], nodes_y[5]]))
    assert np.allclose(weights_xy[2 * deg + 6], np.array([weights_x[2] * weights_y[6]]))


def test_cheb_eval_fun_at_nodes_2d():
    deg, x0, x1, y0, y1 = 8, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    nodes, _ = cheb_get_nodes_2d(rectangle, deg)
    vals = cheb_eval_fun_at_nodes_2d(fun2d, nodes=nodes)
    vals_th = nodes[:, 0] * np.exp(-nodes[:, 1])
    assert np.allclose(vals, vals_th)
    vals2 = cheb_eval_fun_at_nodes_2d(fun2d, rectangle=rectangle, degree=deg)
    assert np.allclose(vals2, vals_th)


def test_cheb_interp_2d():
    deg, x0, x1, y0, y1 = 16, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    vals = 0.1 + np.arange(2) / 3.0
    xy_vals = bsgrid(vals, vals)
    c = cheb_get_coefficients_2d(rectangle, deg, fun=fun2d)
    f_vals, c2 = cheb_interp_2d(xy_vals, rectangle=rectangle, c=c)
    assert np.allclose(c2, c)
    f_vals_th = fun2d(xy_vals)
    assert np.allclose(f_vals, f_vals_th, atol=1e-3)
    f_vals2, c3 = cheb_interp_2d(xy_vals, rectangle, fun=fun2d, degree=deg)
    assert np.allclose(c3, c)
    assert np.allclose(f_vals2, f_vals_th, atol=1e-3)

    # scalar evaluation returns a float
    point = np.array([0.25, 0.75])
    val_scalar, _ = cheb_interp_2d(point, rectangle, fun=fun2d, degree=deg)
    assert isinstance(val_scalar, float)
    assert isclose(val_scalar, fun2d(point))

    # integer dtype inputs should still be converted safely
    xy_int = np.array([[0, 0], [1, 1]], dtype=int)
    vals_from_int, _ = cheb_interp_2d(xy_int, rectangle, fun=fun2d, degree=deg)
    expected = fun2d(xy_int.astype(float))
    assert np.allclose(vals_from_int, expected)


def test_interp_2d_from_nodes():
    deg, x0, x1, y0, y1 = 16, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    nodes, _ = cheb_get_nodes_2d(rectangle, deg)
    m_vals = set(range(deg * deg))
    f_vals = np.zeros(deg * deg)
    for m in m_vals:
        f_vals[m] = fun2d(nodes[m])
    x = np.array([0.3, 0.6])
    val = cheb_interp_2d_from_nodes(f_vals, x, rectangle)
    val_th = fun2d(x)
    print(f"{val=} and {val_th=}")
    assert isclose(val, val_th)


def test_cheb_integrate_from_coeffs_2d():
    deg, x0, x1, y0, y1 = 8, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    c = cheb_get_coefficients_2d(rectangle, deg, fun=fun2d)
    val_integral = cheb_integrate_from_coeffs_2d(c, rectangle)
    val_integral_th = 1.0 / 2.0 * (1.0 - exp(-1.0))
    assert isclose(val_integral, val_integral_th)


def test_cheb_integrate_from_nodes_2d():
    deg, x0, x1, y0, y1 = 32, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    _, weights = cheb_get_nodes_2d(rectangle, deg)
    vals_at_nodes = cheb_eval_fun_at_nodes_2d(fun2d, rectangle=rectangle, degree=deg)
    val_integral = cheb_integrate_from_nodes_2d(vals_at_nodes, weights)
    val_integral_th = 1.0 / 2.0 * (1.0 - exp(-1.0))
    assert isclose(val_integral, val_integral_th, abs_tol=1e-3)


def test_cheb_get_coefficients_2d_invalid_shape():
    deg = 4
    interval = Interval(0.0, 1.0)
    rectangle = Rectangle(interval, interval)
    bad_vals = np.arange(deg * deg + 1, dtype=float)
    with pytest.raises(SystemExit):
        cheb_get_coefficients_2d(rectangle, deg, vals_at_nodes=bad_vals)


def test_cheb_integrate_from_nodes_4d():
    deg, x0, x1, y0, y1 = 128, 0.0, 1.0, 0.0, 1.0
    x_interval = Interval(x0=x0, x1=x1)
    y_interval = Interval(x0=y0, x1=y1)
    rectangle = Rectangle(x_interval=x_interval, y_interval=y_interval)
    nodes2d, weights2d = cheb_get_nodes_2d(rectangle, deg)
    vals_at_nodes4d = fun2d_2d(nodes2d, nodes2d)
    val_integral = cheb_integrate_from_nodes_4d(vals_at_nodes4d, weights2d)
    val_integral_th = 1.0 - exp(-1.0)
    assert isclose(val_integral, val_integral_th, abs_tol=1e-3)


def test_cheb_find_root():
    deg, x0, x1 = 2, 0.7, 0.9
    x_interval = Interval(x0=x0, x1=x1)
    roots = cheb_find_root(fun1d2, deg)
    root1 = (1.0 - 1.0 / sqrt(2.0)) / 2.0
    root2 = (1.0 + 1.0 / sqrt(2.0)) / 2.0
    roots_th = np.array([root1, root2])
    assert np.allclose(roots, roots_th)
    roots_bis, roots7 = cheb_find_root(fun1d2, deg, interval=x_interval)
    assert np.allclose(roots_bis, roots_th)
    roots7_th = root2
    assert isclose(roots7, roots7_th)
    x_interval2 = Interval(x0=0.1, x1=0.9)
    roots_ter, roots2 = cheb_find_root(fun1d2, deg, interval=x_interval2)
    assert np.allclose(roots_ter, roots_th)
    assert np.allclose(roots2, roots_th)
    x_interval3 = Interval(x0=0.0, x1=0.1)
    roots_qua, roots3 = cheb_find_root(fun1d2, deg, interval=x_interval3)
    assert np.allclose(roots_qua, roots_th)
    assert roots3 is None
