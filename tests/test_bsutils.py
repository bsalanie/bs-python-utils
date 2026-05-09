from math import exp, isclose, log, sqrt

import numpy as np

from bs_python_utils.core.bsutils import (
    bs_projection_point,
    bs_switch,
    bscomb,
    bsexp,
    bslog,
    bsxlogx,
    final_s,
    find_first,
)


def test_bs_switch():
    calc_dict = {"plus": lambda x, y: x + y, "minus": lambda x, y: x - y}
    plus = bs_switch("plus", calc_dict, default="unintended function")
    minus = bs_switch("min", calc_dict, strict=False, default="unintended function")
    assert plus(6, 4) == 10
    assert minus(6, 4) == 2
    assert bs_switch("plu", calc_dict) == "no match"


def test_find_first():
    onetwo = find_first((1, 2, 3), condition=lambda x: x % 2 == 0)
    assert onetwo == (1, 2)
    zerothree = find_first(range(3, 100))
    assert zerothree == (0, 3)


def test_bscomb():
    assert bscomb(4, 2) == 6
    assert bscomb(5, 0) == 1
    assert bscomb(3, 3) == 1


def test_bs_projection_point():
    x, y = 2.0, 2.0
    a, b, c = 1.0, -1.0, 1.0
    x_proj, y_proj, dist = bs_projection_point(x, y, a, b, c)
    xyd = np.array([x_proj, y_proj, dist])
    assert np.allclose(xyd, np.array([1.5, 2.5, 1.0 / sqrt(2.0)]))


def test_bslog_above():
    lx = -10.0
    x = exp(lx)
    lx2 = bslog(x)
    assert isclose(lx2, lx)
    lx3, dx3 = bslog(x, deriv=1)
    assert isclose(lx3, lx)
    assert isclose(dx3, 1.0 / x)
    lx4, dx4, d2x4 = bslog(x, deriv=2)
    assert isclose(lx4, lx)
    assert isclose(dx4, 1.0 / x)
    assert isclose(d2x4, -1.0 / x / x)


def test_bslog_below():
    lx = -100.0
    x = exp(lx)
    eps = 1e-30
    lxth = log(eps) + (x - eps) / eps - (x - eps) * (x - eps) / eps / eps / 2.0
    dlxth = 1.0 / eps - (x - eps) / eps / eps
    d2lxth = -1.0 / eps / eps
    lx2 = bslog(x)
    assert isclose(lx2, lxth)
    lx3, dx3 = bslog(x, deriv=1)
    assert isclose(lx3, lxth)
    assert isclose(dx3, dlxth)
    lx4, dx4, d2x4 = bslog(x, deriv=2)
    assert isclose(lx4, lxth)
    assert isclose(dx4, dlxth)
    assert isclose(d2x4, d2lxth)


def test_bsxlogx_above():
    lx = -10.0
    x = exp(lx)
    xlogx = x * lx
    dxlogx = 1.0 + lx
    d2xlogx = 1.0 / x
    xlx2 = bsxlogx(x)
    assert isclose(xlx2, xlogx)
    xlx3, dxlx3 = bsxlogx(x, deriv=1)
    assert isclose(xlx3, xlogx)
    assert isclose(dxlx3, dxlogx)
    xlx4, dxlx4, d2xlx4 = bsxlogx(x, deriv=2)
    assert isclose(xlx4, xlogx)
    assert isclose(dxlx4, dxlogx)
    assert isclose(d2xlx4, d2xlogx)


def test_bsxlogx_below():
    lx = -100.0
    x = exp(lx)
    eps = 1e-30
    xlogx = (
        eps * log(eps)
        + (1.0 + log(eps)) * (x - eps)
        + (x - eps) * (x - eps) / eps / 2.0
    )
    dxlogx = 1.0 + log(eps) + (x - eps) / eps
    d2xlogx = 1.0 / eps
    xlx2 = bsxlogx(x)
    assert isclose(xlx2, xlogx)
    xlx3, dxlx3 = bsxlogx(x, deriv=1)
    assert isclose(xlx3, xlogx)
    assert isclose(dxlx3, dxlogx)
    xlx4, dxlx4, d2xlx4 = bsxlogx(x, deriv=2)
    assert isclose(xlx4, xlogx)
    assert isclose(dxlx4, dxlogx)
    assert isclose(d2xlx4, d2xlogx)


def test_bsexp_usual():
    x = 10.0
    ex = exp(x)
    ex2 = bsexp(x)
    assert isclose(ex2, ex)
    ex3, dex3 = bsexp(x, deriv=1)
    assert isclose(ex3, ex)
    assert isclose(dex3, ex)
    ex4, dex4, d2ex4 = bsexp(x, deriv=2)
    assert isclose(ex4, ex)
    assert isclose(dex4, ex)
    assert isclose(d2ex4, ex)


def test_bsexp_below():
    x = -120.0
    lowx = -50.0
    ex = exp(lowx) * (1.0 + (x - lowx) + (x - lowx) * (x - lowx) / 2.0)
    dex = exp(lowx) * (1.0 + (x - lowx))
    d2ex = exp(lowx)
    ex2 = bsexp(x)
    assert isclose(ex2, ex)
    ex3, dex3 = bsexp(x, deriv=1)
    assert isclose(ex3, ex)
    assert isclose(dex3, dex)
    ex4, dex4, d2ex4 = bsexp(x, deriv=2)
    assert isclose(ex4, ex)
    assert isclose(dex4, dex)
    assert isclose(d2ex4, d2ex)


def test_bsexp_above():
    x = 60.0
    bigx = 50.0
    ex = exp(bigx) * (1.0 + (x - bigx) + (x - bigx) * (x - bigx) / 2.0)
    dex = exp(bigx) * (1.0 + (x - bigx))
    d2ex = exp(bigx)
    ex2 = bsexp(x)
    assert isclose(ex2, ex)
    ex3, dex3 = bsexp(x, deriv=1)
    assert isclose(ex3, ex)
    assert isclose(dex3, dex)
    ex4, dex4, d2ex4 = bsexp(x, deriv=2)
    assert isclose(ex4, ex)
    assert isclose(dex4, dex)
    assert isclose(d2ex4, d2ex)


def test_final_s():
    assert final_s(0, "simulation") == "0 simulation"
    assert final_s(1, "simulation") == "1 simulation"
    assert final_s(13, "simulation") == "13 simulations"
