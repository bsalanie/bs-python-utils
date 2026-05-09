from math import exp, isclose, log, pi, sqrt

import numpy as np

from bs_python_utils.core.bsutils import bsexp, bslog
from bs_python_utils.numerical.bsnputils import (
    bs_sqrt_pdmatrix,
    bsgrid,
    ecdf,
    find_row_single_nonzero,
    gauher,
    gaussian_expectation,
    inv_ecdf,
    make_lexico_grid,
    make_lower_tri,
    npexp,
    nplog,
    nppad2_end_zeros,
    nppad_beg_zeros,
    nppad_end_zeros,
    nppow,
    nprepeat_col,
    nprepeat_row,
    print_quantiles,
    rice_stderr,
)


def test_print_quantiles_vec():
    v = np.arange(101.0)
    q = np.arange(10.0) / 10.0
    qvals = print_quantiles(v, q)
    qvals_expected = np.arange(start=0.0, stop=100.0, step=10.0)
    assert np.allclose(qvals, qvals_expected)


def test_print_quantiles_list():
    v1 = np.arange(101.0)
    v2 = np.arange(101.0, 202.0)
    lst = [v1, v2]
    q = np.arange(10.0) / 10.0
    qvals = print_quantiles(lst, q)
    qvals_expected = np.column_stack(
        (
            np.arange(start=0.0, stop=100.0, step=10.0),
            np.arange(start=101.0, stop=200.0, step=10.0),
        )
    )
    assert np.allclose(qvals, qvals_expected)


def test_bs_sqrt_pdmatrix():
    a = np.array([[3, 2], [2, 3]])
    b = bs_sqrt_pdmatrix(a)
    assert np.allclose(b, b.T)
    assert np.allclose(b @ b, a)


def test_ecdf():
    x = np.array([1.2, -3.6, 6.7, 1.1, -9.0])
    assert np.allclose(ecdf(x), np.array([0.8, 0.4, 1.0, 0.6, 0.2]))


def test_inv_ecdf():
    x = np.array([1.2, -3.6, 6.7, 1.1, -9.0])
    q1 = 0.4
    assert np.allclose(inv_ecdf(x, q1), -3.6)
    q2 = np.array([0.1, 0.6, 1.0])
    assert np.allclose(inv_ecdf(x, q2), np.array([-11.7, 1.1, 6.7]))


def test_nprepeat_row():
    v = np.arange(3)
    vm = nprepeat_row(v, 2)
    vmth = np.array([[0, 1, 2], [0, 1, 2]])
    assert np.allclose(vm, vmth)


def test_nprepeat_col():
    v = np.arange(3)
    vn = nprepeat_col(v, 4)
    vnth = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    assert np.allclose(vn, vnth)


def test_nppad_beg_zeros():
    a = np.arange(1, 5)
    n = 8
    an = nppad_beg_zeros(a, n)
    anth = np.array([0, 0, 0, 0, 1, 2, 3, 4])
    assert np.allclose(an, anth)


def test_nppad_end_zeros():
    a = np.arange(1, 5)
    n = 8
    an = nppad_end_zeros(a, n)
    anth = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    assert np.allclose(an, anth)


def test_nppad2_end_zeros():
    mat = np.array([[0, 1], [2, 3]])
    m = 3
    n = 4
    mp = nppad2_end_zeros(mat, m, n)
    mpth = np.array([[0, 1, 0, 0], [2, 3, 0, 0], [0, 0, 0, 0]])
    assert np.allclose(mp, mpth)


def test_bsgrid():
    g = bsgrid(np.arange(3), np.arange(4))
    gth = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
        ]
    )
    assert np.allclose(g, gth)


def test_make_lexico_grid():
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    mlg1 = make_lexico_grid(arr[:, 0])
    assert np.allclose(mlg1, arr[:, 0])
    mlg2 = make_lexico_grid(arr[:, :-1])
    assert np.allclose(mlg2, np.array([[0, 1], [0, 4], [3, 1], [3, 4]]))
    mlg = make_lexico_grid(arr)
    arrth = np.array(
        [
            [0, 1, 2],
            [0, 1, 5],
            [0, 4, 2],
            [0, 4, 5],
            [3, 1, 2],
            [3, 1, 5],
            [3, 4, 2],
            [3, 4, 5],
        ]
    )
    assert np.allclose(mlg, arrth)


def test_gauher():
    x8, w8 = gauher(8)

    def f4(x):
        return x**4

    ge4_pre = gaussian_expectation(f4, x=x8, w=w8)
    assert isclose(ge4_pre, 3.0)
    ge4_xw = np.sum(w8 / sqrt(pi) * f4(x8 * sqrt(2.0)))
    assert isclose(ge4_xw, 3.0)

    def fp(x, pars):
        return x ** pars[0]

    ge6_pars_pre = gaussian_expectation(fp, pars=[6], x=x8, w=w8)
    assert isclose(ge6_pars_pre, 15.0)

    def fpv(x, pars):
        return nppow(x, pars[0])

    ge6_parsv_pre = gaussian_expectation(fpv, vectorized=True, pars=[6], x=x8, w=w8)
    assert isclose(ge6_parsv_pre, 15.0)


def test_rice_stderr():
    n = 100_000
    sx = sy = 1.0
    x = np.arange(n) / n * sx
    rng = np.random.default_rng()
    y = x + rng.normal(size=n) * sy
    s = rice_stderr(y, x, is_sorted=True)
    assert isclose(np.mean(s), sy, rel_tol=0.05)
    x2 = rng.uniform(size=n) * sx
    y2 = x2 + rng.normal(size=n) * sy
    s2 = rice_stderr(y2, x2)
    assert isclose(np.mean(s2), sy, rel_tol=0.05)


def test_nplog():
    a = np.arange(1, 7).reshape((2, 3))
    a[0, 0] = exp(-100.0)
    res = nplog(a)
    resth00, dresth00, d2resth00 = bslog(a[0, 0], deriv=2)
    resth = np.array([[resth00, log(2), log(3)], [log(4), log(5), log(6)]])
    assert np.allclose(res, resth)
    res, dres = nplog(a, deriv=1)
    dresth = np.array([[dresth00, 1 / 2.0, 1 / 3.0], [1 / 4.0, 1 / 5.0, 1 / 6.0]])
    assert np.allclose(dres, dresth)
    res, dres, d2res = nplog(a, deriv=2)
    d2resth = np.array(
        [[d2resth00, -1 / 4.0, -1 / 9.0], [-1 / 16.0, -1 / 25.0, -1 / 36.0]]
    )
    assert np.allclose(d2res, d2resth)


def test_npexp():
    a = np.arange(1, 5).reshape((2, 2))
    a[0, 0] = -70.0
    a[1, 1] = 60.0
    res = npexp(a)
    resth00, dresth00, d2resth00 = bsexp(a[0, 0], deriv=2)
    resth11, dresth11, d2resth11 = bsexp(a[1, 1], deriv=2)
    resth = np.array([[resth00, exp(2)], [exp(3), resth11]])
    assert np.allclose(res, resth)
    res, dres = npexp(a, deriv=1)
    dresth = np.array([[dresth00, exp(2)], [exp(3), dresth11]])
    assert np.allclose(dres, dresth)
    res, dres, d2res = npexp(a, deriv=2)
    d2resth = np.array([[d2resth00, exp(2)], [exp(3), d2resth11]])
    assert np.allclose(d2res, d2resth)


def test_nppow_int():
    a = np.arange(1, 7).reshape((2, 3))
    b = 2
    res = nppow(a, b)
    resth = np.array([[1, 4, 9], [16, 25, 36]])
    assert np.allclose(res, resth)
    res, dresa, dresb = nppow(a, b, deriv=1)
    dresath = 2 * a
    loga = np.log(a)
    dresbth = resth * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    res, dresa, dresb, d2resaa, d2resab, d2resbb = nppow(a, b, deriv=2)
    d2resaath = np.full((2, 3), 2)
    d2resabth = a * (1.0 + b * loga)
    d2resbbth = resth * loga * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    assert np.allclose(d2resaa, d2resaath)
    assert np.allclose(d2resab, d2resabth)
    assert np.allclose(d2resbb, d2resbbth)


def test_nppow_float():
    a = np.arange(1, 7).reshape((2, 3))
    b = 2.0
    res = nppow(a, b)
    resth = np.array([[1, 4, 9], [16, 25, 36]])
    assert np.allclose(res, resth)
    res, dresa, dresb = nppow(a, b, deriv=1)
    dresath = 2 * a
    loga = np.log(a)
    dresbth = resth * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    res, dresa, dresb, d2resaa, d2resab, d2resbb = nppow(a, b, deriv=2)
    d2resaath = np.full((2, 3), 2)
    d2resabth = a * (1.0 + b * loga)
    d2resbbth = resth * loga * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    assert np.allclose(d2resaa, d2resaath)
    assert np.allclose(d2resab, d2resabth)
    assert np.allclose(d2resbb, d2resbbth)


def test_nppow_array():
    a = np.arange(1, 7).reshape((2, 3))
    b = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    res = nppow(a, b)
    resth = np.array([[1, 4, 27], [64, 25, 6]])
    assert np.allclose(res, resth)
    res, dresa, dresb = nppow(a, b, deriv=1)
    dresath = b * res / a
    loga = np.log(a)
    dresbth = resth * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    res, dresa, dresb, d2resaa, d2resab, d2resbb = nppow(a, b, deriv=2)
    d2resaath = b * (b - 1.0) * res / a / a
    d2resabth = res / a * (1.0 + b * loga)
    d2resbbth = resth * loga * loga
    assert np.allclose(res, resth)
    assert np.allclose(dresa, dresath)
    assert np.allclose(dresb, dresbth)
    assert np.allclose(d2resaa, d2resaath)
    assert np.allclose(d2resab, d2resabth)
    assert np.allclose(d2resbb, d2resbbth)


def test_find_row_single_nonzero():
    m1 = np.array([[2, 3, 0], [7, 4, 5], [0, 6, 0]])
    assert find_row_single_nonzero(m1) == (2, 1)
    m2 = np.array([[2, 3, 0], [1, 0, 5], [1, 6, 0]])
    assert find_row_single_nonzero(m2) is None
    m3 = np.array([[2, 3], [0, 0], [1, 6]])
    assert find_row_single_nonzero(m3) == (1, 0)


def test_make_lowertri():
    m6 = np.array(
        [
            [1, 2, 0, 1, 0, 2],
            [4, 1, 0, 5, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [2, 1, 7, 2, 1, 3],
            [2, 3, 2, 0, 0, 3],
            [0, 2, 0, 1, 0, 0],
        ]
    )
    lower_m6, prows_m6, pcols_m6 = make_lower_tri(m6)
    assert np.allclose(
        lower_m6,
        np.array(
            [
                [4, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0],
                [5, 1, 4, 0, 0, 0],
                [1, 2, 1, 2, 0, 0],
                [0, 3, 2, 3, 2, 0],
                [2, 1, 2, 3, 7, 1],
            ]
        ),
    )
    assert prows_m6 == [3, 6, 2, 1, 5, 4]
    assert pcols_m6 == [4, 2, 1, 6, 3, 5]
    m = np.full((3, 3), 1)
    assert make_lower_tri(m) is None
