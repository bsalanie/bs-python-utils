"""
Contains various `numpy` utility programs.
"""

import sys
from math import cos, exp, floor, log, pi, sqrt
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.polynomial import Polynomial

from bs_python_utils.bsutils import bs_error_abort, print_stars


# some useful types
TwoArrays = Tuple[np.ndarray, np.ndarray]
ThreeArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]
FourArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
SixArrays = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


# Numpy parallel RNG
def generate_RNG_streams(
    nsim: int, initial_seed: Optional[int] = 13091962
) -> list[np.random.Generator]:
    """
    return `nsim` RNGs

    Args:
        nsim:  number of RNGs we want
        initial_seed: any large integer

    Returns:
        `nsim` streams

    Example:
        streams = generate_RNG_streams(10, 575856896)
        x = streams[i].normal(scale=s, size=(nmarkets, nproducts))
    """
    ss = np.random.SeedSequence(initial_seed)
    # Spawn off child SeedSequences to pass to child processes.
    child_seeds = ss.spawn(nsim)
    streams = [np.random.default_rng(s) for s in child_seeds]
    return streams


def ecdf(x: np.ndarray) -> np.ndarray:
    """Evaluate the empirical cdf at each point in sample

    Args:
        x: 1-dim array `(nobs)`

    Returns:
        A 1-dim array `(nobs)`  with the values of the empirical cdf at `x`, from 1/`nobs` to 1

    """
    if x.ndim != 1:
        print_stars(f"ecdf: x should have 1 dimension, not {x.ndim}")
        sys.exit()
    nx = x.size
    order_x = np.argsort(x)
    ecdf_val = np.zeros(nx)
    for i_order, n_order in enumerate(order_x):
        ecdf_val[n_order] = (i_order + 1.0) / nx
    return ecdf_val


def inv_ecdf(v: np.ndarray, q: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Evaluate the empirical `q`-quantiles of the sample `v`
    in a way that is consistent with `ecdf`.

    Args:
        v: 1-dim array `(nobs)` of the data points
        q: 1-dim array `(nobs)` of quantiles or float

    Returns:
        A 1-dim array `(nobs)`  with the values of the `q`-quantiles of `v`, or just the one quantile

    """
    if v.ndim != 1:
        bs_error_abort(f"v should have 1 dimension, not {v.ndim}")
    nv = v.size
    sorted_v = np.zeros(nv + 2)
    sorted_v[1 : (nv + 1)] = np.sort(v)
    sorted_v[0] = 2.0 * sorted_v[1] - sorted_v[2]  # added to extend for q < 1/nv
    sorted_v[nv + 1] = sorted_v[nv]  # added to extend for q = 1
    if isinstance(q, float):
        q = np.array([q])
        q_floor = floor(nv * q)
    elif isinstance(q, np.ndarray):
        q_floor = np.floor(nv * q).astype(int)
    return sorted_v[q_floor] + (nv * q - q_floor) * (
        sorted_v[q_floor + 1] - sorted_v[q_floor]
    )


def nprepeat_col(v: np.ndarray, n: int) -> np.ndarray:
    """
    create a matrix with `n` columns equal to `v`

    Args:
        v: a 1-dim array of size `m`
        n: the number of columns requested

    Returns:
        a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[:, np.newaxis], n, axis=1)


def nprepeat_row(v: np.ndarray, m: int) -> np.ndarray:
    """
    create a matrix with `m` rows equal to `v`

    Args:
        v: a 1-dim array of size `n`
        m: the number of rows requested

    Returns:
        a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[np.newaxis, :], m, axis=0)


def npmaxabs(arr: np.ndarray) -> float:
    """
    maximum absolute value in an array

    Args:
        arr: any Numpy array

    Returns:
        the largest element in absolute value
    """
    return np.max(np.abs(arr))


def rice_stderr(
    y: np.ndarray, x: np.ndarray, sorted: Optional[bool] = False
) -> np.ndarray | float:
    """
    computes the Rice local estimators of the standard error of y | x

    Args:
        y: vector of y-values
        x: vector of x-values
        sorted: set it to `True` if `x` is in increasing order

    Returns:
        an array of the same size with the stderr(y | x)
    """
    n = test_vector(x)
    ny = test_vector(y)
    if ny != n:
        bs_error_abort("x and y should have the same size")

    if not sorted:
        # need to sort by increasing value of x
        order_x = np.argsort(x)
        ys = y[order_x]
    else:
        ys = y

    variance_estimator = np.zeros(n)

    # we average over neighbors
    n_neighbors = int(sqrt(float(n)) / 2.0)
    facd = 1.0 / (2.0 * n_neighbors)
    n_neighbors2 = n_neighbors // 2

    # for the first observations
    yleft = ys[:n_neighbors2]
    dy = yleft[1:] - yleft[:-1]
    variance_estimator[:n_neighbors2] = np.sum(dy * dy) * facd

    # for the middle of the sample
    minus_nn2 = n - n_neighbors2
    for ix in range(n_neighbors2, minus_nn2):
        ix_neighbors = slice(ix - n_neighbors2, ix + n_neighbors2)
        yx = ys[ix_neighbors]
        dy = yx[1:] - yx[:-1]
        variance_estimator[ix] = np.sum(dy * dy) * facd

    # and for the last observations
    yright = ys[minus_nn2:]
    dy = yright[1:] - yright[:-1]
    variance_estimator[minus_nn2:] = np.sum(dy * dy) * facd

    stderr_estimator = np.sqrt(variance_estimator)

    return stderr_estimator


def nplog(
    arr: np.ndarray,
    eps: Optional[float] = 1e-30,
    deriv: Optional[int] = 0,
    verbose: Optional[bool] = False,
) -> np.ndarray | TwoArrays | ThreeArrays:
    """
    `C^2` extension of  `\\ln(a)` below `eps`, perhaps with derivatives

    Args:
        arr: any Numpy array
        eps: lower bound
        deriv: if 1, compute derivative, if 2, second derivative
        verbose: prints debugging info

    Returns:
        `\\ln(a)` `C^2`-extended below `eps`, perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv can only be 0, 1, or 2; not {deriv}")
    if np.min(arr) > eps:
        if deriv == 0:
            return np.log(arr)
        elif deriv == 1:
            return np.log(arr), 1.0 / arr
        elif deriv == 2:
            return np.log(arr), 1.0 / arr, -1.0 / (arr * arr)
    else:
        logarreps = np.log(np.maximum(arr, eps))
        darr = 1.0 - arr / eps
        logarr_smaller = log(eps) - darr * (1.0 + darr / 2.0)
        if verbose:
            n_small_args = np.sum(arr < eps)
            if n_small_args > 0:
                finals = "s" if n_small_args > 1 else ""
                print(
                    f"nplog: {n_small_args} argument{finals} smaller than {eps}: mini ="
                    f" {np.min(arr)}"
                )
        logeps = np.where(arr > eps, logarreps, logarr_smaller)
        if deriv == 0:
            return logeps
        else:
            arreps = np.maximum(arr, eps)
            der_logarreps = 1.0 / arreps
            der_logarr_smaller = (1.0 + darr) / eps
            dlogeps = np.where(arr > eps, der_logarreps, der_logarr_smaller)
            if deriv == 1:
                return logeps, dlogeps
            else:
                der2_logarreps = -1.0 / (arreps * arreps)
                der2_logarr_smaller = np.full(arr.shape, -1.0 / (eps * eps))
                d2logeps = np.where(arr > eps, der2_logarreps, der2_logarr_smaller)
                return logeps, dlogeps, d2logeps


def npexp(
    arr: np.ndarray,
    bigx: Optional[float] = 50.0,
    lowx: Optional[float] = -50.0,
    deriv: Optional[int] = 0,
    verbose: Optional[bool] = False,
) -> np.ndarray | TwoArrays | ThreeArrays:
    """
    `C^2` extension of  `\\exp(a)` above `bigx` and below `lowx`,
    perhaps with derivatives

    Args:
        arr: any Numpy array
        bigx: upper bound
        lowx: lower bound
        deriv: if 1, compute derivative, if 2, second derivative
        verbose: prints debugging info


    Returns:
        `\\exp(a)`  `C^2`-extended above `bigx` and below `lowx`,
        perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv can only be 0, 1, or 2; not {deriv}")
    min_arr, max_arr = np.min(arr), np.max(arr)
    if max_arr <= bigx and min_arr >= lowx:
        exparr = np.exp(arr)
        if deriv == 0:
            return exparr
        elif deriv == 1:
            return exparr, exparr
        elif deriv == 2:
            return exparr, exparr, exparr
    else:  # some large and/or small arguments
        exparr = np.exp(np.maximum(np.minimum(arr, bigx), lowx))
        print(f"{exparr=}")
        ebigx = exp(bigx)
        elowx = exp(lowx)
        darrb = arr - bigx
        darrl = lowx - arr
        exparr_larger = ebigx * (1.0 + darrb * (1.0 + 0.5 * darrb))
        exparr_smaller = elowx * (1.0 - darrl * (1.0 - 0.5 * darrl))
        if verbose:
            n_large_args = np.sum(arr > bigx)
            finals = "s" if n_large_args > 1 else ""
            print(
                f"npexp: {n_large_args} argument{finals} larger than {bigx}:\n"
                f"maxi = {np.max(arr)}"
            )
            n_small_args = np.sum(arr < lowx)
            finals = "s" if n_small_args > 1 else ""
            print(
                f"npexp: {n_small_args} argument{finals} smaller than {lowx}:\n"
                f"mini = {np.min(arr)}"
            )
        expval = exparr
        print(expval)
        expval = np.where(arr > bigx, exparr_larger, expval)
        expval = np.where(arr < lowx, exparr_smaller, expval)
        if deriv == 0:
            return expval
        dexpval = exparr
        dexparr_larger = ebigx * (1.0 + darrb)
        dexparr_smaller = elowx * (1.0 - darrl)
        dexpval = np.where(arr > bigx, dexparr_larger, dexpval)
        dexpval = np.where(arr < lowx, dexparr_smaller, dexpval)
        if deriv == 1:
            return expval, dexpval
        if deriv == 2:
            d2expval = exparr
            return expval, dexpval, d2expval


def nppow(
    a: np.ndarray, b: Union[int, float, np.ndarray], deriv: Optional[int] = 0
) -> np.ndarray | ThreeArrays | SixArrays:
    """
    evaluates a**b element-by-element, perhaps with derivatives

    Args:
        a: an array
        b: if an array, should have the same shape as `a`
        deriv: if 1, compute derivative, if 2, second derivative

    Returns:
        an array of the same shape as `a`
    """
    if isinstance(b, float):
        mina = np.min(a)
        if mina < 0.0:
            bs_error_abort("All elements of a must be positive!")

    if isinstance(b, (int, float)):
        a_pow_b = a**b
        if deriv == 0:
            return a_pow_b
        log_a = np.log(a)
        derivs1 = (b * a_pow_b / a, a_pow_b * log_a)
        if deriv == 1:
            return a_pow_b, *derivs1
        b1 = b - 1.0
        a_pow_b1 = a_pow_b / a
        if deriv == 2:
            derivs2 = (
                b * b1 * a_pow_b1 / a,
                a_pow_b1 * (1.0 + b * log_a),
                a_pow_b * log_a * log_a,
            )
            return a_pow_b, *derivs1, *derivs2
    else:
        if a.shape != b.shape:
            bs_error_abort("b is not a number or an array of the same shape as a!")
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec**bvec
        a_pow_br = a_pow_b.reshape(a.shape)
        if deriv == 0:
            return a_pow_br
        der_wrt_a = a_pow_b * bvec / avec
        log_avec = nplog(avec)
        der_wrt_b = a_pow_b * log_avec
        derivs1 = (der_wrt_a.reshape(a.shape), der_wrt_b.reshape(a.shape))
        if deriv == 1:
            return a_pow_br, *derivs1
        a_pow_b1 = a_pow_b / avec
        b1 = bvec - 1.0
        der2_wrt_aa = bvec * b1 * a_pow_b1 / avec
        der2_wrt_ab = a_pow_b1 * (1.0 + bvec * log_avec)
        der2_wrt_bb = a_pow_b * log_avec * log_avec
        derivs2 = (
            der2_wrt_aa.reshape(a.shape),
            der2_wrt_ab.reshape(a.shape),
            der2_wrt_bb.reshape(a.shape),
        )
        if deriv == 2:
            return a_pow_br, *derivs1, *derivs2


def nppad_beg_zeros(v: np.ndarray, n: int) -> np.ndarray:
    """
    pad the beginning of a 1-dim array with zeros to increase its size to `n`, if needed

    Args:
        v: 1-dim array of size `(nv)`
        n: size requested

    Returns:
        padded array if `nv` < `n`, otherwise `v`
    """
    nv = test_vector(v)
    if nv < n:
        return np.pad(v, (n - nv, 0))
    else:
        return v


def nppad_end_zeros(v: np.ndarray, n: int) -> np.ndarray:
    """
    pad the end of a 1-dim array with zeros to increase its size to `n`, if needed

    Args:
        v: 1-dim array of size `(nv)`
        n: size requested

    Returns:
        padded array if `nv` < `n`, else `v`
    """
    nv = test_vector(v)
    if nv < n:
        return np.pad(v, (0, n - nv))
    else:
        return v


def nppad2_end_zeros(mat: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    pad the ends of a 2-dim array with zeros to increase its size to `(m,n)`, if needed

    Args:
        mat: 2-dim array
        m: number of rows requested
        n: number of columns requested

    Returns:
        padded array, where needed
    """
    nrows, ncols = test_matrix(mat)
    max_rows = max(m, nrows)
    max_cols = max(n, ncols)
    if nrows < max_rows and ncols < max_cols:  # pad both dimensions
        pmat = np.zeros((m, n))
        pmat[:nrows, :ncols] = mat
        return pmat
    elif nrows < max_rows:  # pad rows
        pmat = np.zeros((m, ncols))
        pmat[:nrows, :] = mat
        return pmat
    elif ncols < max_cols:  # pad columns
        pmat = np.zeros((nrows, n))
        pmat[:, :ncols] = mat
        return pmat
    else:  # no need for padding
        return mat


def bsgrid(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    make a two-dimensional matrix of all pairs of elements of the vectors `v` and `w`

    Args:
        v: basis vector, size m
        w: basis vector, size n

    Returns: an array of shape `(m n, 2)`
    """
    m = test_vector(v)
    n = test_vector(w)
    m, n = v.size, w.size
    v1 = np.repeat(v, n)
    v2 = np.tile(w, m)
    return np.column_stack((v1, v2))

    """
    This is a Python function that tests whether a given input is a vector and returns its size if
    successful.
    
    :param v: `v` is a numpy array that is expected to be a vector
    :type v: np.ndarray
    :param fun_name: `fun_name` is an optional parameter that represents the name of the calling
    function. If provided, it will be used in error messages to indicate which function caused the
    error. If not provided, the error message will not include the function name
    :type fun_name: Optional[str]
    """


def test_vector(v: np.ndarray, fun_name: Optional[str] = None) -> int:
    """
    test that `v` is a vector; aborts otherwise

    Args:
        v: a vector, we hope
        fun_name: name of the calling function

    Returns:
        the size if successful
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(v, np.ndarray):
        bs_error_abort(f"{fun_str} v should be a Numpy array")
    ndims_v = v.ndim
    if ndims_v != 1:
        bs_error_abort(f"{fun_str} v should have one dimension, not {ndims_v}")
    return v.size


def test_matrix(x: np.ndarray, fun_name: Optional[str] = None) -> Tuple[int, int]:
    """
    test that `x` is a matrix; aborts otherwise

    Args:
        x: a matrix, we hope
        fun_name: name of the calling function

    Returns:
        the shape if successful
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(x, np.ndarray):
        bs_error_abort(f"{fun_str} Xx should be a Numpy array")
    ndims_x = x.ndim
    if ndims_x != 2:
        bs_error_abort(f"{fun_str} x should have two dimensions, not {ndims_x}")
    return x.shape


def test_vector_or_matrix(x: np.ndarray, fun_name: Optional[str] = None) -> int:
    """
    test that `x` is a vector or a matrix; aborts otherwise

    Args:
        x: a vector or matrix, we hope
        fun_name: name of the calling function

    Returns:
        the number of dimensions of `x` (1 or 2)
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(x, np.ndarray):
        bs_error_abort(f"{fun_str} X should be a Numpy array")
    ndims_x = x.ndim
    if ndims_x != 1 and ndims_x != 2:
        bs_error_abort(f"{fun_str} x should have at most two dimensions, not {ndims_x}")
    return ndims_x


def bs_sqrt_pdmatrix(m: np.ndarray) -> np.ndarray:
    """
    square root of a positive definite matrix

    Args:
        m: a positive definite matrix

    Returns:
        the square root of the matrix
    """
    _ = test_square(m, "bs_sqrt_pdmatrix")
    eigval, eigvec = np.linalg.eigh(m)
    eigval = np.maximum(eigval, 0.0)
    eigval_sqrt = np.sqrt(eigval)
    eigval_sqrt_diag = np.diag(eigval_sqrt)
    res = eigvec @ eigval_sqrt_diag @ eigvec.T
    return res


def test_square(A: np.ndarray, fun_name: Optional[str] = None) -> int:
    """
    test that a matrix used in `fun_name` is square

    Args:
        A: square matrix, we hope
        fun_name: the name of the calling function

    Returns:
        the number of rows and columns of `A`
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(A, np.ndarray):
        bs_error_abort(f"{fun_str} A should be a Numpy array")
    if A.ndim == 2:
        n, nv = A.shape
        if nv != n:
            bs_error_abort(f"{fun_str} The matrix A should be square, not {A.shape}")
    else:
        bs_error_abort(f"{fun_name} A should have  two dimensions, not {A.ndim}")
    return n


def test_tensor(
    x: np.ndarray, n_dims: int, fun_name: Optional[str] = None
) -> Tuple[int]:
    """
    test that `x` is an `n_dims` dimensional array; aborts otherwise

    Args:
        x: an `n_dims` dimensional array, we hope
        fun_name: name of the calling function

    Returns:
        the shape if successful
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(x, np.ndarray):
        bs_error_abort(f"{fun_str} x should be a Numpy array")
    ndims_x = x.ndim
    if ndims_x != n_dims:
        bs_error_abort(f"{fun_str} x should have {n_dims} dimensions, not {ndims_x}")
    return x.shape


def make_lexico_grid(arr: np.ndarray) -> np.ndarray:
    """
    make a lexicographic grid

    Args:
        arr: `nr`-vector or `(nr,nc)` matrix; `nc` must be 1, 2 or 3

    Returns:
        `arr` if it is a vector; otherwise a matrix `({nr}^{nc}, {nc})`
         for `nc=2`  it is like `bsgrid`
    """
    ndims_arr = test_vector_or_matrix(arr, "make_lexico_grid`")
    if ndims_arr == 1:
        return arr
    else:
        nr, nc = arr.shape
        if nc == 2:
            n0 = np.repeat(arr[:, 0], nr)
            n1 = np.tile(arr[:, 1], nr)
            return np.column_stack((n0, n1))
        elif nc == 3:
            nr2 = nr * nr
            n0 = np.repeat(arr[:, 0], nr2)
            n1 = np.repeat(np.tile(arr[:, 1], nr), nr)
            n2 = np.tile(arr[:, 2], nr2)
            return np.column_stack((n0, n1, n2))
        else:
            bs_error_abort(
                f"at this stage, the number of columns must be 3 or less, not {nc}..."
            )


class BivariatePolynomial:
    """
    a class for bivariate polynomials as a list of `Polynomial` objects

    minimal interface:

        * construct from matrix of coefficients
        * add, subtract, multiply (with constant and with :class:`BivariatePolynomial`)
        * evaluate  p(x, y) when x, y are at most vectors (and have the same shape if both vectors)
    """

    def __init__(self, coeffs: np.ndarray):
        """
        constructor for :class:`BivariatePolynomial`

        coeffs: a two-dimensional array `(deg1+1, deg2+2)`
        """
        self.deg1, self.deg2 = coeffs.shape[0] - 1, coeffs.shape[1] - 1
        self.coef = coeffs
        self.listpol2 = []
        for k in range(self.deg1 + 1):
            self.listpol2.append(Polynomial(coeffs[k, :]))

    def __add__(self, bivpol):
        if isinstance(bivpol, (int, float)):
            coeffs = self.coef.copy()
            coeffs[0, 0] += bivpol
            return BivariatePolynomial(coeffs)
        degbp1, degbp2 = bivpol.deg1, bivpol.deg2
        max_deg1 = max(degbp1, self.deg1)
        max_deg2 = max(degbp2, self.deg2)
        coeffs_new = nppad2_end_zeros(self.coef, max_deg1 + 1, max_deg2 + 1)
        coeffsbp_new = nppad2_end_zeros(bivpol.coef, max_deg1 + 1, max_deg2 + 1)
        return BivariatePolynomial(coeffs_new + coeffsbp_new)

    def __repr__(self):
        return f"BivariatePolynomial({self.deg1!r}, {self.deg2!r})"

    def __iadd__(self, bivpol):
        return self.__add__(bivpol)

    def __radd__(self, bivpol):
        return self.__add__(bivpol)

    def __sub__(self, bivpol):
        if isinstance(bivpol, (int, float)):
            coeffs = self.coef.copy()
            coeffs[0, 0] -= bivpol
            return BivariatePolynomial(coeffs)
        degbp1, degbp2 = bivpol.deg1, bivpol.deg2
        max_deg1 = max(degbp1, self.deg1)
        max_deg2 = max(degbp2, self.deg2)
        coeffs_new = nppad2_end_zeros(self.coef, max_deg1 + 1, max_deg2 + 1)
        coeffsbp_new = nppad2_end_zeros(bivpol.coef, max_deg1 + 1, max_deg2 + 1)
        return BivariatePolynomial(coeffs_new - coeffsbp_new)

    def __mul__(self, bivpol):
        if isinstance(bivpol, (int, float)):
            return BivariatePolynomial(bivpol * self.coef)
        deg1, degbp1 = self.deg1, bivpol.deg1
        deg2, degbp2 = self.deg2, bivpol.deg2
        degmul1 = deg1 + degbp1
        degmul2 = deg2 + degbp2
        lp2, blp2 = self.listpol2, bivpol.listpol2

        coeffs_mul = np.zeros((degmul1 + 1, degmul2 + 1))
        for m in range(degmul1 + 1):
            minm = max(0, m - degbp1)
            maxm = min(m, self.deg1)
            pm = Polynomial(0)
            for i in range(minm, maxm + 1):
                pm += lp2[i] * blp2[m - i]
            coeffs_mul[m, :] += pm.coef

        bp_mul = BivariatePolynomial(coeffs_mul)
        return bp_mul

    def __rmul__(self, bivpol):
        return self.__mul__(bivpol)

    def __radd__(self, bivpol):
        return self.__add__(bivpol)

    def __call__(self, x1, x2):
        x1fac = 1.0
        val = 0.0
        for p in self.listpol2:
            val += p(x2) * x1fac
            x1fac *= x1
        return val


def outer_bivar(pol1: Polynomial, pol2: Polynomial) -> BivariatePolynomial:
    """
    make a `BivariatePolynomial` from the  product of two `Polynomial` objects

    Args:
        pol1: Polynomial in the first variable
        pol2: Polynomial in the second variable

    Returns:
        a `BivariatePolynomial` = `pol1 * pol2`
    """
    p1 = pol1.coef
    p2 = pol2.coef
    prod_coef = np.outer(p1, p2)
    return BivariatePolynomial(prod_coef)


def npxlogx(
    arr: np.ndarray,
    eps: Optional[float] = 1e-30,
    deriv: Optional[int] = 0,
    verbose: Optional[bool] = False,
) -> np.ndarray:
    """
    C^2` extension of  `a\\ln(a)` below `eps`, perhaps with derivatives

    Args:
        arr: a Numpy array
        eps: lower bound
        deriv: if 1, compute derivative, if 2, second derivative
        verbose: prints debugging info

    Returns:
        `a\\ln(a)`  `C^2`-extended  below `eps`, perhaps with derivatives
    """
    if deriv not in [0, 1, 2]:
        bs_error_abort(f"deriv must be 0, 1, or 2; not {deriv}")
    if np.min(arr) > eps:
        return arr * np.log(arr)
    else:
        logeps = log(eps)
        logarreps = np.log(np.maximum(arr, eps))
        xlogarreps = arr * logarreps
        xlogarr_smaller = arr * (arr / eps + logeps - 1.0)
        if verbose:
            n_small_args = np.sum(arr < eps)
            if n_small_args > 0:
                finals = "s" if n_small_args > 1 else ""
                print(
                    f"npxlogx: {n_small_args} argument{finals} smaller than {eps}: mini"
                    f" = {np.min(arr)}"
                )
        xlogval = np.where(arr > eps, xlogarreps, xlogarr_smaller)
        if deriv == 0:
            return xlogval
        dxlogarreps = 1.0 + logarreps
        dxlogarr_smaller = logeps + arr / eps
        dxlogval = np.where(arr > eps, dxlogarreps, dxlogarr_smaller)
        if deriv == 1:
            return xlogval, dxlogval
        if deriv == 2:
            d2xlogval = 1.0 / np.maximum(arr, eps)
            return xlogval, dxlogval, d2xlogval


def gauher(n: int) -> TwoArrays:
    """
    nodes and weights for Gauss-Hermite integration

    Args:
        n: number of nodes

    Returns:
        array of `n` nodes, array of `n` weights
    """
    EPS = 1.0e-14
    PIM4 = 0.7511255444649425
    MAXIT = 10

    x = np.zeros(n)
    w = np.zeros(n)

    m = (n + 1) // 2

    for i in range(m):
        if i == 0:
            n2 = 2.0 * n + 1.0
            z = sqrt(n2) - 1.85575 * (n2**-0.16667)
        elif i == 1:
            z -= 1.14 * (n**0.426) / z
        elif i == 2:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2.0 * z - x[i - 2]
        for its in range(MAXIT):
            p1 = PIM4
            p2 = 0.0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = z * sqrt(2.0 / (j + 1)) * p2 - sqrt(j / (j + 1)) * p3
            pp = sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= EPS:
                break
        if its >= MAXIT:
            sys.exit("too many iterations in gauher")
        x[i] = z
        x[n - 1 - i] = -z
        w[i] = 2.0 / (pp * pp)
        w[n - 1 - i] = w[i]

    # need to reverse order for x (w is symmetric)
    return x[::-1], w


def gauleg(n: int) -> TwoArrays:
    """
    nodes and weights for Gauss-Legendre integration `\\int_{-1}^1 f(x)dx`

    Args:
        n: number of nodes

    Returns:
        array of `n` nodes, array of `n` weights
    """
    x = np.zeros(n)
    w = np.zeros(n)
    EPS = 3e-11
    m = (n + 1) // 2
    for i in range(1, m + 1):
        z = cos(pi * (i - 0.25) / (n + 0.5))
        z1 = np.inf
        while abs(z - z1) > EPS:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        x[i - 1] = -z
        x[n - i] = z
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n - i] = w[i - 1]

    return x, w


def gaussian_expectation(
    f: Callable,
    vectorized: Optional[bool] = False,
    pars: Optional[Iterable] = None,
    n: Optional[int] = 16,
    x: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
) -> Union[np.ndarray, float]:
    """
    computes the expectation of a function of an `N(0,1)` random variable
     using Gauss-Hermite with n nodes
     the nodes and weights can be provided, if available

    Args:
        f: a scalar or array function of a scalar or array variable and possibly other parameters
        vectorized: if True, the function accepts an array as argument
        pars: parameters for `f`, if any
        n: number of nodes
        x: locations of the nodes
        w: their weights

    Returns:
        the expectation of `f(N(0,1))`
    """
    if x is None:
        nodes, weights = gauher(n)
        nodes *= sqrt(2.0)
        weights /= sqrt(pi)
        n_nodes = n
    elif w is None:
        bs_error_abort("x is None but w is not")
    elif w.size != x.size:
        bs_error_abort("x has {x.size} elements and w has {w.size}")
    else:
        nodes = x * sqrt(2.0)
        weights = w / sqrt(pi)
        n_nodes = nodes.size
    if pars is None:
        if vectorized:
            integral_val = f(nodes) @ weights
        else:
            # to ensure integral_val has the same shape as f
            integral_val = weights[0] * f(nodes[0])
            for i in range(1, n_nodes):
                integral_val += weights[i] * f(nodes[i])
    else:
        if vectorized:
            integral_val = f(nodes, pars) @ weights
        else:
            # to ensure integral_val has the same shape as f
            integral_val = weights[0] * f(nodes[0], pars)
            for i in range(1, n_nodes):
                integral_val += weights[i] * f(nodes[i], pars)

    return integral_val
