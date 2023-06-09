"""
Contains some statistical routines.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import cast

import numpy as np
import scipy.linalg as spla
import scipy.stats as sts
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg

from bs_python_utils.bsnputils import (
    check_matrix,
    check_vector,
    check_vector_or_matrix,
    make_lexico_grid,
)
from bs_python_utils.bssputils import spline_reg
from bs_python_utils.bsutils import bs_error_abort


@dataclass
class TslsResults:
    """
    contains full results of a TSLS regression
    """

    iv_estimates: float | np.ndarray | None
    r2_first_iv: float | np.ndarray | None
    r2_y: float | None
    r2_second: float | np.ndarray | None
    y_proj: float | np.ndarray | None
    y_coeffs: float | np.ndarray | None
    X_IV_proj: float | np.ndarray | None
    b_proj_IV: float | np.ndarray | None


def _powers_Z(Z: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """
    used internally by `proj_Z`; returns `\\prod_{k=1}^m  Z_{\\cdot k}^{l_k}`

    Args:
        Z: a matrix `(n, m)`
        list_ints: a list of integers

    Returns:
        the product of the powers of `Z`
    """
    if Z.ndim != 2:
        bs_error_abort(f"Z should have dimension 2, not {Z.ndim}")
    m = Z.shape[1]
    mdegs = check_vector(degrees)
    if mdegs != m:
        bs_error_abort("The size of degrees should equal the number of columns of Z")
    res = np.ones(Z.shape[0])
    for i, degi in enumerate(degrees):
        res *= Z[:, i] ** degi
    return res


def _final_proj(Zp: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    MINVAR = 1e-12
    b_proj, _, _, _ = spla.lstsq(Zp, W)
    W_proj = Zp @ b_proj
    if W.ndim == 1:
        var_w = np.var(W)
        r2 = np.var(W_proj) / var_w if var_w > MINVAR else 1.0
    elif W.ndim == 2:
        nw = W.shape[1]
        r2 = np.ones(nw)
        for i in range(nw):
            var_w = np.var(W[:, i])
            if var_w > MINVAR:
                r2[i] = np.var(W_proj[:, i]) / var_w
    else:
        bs_error_abort(f"Wrong number of dimensions {W.ndim} for W")
    return W_proj, b_proj, r2


def _make_Zp(Z: np.ndarray, p: int) -> tuple[np.ndarray, int]:
    nobs, m = Z.shape
    list_vars = list(range(m))
    MAX_NTERMS = round(nobs / 5)
    Zp = np.zeros((nobs, MAX_NTERMS))
    Zp[:, 0] = np.ones(nobs)
    k = 1
    for q in range(1, p + 1):
        listq = list(combinations_with_replacement(list_vars, q))
        lenq = len(listq)
        degrees = np.zeros((m, lenq))
        for i in range(m):
            degrees[i, :] = np.ndarray([x.count(i) for x in listq])
        for j in range(lenq):
            Zp[:, k] = _powers_Z(Z, degrees[:, j])
            k += 1
            if k >= MAX_NTERMS:
                bs_error_abort(f"We don't allow more than {MAX_NTERMS} terms")
    Zp = Zp[:, :k]
    return Zp, k


def proj_Z(
    W: np.ndarray, Z: np.ndarray, p: int = 1, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    project `W` on `Z` up to degree `p` interactions

    Args:
        W: variable(s) `(nobs)` or `(nobs, nw)`
        Z: instruments `(nobs) or `(nobs, nz)`;
            they should **not** include a constant term
        p: maximum total degree for interactions of the columns of `Z`
        verbose: prints stuff if True

    Returns:
        the projections of the columns of `W` on `Z` etc, the coefficients, and the `R^2` of each column
    """
    nobs = Z.shape[0]
    if W.shape[0] != nobs:
        bs_error_abort("W and Z should have the same number of rows")
    if W.ndim > 2:
        bs_error_abort("W should have 1 or 2 dimensions")
    if Z.ndim > 2:
        bs_error_abort("Z should have 1 or 2 dimensions")

    if Z.ndim == 1:
        Zp = np.zeros((nobs, 1 + p))
        Zp[:, 0] = np.ones(nobs)
        for q in range(1, p + 1):
            Zp[:, q] = Z**q
    else:  # Z is a matrix
        Zp, k = _make_Zp(Z, p)
        if verbose:
            print(f"_proj_Z with degree {p}, using {k} regressors")

    return _final_proj(Zp, W)


def tsls(y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> TslsResults:
    """
    TSLS of `y` on `X` with instruments `Z`

    Args:
        y: independent variable `(nobs)`
        X: covariates `(nobs, nx)`
        Z: instruments `(nobs, nz)`

    Returns:
        a `tsls_results` object
    """
    # first stage
    X_IV_proj, b_proj_IV, r2_first_iv = proj_Z(X, Z)
    # second stage
    y_proj, y_coeffs, r2_y = proj_Z(y, Z)
    _, iv_estimates, r2_second = proj_Z(y_proj, X_IV_proj)
    return TslsResults(
        iv_estimates,
        r2_first_iv,
        r2_y,
        r2_second,
        y_proj,
        y_coeffs,
        X_IV_proj,
        b_proj_IV,
    )


def reg_nonpar(
    y: np.ndarray,
    X: np.ndarray,
    var_types: str | None = None,
    n_sub: int | None = None,
    n_res: int | None = 1,
) -> tuple[KernelReg, np.ndarray]:
    """
    nonparametric regression of y on the columns of X;
    bandwidth chosen on a subsample of size nsub if nsub < nobs, and rescaled

    Args:
        y: a vector of size nobs
        X: a (nobs) vector or a matrix of shape (nobs, m)
        var_types: specify types of all `X` variables if not all of them are continuous;
            one character per variable
            * 'c' for continuous
            * 'u' discrete unordered
            * 'o' discrete ordered
        n_sub: size of subsample for cross-validation;  by default it is `200^{(m+4)/5}`
        n_res: how many subsamples we draw; 1 by default

    Returns:
        fitted on sample (nobs, with derivatives)
        and bandwidths (m)
    """
    _ = check_vector_or_matrix(X)
    n_obs = check_vector(y)
    if X.shape[0] != n_obs:
        bs_error_abort("X and y should have the same number of observations")
    m = 1 if X.ndim == 1 else X.shape[1]
    if var_types is None:
        types = "c" * m
    else:
        if len(var_types) != m:
            bs_error_abort("var_types should have one entry for each column of X")
        types = var_types

    if n_sub is None:
        n_sub = round(200 ** ((m + 4.0) / 5.0))

    k = KernelReg(
        y,
        X,
        var_type=types,
        defaults=EstimatorSettings(
            efficient=True, n_sub=n_sub, randomize=True, n_res=n_res
        ),
    )
    return k.fit(), k.bw


def reg_nonpar_fit(
    y: np.ndarray,
    X: np.ndarray,
    var_types: str | None = None,
    n_sub: int | None = None,
    n_res: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    nonparametric regression of y on the columns of X; bandwidth chosen on a subsample of size nsub if nsub < nobs, and rescaled

    Args:
        y: a vector of size nobs
        X: a (nobs) vector or a matrix of shape (nobs, m)
        var_types: specify types of all `X` variables if not all of them are continuous;
            one character per variable
            * 'c' for continuous
            * 'u' discrete unordered
            * 'o' discrete ordered
        n_sub: size of subsample for cross-validation; by default it is `200^{(m+4)/5}`
        n_res: how many subsamples we draw; 1 by default
        verbose: prints stuff if True

    Returns:
        fitted values on sample (nobs)
    """
    kfbw = reg_nonpar(y, X, var_types, n_sub, n_res)
    fitted_vals = cast(np.ndarray, kfbw[0][0])
    return fitted_vals


def flexible_reg(
    Y: np.ndarray,
    X: np.ndarray,
    mode: str = "NP",
    var_types: str | None = None,
    n_sub: int | None = None,
    n_res: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """
    flexible regression  of `Y` on `X`

    Args:
        Y: independent variable `(nobs)` or `(nobs, ny)`
        X: covariates `(nobs)` or `(nobs, nx)`; should **not** include a constant term
        mode: what flexible means
            * 'NP': non parametric
            * 'SPL': spline regression, only on one covariate
            * '1': linear
            * '2': quadratic
        var_types: [for 'NP' only]  specify types of all `X` variables if not all of them are continuous; 
            one character per variable
            * 'c' for continuous
            * 'u' discrete unordered
            * 'o' discrete ordered
        n_sub: [for 'NP' only] size of subsample for cross-validation; \
            by default it is `200^{(m+4)/5}`
        n_res: [for 'NP' only] how many subsamples we draw; 1 if `None`
        verbose: prints stuff if True

    Returns: 
        `E(y|X)` at the sample points
    """
    if mode == "NP":
        if Y.ndim == 2:
            ny = Y.shape[1]
            Y_fit = np.zeros_like(Y)
            for iy in range(ny):
                Y_fit[:, iy] = reg_nonpar_fit(
                    Y[:, iy],
                    X,
                    var_types=var_types,
                    n_sub=n_sub,
                    n_res=n_res,
                    verbose=verbose,
                )
            return Y_fit
        else:
            return reg_nonpar_fit(
                Y, X, var_types=var_types, n_sub=n_sub, n_res=n_res, verbose=verbose
            )
    elif mode == "SPL":
        if X.ndim > 1:
            bs_error_abort("with a spline, only works in one dimension")
        return spline_reg(Y, X)
    else:
        try:
            imode = int(mode)
        except TypeError:
            bs_error_abort(f"does not accept mode={mode}")
        preg, _, _ = proj_Z(Y, X, p=imode, verbose=verbose)
        return preg


def bs_multivariate_normal_pdf(
    values_x: np.ndarray, means_x: float | np.ndarray, cov_mat: float | np.ndarray
) -> np.ndarray:
    """
    Multivariate (or univariate) normal probability density function at values_x

    Args:
        values_x: values at which to evaluate the pdf, an `n`-vector or an `(n, nvars)` matrix
        means_x: means of the multivariate normal, a float or an `(nvars)` vector
        cov_mat: covariance matrix of the multivariate normal, a float or an `(nvars, nvars)` matrix

    Returns:
        the values of the density at `values_x`
    """
    ndims_values = check_vector_or_matrix(values_x, "bs_multivariate_normal_pdf")
    if ndims_values == 1:  # we are evaluating a univariate normal
        # if not type(means_x) == float:
        #     bs_error_abort(f"means_x should be a float as values_x is a vector")
        # if not type(cov_mat) == float:
        #     bs_error_abort(f"cov_mat should be a float as values_x is a vector")
        sigma2 = cov_mat
        resid = values_x - means_x
        dval = np.exp(-0.5 * resid * resid / sigma2) / np.sqrt(2 * np.pi * sigma2)
        return cast(np.ndarray, dval)
    else:  # we are evaluating a multivariate normal
        n, nvars = values_x.shape
        n_means = check_vector(means_x, "bs_multivariate_normal_pdf")
        if n_means != nvars:
            bs_error_abort(f"means_x should be a vector of size {nvars} not {n_means}")
        nrows, ncols = check_matrix(cov_mat, "bs_multivariate_normal_pdf")
        if nrows != ncols or nrows != nvars:
            bs_error_abort(
                f"cov_mat should be a matrix ({nvars}, {nvars}) not ({nrows}, {ncols})"
            )
        resid = values_x - means_x
        argresid = spla.solve(cov_mat, resid.T)
        argexp = np.zeros(n)
        for i in range(n):
            argexp[i] = np.dot(resid[i, :], argresid[:, i])
        dval = np.exp(-0.5 * argexp) / np.sqrt(
            ((2 * np.pi) ** nvars) * spla.det(cov_mat)
        )
        return cast(np.ndarray, dval)


def estimate_pdf(
    x_obs: np.ndarray,
    x_points: np.ndarray,
    MIN_SIZE_NONPAR: int = 200,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    return an estimate of the conditional densities of `x` at points `values_x` (Silverman rule)

    Args:
        x_obs: an `n`-vector or an `(n, nvars)` matrix of the observed values of `x`
        x_points: an `m`-vector or an `(m, nvars)` matrix of x values
        MIN_SIZE_NONPAR: minimum size above which we use kernel density estimators
        weights: an `n`-vector of weights for the observations, if present

    Returns:
        the density estimates at `values_x`
    """
    ndims_x = check_vector_or_matrix(x_obs, "estimate_pdf")
    ndims_valx = check_vector_or_matrix(x_points, "estimate_pdf")

    if ndims_x == 1:
        n_obs = x_obs.size
        if ndims_valx != 1:
            bs_error_abort(f"x_points should have one dimension, not {ndims_valx}")
        xt_obs = x_obs.reshape((-1, 1))
        nvars = 1
        xt_points = x_points.reshape((-1, 1))
    else:  # ndims_x == 2
        n_obs, nvars = x_obs.shape
        if ndims_valx == 1:  # only one x point with  nv elements
            nv = x_points.size
        else:  # several x points with  nv elements
            nv = x_points.shape[1]
        if nv != nvars:
            bs_error_abort(f"x_points should have {nvars} variables, not {nv}")
        xt_obs = x_obs
        xt_points = x_points

    if weights is not None:
        n_w = check_vector(weights, "estimate_pdf")
        if n_w != n_obs:
            bs_error_abort(
                f"if weights is given, it should be a vector of size {n_obs} not {n_w}"
            )

    min_size_np = MIN_SIZE_NONPAR ** ((4.0 + nvars) / 5.0)

    if n_obs > min_size_np:  # cell large enough to use nonparametrics
        # fit joint density of x
        kde = sts.gaussian_kde(xt_obs.T, bw_method="silverman", weights=weights)
        # density of x at values_x
        f_x = kde.evaluate(xt_points.T)
    else:
        # sample too small, we fit a normal
        if ndims_x == 1:  # univariate
            mean_x = np.mean(x_obs)
            var_x = np.var(x_obs)
            f_x = bs_multivariate_normal_pdf(x_points, mean_x, var_x)
        else:  # multivariate
            means_x = np.mean(x_obs, 0)
            cov_mat = np.cov(x_obs.T)
            f_x = bs_multivariate_normal_pdf(x_points, means_x, cov_mat)
        if weights is not None:
            f_x *= weights / np.mean(weights)
    return cast(np.ndarray, f_x)


def estimate_densities_at_quantiles(
    X: np.ndarray, qtiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    estimate densities of margins at prespecified quantiles (Silverman rule)
    and the joint density at each vector of these quantiles

    Args:
        X: `n`-vector or `(n, nx)`-matrix
        qtiles: vector of `nq` numbers between 0 and 1

    Returns: 
        if `X` is a matrix, the `({nq}^{nx}, {nx})` matrices of estimated margin densities \
     and the `{nq}^{nx}` vector of the joint density on the lexicographic grid of quantiles;
        if `X` is a vector, the `nq`-vector of the density at the quantiles, twice
    """
    ndims_X = check_vector_or_matrix(X, "estimate_densities_")
    if ndims_X == 1:
        f_X = estimate_pdf(X, np.quantile(X, qtiles))
        return f_X, f_X
    else:
        nx = X.shape[1]
        nq = qtiles.size
        f_X_k = np.zeros((nq, nx))
        nodes_mat = np.zeros((nq, nx))
        for i_x in range(nx):
            X_ix = X[:, i_x]
            nodes_mat[:, i_x] = np.quantile(X_ix, qtiles)
            f_X_k[:, i_x] = estimate_pdf(X_ix, nodes_mat[:, i_x])
        f_margins = make_lexico_grid(f_X_k)
        values_X = make_lexico_grid(nodes_mat)
        f_X = estimate_pdf(X, values_X)  # joint density
        return f_margins, f_X, values_X
