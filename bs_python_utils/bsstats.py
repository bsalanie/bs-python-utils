"""
Contains some statistical routines.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Callable, cast

import numpy as np
import scipy.linalg as spla
import scipy.stats as sts
from emcee import EnsembleSampler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric._kernel_base import EstimatorSettings
from statsmodels.nonparametric.kernel_regression import KernelReg

from bs_python_utils.bsnputils import check_matrix, check_vector, check_vector_or_matrix
from bs_python_utils.bssputils import spline_reg
from bs_python_utils.bsutils import bs_error_abort


@dataclass
class TslsResults:
    """contains the full results of a TSLS regression"""

    iv_estimates: float | np.ndarray | None
    r2_first_iv: float | np.ndarray | None
    r2_y: float | None
    r2_second: float | np.ndarray | None
    y_proj: float | np.ndarray | None
    y_coeffs: float | np.ndarray | None
    X_IV_proj: float | np.ndarray | None
    b_proj_IV: float | np.ndarray | None


def _powers_Z(Z: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """used internally by `proj_Z`; returns $\\prod_{k=1}^m  Z_{\\cdot k}^{l_k}$.

    Args:
        Z: a matrix `(n, m)`
        list_ints: a list of integers

    Returns:
        the product of the powers of `Z`.
    """
    if Z.ndim != 2:
        bs_error_abort(f"Z should have dimension 2, not {Z.ndim}")
    m = Z.shape[1]
    mdegs = check_vector(degrees)
    if mdegs != m:
        bs_error_abort("The size of degrees should equal the number of columns of Z.")
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
            degrees[i, :] = np.array([x.count(i) for x in listq])
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
    """project `W` on all interactions of degree `p` or less of `Z`

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
    """TSLS of `y` on `X` with instruments `Z`

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
    """nonparametric regression of y on the columns of X;
    the bandwidth is chosen on a subsample of size `nsub` if `nsub` < `nobs`, and rescaled.

    Args:
        y: a vector of size nobs
        X: a (nobs) vector or a matrix of shape (nobs, m)
        var_types: specify types of all `X` variables if not all of them are continuous; one character per variable

            * 'c' for continuous
            * 'u' discrete unordered
            * 'o' discrete ordered.
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
    """nonparametric regression of y on the columns of X; the bandwidth is chosen on a subsample of size `nsub` if `nsub` < `nobs`, and rescaled.

    Args:
        y: a vector of size nobs
        X: a (nobs) vector or a matrix of shape (nobs, m)
        var_types: specify types of all `X` variables if not all of them are continuous; one character per variable

            * 'c' for continuous
            * 'u' discrete unordered
            * 'o' discrete ordered.
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
    """flexible regression  of `Y` on `X`

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
        except (TypeError, ValueError):
            bs_error_abort(f"does not accept mode={mode}")
        preg, _, _ = proj_Z(Y, X, p=imode, verbose=verbose)
        return preg


def bs_multivariate_normal_pdf(
    values_x: np.ndarray, means_x: float | np.ndarray, cov_mat: float | np.ndarray
) -> np.ndarray:
    """Multivariate (or univariate) normal probability density function at values_x

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
    """Estimate the (possibly weighted) density of ``x`` at specified points.

    Observations are treated as multivariate when ``x_obs`` has two dimensions. For sufficiently
    large samples a Gaussian KDE with Silverman bandwidth is used; otherwise a normal
    approximation (with weighted mean/covariance when ``weights`` are supplied) is returned.

    Args:
        x_obs: Observed data, either ``(n,)`` or ``(n, nvars)``.
        x_points: Evaluation points, either ``(m,)`` or ``(m, nvars)``.
        MIN_SIZE_NONPAR: Sample-size threshold controlling the nonparametric vs. Gaussian fallback.
        weights: Optional observation weights (length ``n``).

    Returns:
        Array of density estimates at ``x_points``.
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
            if weights is None:
                mean_x = np.mean(x_obs)
                var_x = np.var(x_obs)
            else:
                mean_x = np.average(x_obs, weights=weights)
                var_x = np.average((x_obs - mean_x) ** 2, weights=weights)
            f_x = bs_multivariate_normal_pdf(x_points, mean_x, var_x)
        else:  # multivariate
            if weights is None:
                means_x = np.mean(x_obs, 0)
                cov_mat = np.cov(x_obs.T)
            else:
                weights = weights / np.sum(weights)
                means_x = np.average(x_obs, axis=0, weights=weights)
                centered = x_obs - means_x
                cov_mat = centered.T @ (centered * weights[:, None])
            f_x = bs_multivariate_normal_pdf(x_points, means_x, cov_mat)
    return cast(np.ndarray, f_x)


def emcee_draw(
    n_samples: int,
    log_pdf: Callable[[np.ndarray, list], float],
    p0: np.ndarray,
    params: list | None = None,
    n_burn_in: int | None = 100,
    seed: int | None = 8754,
) -> np.ndarray:
    """Draw samples from a target log-density using ``emcee``.

    Args:
        n_samples: Number of samples to keep after burn-in.
        log_pdf: Callable returning the log-density; invoked as ``log_pdf(theta, *params)``.
        p0: Initial walker states of shape ``(n_walkers, n_dims)``.
        params: Optional list of extra parameters forwarded to ``log_pdf``.
        n_burn_in: Number of burn-in iterations prior to collecting samples.
        seed: Seed for the final random subsampling step.

    Returns:
        An ``(n_samples, n_dims)`` array of draws selected (without replacement when possible)
        from the flattened ``emcee`` chain.

    Note:
        ``log_pdf`` should return ``-np.inf`` outside the support, and ``p0`` should lie within
        that support to ensure the sampler mixes correctly.
    """
    n_walkers, n_dims = p0.shape
    # burn in
    sampler = EnsembleSampler(n_walkers, n_dims, log_pdf, args=tuple(params or ()))
    state = sampler.run_mcmc(p0, n_burn_in)
    sampler.reset()
    # generate the samples
    sampler.run_mcmc(state, n_samples)
    samples = sampler.get_chain(flat=True)
    samples = samples.reshape(-1, n_dims)
    rng = np.random.default_rng(seed)
    replace = samples.shape[0] < n_samples
    samples = rng.choice(samples, size=n_samples, replace=replace, axis=0)
    return cast(np.ndarray, samples)


def kde_resample(
    data: np.ndarray, n_samples: int, n_bw: int = 10
) -> tuple[np.ndarray, float]:
    """Fit a nonparametric density to data by KDE, with cross-validation with starting point at rule of thumb,
    and generate samples from the estimated density.

    Args:
        data: an `(n_obs, n_dims)` matrix of data
        n_samples: how many iid draws we want
        n_bw: how mamy bandwidths we try from 1/10th to 10 times the rule-of-thumb

    Returns:
        an `(n_samples, n_dims)` matrix of draws, and the bandwidth used.
    """
    n_obs, n_dims = check_matrix(data)
    spreads = np.std(data, axis=0, ddof=1)
    h_scale = float(np.mean(spreads))
    if not np.isfinite(h_scale) or h_scale <= 0.0:
        h_scale = 1.0
    h_rot = h_scale * (n_obs ** (-1 / (n_dims + 4)))
    params = {"bandwidth": h_rot * np.logspace(-1, 1, n_bw)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    kde = grid.best_estimator_
    best_bandwidth = grid.best_params_["bandwidth"]
    resampled_data = kde.sample(n_samples)
    return resampled_data, best_bandwidth
