"""
distance covariance and partial distance covariance (Szekely and Rizzo)
evaluation and tests of independence and conditional independence
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np

from bs_python_utils.bsnputils import test_square, test_vector_or_matrix
from bs_python_utils.bsutils import bs_error_abort


@dataclass
class DcovResults:
    dcov: float
    dcov_stat: float
    dcor: float
    X_dd: np.ndarray
    Y_dd: np.ndarray
    unbiased: bool


@dataclass
class PdcovResults:
    pdcov: float
    pdcov_stat: float
    pdcor: float
    X_dd: np.ndarray
    Y_dd: np.ndarray
    Z_dd: np.ndarray


def _compute_distances(T: np.ndarray) -> np.ndarray:
    """
    compute the Euclidian norms (or absolute values)
    of all row differences `T_k - T_l`

    Args:
        T: a vector or a matrix

    Returns:
        the matrix of norms of differences
    """
    ndims_T = test_vector_or_matrix(T, "_compute_distances")
    if ndims_T == 1:
        return np.abs(np.subtract.outer(T, T))
    else:
        n, nv = T.shape
        A = np.zeros((n, n))
        for iv in range(nv):
            Tiv = T[:, iv]
            Aiv = np.subtract.outer(Tiv, Tiv)
            A += Aiv * Aiv
        return np.sqrt(A)


def _double_decenter(A: np.ndarray, unbiased: Optional[bool] = False) -> np.ndarray:
    """
    does double decentering on a square matrix A

    Args:
        A: a matrix
        unbiased: if `True`, we use the Szekely and Rizzo 2014 formula

    Returns:
        the doubly decentered matrix
    """
    n = test_square(A, "_double_decenter")
    A_1 = np.sum(A, 0)
    A_2 = np.sum(A, 1).reshape((-1, 1))
    A_0 = np.sum(A_1)
    fac2 = (n - 2) if unbiased else n
    fac1 = (n - 1) if unbiased else n
    A_dd = A - A_1 / fac2 - A_2 / fac2 + A_0 / (fac1 * fac2)
    if unbiased:
        np.fill_diagonal(A_dd, np.zeros(n))
    return A_dd


def _dcov_prod(A: np.ndarray, B: np.ndarray, unbiased: Optional[bool] = False) -> float:
    n = test_square(A, "_dcov_prod")
    m = test_square(B, "_dcov_prod")
    if m == n:
        fac3 = (n - 3) if unbiased else n
        return np.sum(A * B) / (n * fac3)
    else:
        bs_error_abort(f"A and B should be square matrices of the same size")


def dcov_dcor(
    X: np.ndarray, Y: np.ndarray, unbiased: Optional[bool] = False
) -> DcovResults:
    """
    evaluate the distance covariance and correlation of `X` and `Y`

    Args:
        X: `n` observations of a random variable or vector
        Y: `n` observations of a random variable or vector
        unbiased: if `True`, we use the Szekely and Rizzo 2014 formula

    Returns:
        `dCov^2(X,Y)` and `dCor^2(X,Y)`
    """
    X_dist = _compute_distances(X)
    n = X_dist.shape[0]
    X_dd = _double_decenter(X_dist, unbiased)
    Y_dist = _compute_distances(Y)
    Y_dd = _double_decenter(Y_dist, unbiased)
    dcov2 = _dcov_prod(X_dd, Y_dd, unbiased)
    dcor2 = dcov2 / sqrt(
        _dcov_prod(X_dd, X_dd, unbiased) * _dcov_prod(Y_dd, Y_dd, unbiased)
    )
    return DcovResults(
        dcov=dcov2,
        dcor=dcor2,
        X_dd=X_dd,
        Y_dd=Y_dd,
        unbiased=unbiased,
        dcov_stat=n * dcov2,
    )


def _dcov_bootstrap(
    X_dd: np.ndarray,
    Y_dd: np.ndarray,
    unbiased: Optional[bool] = False,
    ndraws: Optional[int] = 199,
) -> np.ndarray:
    """
    use bootstrap on the test statistics of independence

    Args:
        X_dd: the doubly decentered distances for `X`
        Y_dd: the doubly decentered distances for `Y`
        unbiased:  if `True`, we use the Szekely and Rizzo 2014 formula
        ndraws: number of permutations

    Returns:
        the values of the `ndraws` bootstrapped test stats
    """
    n = X_dd.shape[0]
    dcov_stats_boot = np.zeros(ndraws)
    for idraw in range(ndraws):
        draws = np.random.choice(np.arange(n), n)
        X_ddi = X_dd[draws, :][:, draws]
        Y_ddi = Y_dd[draws, :][:, draws]
        if idraw % 50 == 0:
            print(f"    bootstrap draw {idraw}")
        dcov_stats_boot[idraw] = _dcov_prod(X_ddi, Y_ddi, unbiased)
    return n * dcov_stats_boot


def pvalue_dcov(dcov_results: DcovResults, ndraws: Optional[int] = 199) -> float:
    """
    test of no dependence between `X` and `Y` given `Z`

    Args:
        dcov_results:  results from `dcov_dcor`
        ndraws: the number of draws we use

    Returns:
        the bootstrapped  p-value of the test
    """
    X_dd = dcov_results.X_dd
    Y_dd = dcov_results.Y_dd
    dcov_stat = dcov_results.dcov_stat
    unbiased = dcov_results.unbiased
    dcov_stats_boot = _dcov_bootstrap(X_dd, Y_dd, unbiased, ndraws)
    return (1.0 + np.sum(dcov_stat < dcov_stats_boot)) / (1.0 + ndraws)


def pdcov_pdcor(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> PdcovResults:
    """
    evaluate the partial distance covariance and correlation of `X` and `Y` given `Z`

    Args:
        X: `n` observations of a random variable or vector
        Y: `n` observations of a random variable or vector
        Z: `n` observations of a random variable or vector
        unbiased: if `True`, we use the Szekely and Rizzo 2014 formula

    Returns:
        a `PdcovResults` instance
    """
    unbiased = True
    X_dist = _compute_distances(X)
    X_dd = _double_decenter(X_dist, unbiased)
    Y_dist = _compute_distances(Y)
    Y_dd = _double_decenter(Y_dist, unbiased)
    Z_dist = _compute_distances(Z)
    Z_dd = _double_decenter(Z_dist, unbiased)
    C_XX = _dcov_prod(X_dd, X_dd, unbiased)
    C_XY = _dcov_prod(X_dd, Y_dd, unbiased)
    C_YY = _dcov_prod(Y_dd, Y_dd, unbiased)
    C_XZ = _dcov_prod(X_dd, Z_dd, unbiased)
    C_YZ = _dcov_prod(Y_dd, Z_dd, unbiased)
    C_ZZ = _dcov_prod(Z_dd, Z_dd, unbiased)
    pdcov = C_XY - (C_XZ * C_YZ) / C_ZZ
    pdcor = pdcov / sqrt((C_XX - C_XZ * C_XZ / C_ZZ) * (C_YY - C_YZ * C_YZ / C_ZZ))
    n = X.shape[0]
    return PdcovResults(
        pdcov=pdcov, pdcor=pdcor, pdcov_stat=n * pdcov, X_dd=X_dd, Y_dd=Y_dd, Z_dd=Z_dd
    )


def _pdcovs_bootstrap(
    X_dd: np.ndarray, Y_dd: np.ndarray, Z_dd: np.ndarray, ndraws: Optional[int] = 199
) -> np.ndarray:
    """
    use permutations and recompute the test statistics of independence

    Args:
        X_dd: the doubly decentered distances for `X`
        Y_dd: the doubly decentered distances for `Y`
        Z_dd: the doubly decentered distances for `Y`
        ndraws: the number of draws we use

    Returns:
        the `ndraws` values of `pdCov(X,Y ; Z)`
    """
    pdcov_stats_boot = np.zeros(ndraws)
    unbiased = True
    n = X_dd.shape[0]
    for idraw in range(ndraws):
        if idraw % 50 == 0:
            print(f"pdcov test: bootstrap draw {idraw}")
        draws = np.random.choice(np.arange(n), n)
        X_ddi = X_dd[draws, :][:, draws]
        Y_ddi = Y_dd[draws, :][:, draws]
        Z_ddi = Z_dd[draws, :][:, draws]
        C_XY = _dcov_prod(X_ddi, Y_ddi, unbiased)
        C_XZ = _dcov_prod(X_ddi, Z_ddi, unbiased)
        C_YZ = _dcov_prod(Y_ddi, Z_ddi, unbiased)
        C_ZZ = _dcov_prod(Z_ddi, Z_ddi, unbiased)
        pdcov_stats_boot[idraw] = C_XY - (C_XZ * C_YZ) / C_ZZ
    return n * pdcov_stats_boot


def pvalue_pdcov(pdcov_results: PdcovResults, ndraws: Optional[int] = 199) -> float:
    """
    test of no dependence between `X` and `Y` given `Z`

    Args:
        pdcov_results: the results of `pdcov_pdcor`
        ndraws: the number of draws we use

    Returns:
        the bootstrapped  p-value of the test
    """
    X_dd = pdcov_results.X_dd
    Y_dd = pdcov_results.Y_dd
    Z_dd = pdcov_results.Z_dd
    pdcov_stat = pdcov_results.pdcov_stat
    pdcov_stats_boot = _pdcovs_bootstrap(X_dd, Y_dd, Z_dd, ndraws)
    return (1.0 + np.sum(pdcov_stat < pdcov_stats_boot)) / (1.0 + ndraws)


if __name__ == "__main__":
    ## test partial distance covariance
    # example page 2396 of Szekely and Rizzo 2014
    n = 2000
    do_bootstrap = False
    Z1 = np.random.normal(size=n)
    Z2 = np.random.normal(size=n)
    Z3 = np.random.normal(size=n)
    X = Z1 + Z3
    Y = Z2 + Z3
    Z = Z3
    print("\n\n     Test of page 2396")
    dcov_XY = dcov_dcor(X, Y)
    dcov_XZ = dcov_dcor(X, Z)
    dcov_YZ = dcov_dcor(Y, Z)
    pdcov_XYZ = pdcov_pdcor(X, Y, Z)
    print(f"         dCor(X, Y)={dcov_XY.dcor}, should be 0.2062 in large samples")
    print(f"         dCor(X, Z)={dcov_XZ.dcor}, should be 0.4319 in large samples")
    print(f"         dCor(Y, Z)={dcov_YZ.dcor}, should be 0.4319 in large samples")
    print(
        f"         pdCor(X, Y; Z)={pdcov_XYZ.pdcor}, should be 0.0242 in large samples"
    )

    if do_bootstrap:
        ndraws = 499
        pval = pvalue_pdcov(pdcov_XYZ)
        print(
            f"\n\n test stat={pdcov_XYZ.pdcov_stat: >.2f} has p-value {pval} for"
            f" {ndraws} draws"
        )