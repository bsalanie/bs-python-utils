import numpy as np

from bs_python_utils.numerical.bsnputils import npmaxabs
from bs_python_utils.stats.bivariate_quantiles import (
    _compute_ad,
    _solve_for_v,
    bivariate_ranks_simul,
)


def test_bivariate_ranks():
    rng = np.random.default_rng(123)
    n = 500
    y = rng.normal(size=(n, 2))
    nodes = 32

    _, bivranks = _solve_for_v(y, n_nodes=nodes, verbose=False)

    bivranks_th = np.empty_like(bivranks)
    for k in range(n):
        bivranks_th[k, 0] = np.mean(y[:, 0] < y[k, 0]) + 0.5 * np.mean(
            y[:, 0] == y[k, 0]
        )
        bivranks_th[k, 1] = np.mean(y[:, 1] < y[k, 1]) + 0.5 * np.mean(
            y[:, 1] == y[k, 1]
        )

    assert npmaxabs(bivranks - bivranks_th) < 0.1


def test_bivariate_ranks_simul():
    rng = np.random.default_rng(123)
    n = 500
    y = rng.normal(size=(n, 2))

    bivranks = bivariate_ranks_simul(y, rng)

    bivranks_th = np.empty_like(bivranks)
    for k in range(n):
        bivranks_th[k, 0] = np.mean(y[:, 0] < y[k, 0]) + 0.5 * np.mean(
            y[:, 0] == y[k, 0]
        )
        bivranks_th[k, 1] = np.mean(y[:, 1] < y[k, 1]) + 0.5 * np.mean(
            y[:, 1] == y[k, 1]
        )

    assert npmaxabs(bivranks - bivranks_th) < 0.1


def test_compute_ad():
    y = np.array([[0.0, 1.0], [1.0, 1.3], [-1.0, 3.0]])
    a_mat, dy2 = _compute_ad(y)

    a_mat_th = np.array(
        [[0.0, -1.0 / 0.3, 0.5], [-1.0 / 0.3, 0.0, 2.0 / 1.7], [0.5, 2.0 / 1.7, 0.0]]
    )
    dy2_th = np.array([[1.0, 0.3, 2.0], [-0.3, 1.0, 1.7], [-2.0, -1.7, 1.0]])

    assert np.allclose(a_mat, a_mat_th)
    assert np.allclose(dy2, dy2_th)
