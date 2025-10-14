import numpy as np
import pytest

from bs_python_utils.bivariate_quantiles import (
    _compute_ab,
    _compute_u2_bounds,
    bivariate_quantiles,
    bivariate_quantiles_v,
    bivariate_ranks,
    bivariate_ranks_v,
    solve_for_v_,
)
from bs_python_utils.bsnputils import bsgrid, ecdf, npmaxabs


def test_bivariate_quantiles_recompute_matches():
    rng = np.random.default_rng(123)
    y = rng.normal(size=(64, 2))
    nodes = 16

    weights = solve_for_v_(y, n_nodes=nodes, verbose=False)

    q = np.linspace(0.0, 1.0, 5)
    u_points = bsgrid(q, q)

    direct = bivariate_quantiles_v(y, u_points, weights)
    via_solver = bivariate_quantiles(y, u_points, n_nodes=nodes, verbose=False)

    ranks_direct = bivariate_ranks_v(y, weights, n_nodes=nodes)
    ranks_via_solver = bivariate_ranks(y, n_nodes=nodes, verbose=False)

    assert np.allclose(direct, via_solver)
    assert np.allclose(ranks_direct, ranks_via_solver)
    assert (
        npmaxabs(ranks_direct - np.column_stack((ecdf(y[:, 0]), ecdf(y[:, 1])))) < 0.2
    )


def test_chunked_quantiles_matches_numpy_argmax():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(10, 2))
    v = rng.normal(size=10)
    u = rng.uniform(0.0, 1.0, size=(6000, 2))

    chunked = bivariate_quantiles_v(y, u, v)
    expected = y[np.argmax(u @ y.T - v, axis=1)]

    assert np.allclose(chunked, expected)


def test_compute_ab_handles_ties():
    y = np.array([[0.0, 1.0], [1.0, 1.0]])
    v = np.array([0.2, -0.2])
    a_mat, b_mat = _compute_ab(y, v)

    assert np.isfinite(a_mat).all()
    assert np.isfinite(b_mat).all()


@pytest.mark.parametrize("k", [0, 1])
def test_compute_u2_bounds_degenerate(k):
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    v = np.array([0.1, -0.1])
    a_mat, b_mat = _compute_ab(y, v)
    nodes = np.linspace(0.0, 1.0, 5)

    left, right = _compute_u2_bounds(k, nodes, a_mat, b_mat)

    assert np.isfinite(left).all()
    assert np.isfinite(right).all()
    assert (right >= left).all()


def test_bivariate_ranks_zero_mass_returns_nan():
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    v = np.array([0.1, -0.1])
    ranks = bivariate_ranks_v(y, v, n_nodes=8, presorted=True)

    assert np.isnan(ranks[0]).all()
    assert np.isfinite(ranks[1]).all()
