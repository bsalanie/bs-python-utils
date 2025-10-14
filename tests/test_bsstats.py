import numpy as np

from bs_python_utils import bsstats


def test_proj_Z_multivariate_degree_two():
    rng = np.random.default_rng(123)
    Z = rng.normal(size=(50, 2))
    W = 1.0 + 0.5 * Z[:, 0] - 0.3 * Z[:, 1] + 0.2 * Z[:, 0] * Z[:, 1]

    W_proj, coeffs, r2 = bsstats.proj_Z(W, Z, p=2)

    assert W_proj.shape == W.shape
    assert coeffs.shape[0] > 0
    assert np.all(np.isfinite(W_proj))
    assert r2 > 0.0


def test_estimate_pdf_weighted_fallback_univariate():
    x_obs = np.array([0.0, 1.0, 3.0, 4.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    x_points = np.array([2.0])

    est = bsstats.estimate_pdf(x_obs, x_points, MIN_SIZE_NONPAR=500, weights=weights)

    mean = np.average(x_obs, weights=weights)
    var = np.average((x_obs - mean) ** 2, weights=weights)
    expected = np.exp(-0.5 * (x_points - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
    np.testing.assert_allclose(est, expected)


def test_flexible_reg_accepts_string_mode():
    X = np.linspace(-1.0, 1.0, 20)
    Y = 2.0 + 3.0 * X
    fitted = bsstats.flexible_reg(Y, X, mode="1")

    np.testing.assert_allclose(fitted, Y, rtol=1e-5, atol=1e-5)


def test_emcee_draw_without_params():
    def log_pdf(theta, params=()):
        return -0.5 * np.dot(theta, theta)

    p0 = np.random.randn(4, 2)
    samples = bsstats.emcee_draw(5, log_pdf, p0, params=None)

    assert samples.shape == (5, 2)


def test_kde_resample_returns_positive_bandwidth():
    rng = np.random.default_rng(321)
    data = rng.normal(size=(200, 2))
    samples, bandwidth = bsstats.kde_resample(data, n_samples=10, n_bw=3)

    assert samples.shape == (10, 2)
    assert bandwidth > 0.0
