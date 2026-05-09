import numpy as np

from bs_python_utils.stats.bsstats import emcee_draw, kde_resample


def log_gauss2(x, *args):
    dx0 = x[0] - args[0]
    dx1 = x[1] - args[1]
    v0, v1 = args[2:]
    return -0.5 * np.sum(dx0 * dx0 / v0 + dx1 * dx1 / v1)


def test_emcee_draw():
    val_means = [0.3, -0.7]
    val_vars = [2.0, 0.5]
    rng = np.random.default_rng(seed=19674)
    p0 = rng.normal(size=(100, 2))
    samples = emcee_draw(10_000, log_gauss2, p0, params=[*val_means, *val_vars])
    print(
        f"samples have means {np.mean(samples, 0)} and variances {np.var(samples, 0)}"
    )
    assert np.allclose(np.mean(samples, 0), np.array(val_means), atol=0.1)
    assert np.allclose(np.var(samples, 0), np.array(val_vars), atol=0.1)


def test_kde_resample():
    n_obs, n_dims = 10_000, 2
    rng = np.random.default_rng(seed=45678)
    x = rng.normal(size=(n_obs, n_dims))
    samples, _ = kde_resample(x, 10_000)
    assert np.allclose(np.mean(samples, 0), np.mean(x, 0), atol=0.1)
    assert np.allclose(np.var(samples, 0), np.var(x, 0), atol=0.15)
