import numpy as np

import matplotlib.pyplot as plt

from bs_python_utils import sklearn_utils


def test_skl_npreg_lasso_returns_predictions_and_model():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)

    preds, model = sklearn_utils.skl_npreg_lasso(
        y,
        X,
        alpha=0.05,
        degree=3,
        include_bias=True,
        return_model=True,
        lasso_kwargs={"fit_intercept": False},
    )

    assert preds.shape == (20,)
    assert np.isfinite(preds).all()

    assert "scaler" in model.named_steps
    assert "poly" in model.named_steps
    assert "lasso" in model.named_steps
    assert model.named_steps["lasso"].fit_intercept is False

    # ensure pipeline can predict out-of-sample without errors
    preds_new = model.predict(X[:5])
    assert preds_new.shape == (5,)


def test_plot_lasso_path_returns_values_and_plots():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.linspace(-1.0, 1.0, 6)

    fig, ax = plt.subplots()
    alphas, coefs, returned_ax = sklearn_utils.plot_lasso_path(
        y, X, eps=1e-2, standardize=True, ax=ax
    )

    try:
        assert returned_ax is ax
        assert np.all(alphas > 0)
        assert coefs.shape[1] == len(alphas)
        # one line per feature
        assert len(ax.get_lines()) == X.shape[1]
    finally:
        plt.close(fig)
