"""
Contains Lasso `scikit-learn` utility programs:

* `skl_npreg_lasso`: Lasso regression on polynomial interactions of the covariates
* `plot_lasso_path`: plots the Lasso coefficient paths.
"""

from itertools import cycle
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.linear_model import Lasso, lasso_path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def skl_npreg_lasso(
    y: np.ndarray,
    X: np.ndarray,
    alpha: float,
    degree: int = 4,
    *,
    include_bias: bool = False,
    return_model: bool = False,
    lasso_kwargs: dict[str, Any] | None = None,
) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    """Fit a polynomial Lasso regression with standard preprocessing.

    Inputs are reshaped if necessary and passed through a ``StandardScaler`` followed by
    ``PolynomialFeatures`` and ``Lasso``. Extra keyword arguments can be forwarded to the Lasso
    estimator, and the fitted pipeline can optionally be returned.

    Args:
        y: Response vector of shape ``(n_obs,)`` (flattened automatically if needed).
        X: Feature matrix of shape ``(n_obs, n_features)``; 1-D inputs are reshaped.
        alpha: Lasso penalty parameter.
        degree: Total polynomial degree for ``PolynomialFeatures``.
        include_bias: When ``True``, keep the bias column in the polynomial design matrix.
        return_model: If ``True``, also return the fitted scikit-learn ``Pipeline``.
        lasso_kwargs: Extra keyword arguments forwarded to ``sklearn.linear_model.Lasso``.

    Returns:
        Either the fitted values ``E[y | X]`` (shape ``(n_obs,)``) or a tuple
        ``(fitted, pipeline)`` when ``return_model`` is ``True``.
    """
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    extra_kwargs = {} if lasso_kwargs is None else dict(lasso_kwargs)
    extra_kwargs.setdefault("max_iter", 10_000)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            (
                "lasso",
                Lasso(alpha=alpha, **extra_kwargs),
            ),
        ]
    )
    model.fit(X, y)
    fitted = cast(np.ndarray, model.predict(X))
    if return_model:
        return fitted, model
    return fitted


def plot_lasso_path(
    y: np.ndarray,
    X: np.ndarray,
    eps: float = 1e-3,
    *,
    standardize: bool = True,
    ax: Axes | None = None,
) -> tuple[np.ndarray, np.ndarray, Axes]:
    """Compute and plot the Lasso regularization path.

    Args:
        y: Response vector of shape ``(n_obs,)`` (flattened automatically if necessary).
        X: Feature matrix of shape ``(n_obs, n_features)``; 1-D inputs are reshaped.
        eps: Path length parameter passed to ``sklearn.linear_model.lasso_path``.
        standardize: When ``True`` (default) the predictors are standardized and the response
            is mean-centered before computing the path.
        ax: Optional Matplotlib ``Axes`` on which to draw; when ``None`` a new axes is created.

    Returns:
        A tuple ``(alphas, coefficients, axes)`` where ``alphas`` are the regularization
        strengths, ``coefficients`` has shape ``(n_features, n_alphas)``, and ``axes`` is the
        Matplotlib axes used for plotting.
    """
    if y.ndim != 1:
        y = y.reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)
        y_proc = y - np.mean(y)
    else:
        X_proc, y_proc = X, y

    # Compute paths
    alphas_lasso, coefs_lasso, _ = lasso_path(X_proc, y_proc, eps=eps)
    neg_log_alphas_lasso = -np.log10(alphas_lasso)

    if ax is None:
        _, ax = plt.subplots()

    colors = cycle(["b", "r", "g", "c", "k", "m", "y"])
    for coef_l, c in zip(coefs_lasso, colors):
        ax.plot(neg_log_alphas_lasso, coef_l, c=c)

    ax.set_xlabel("-Log(alpha)")
    ax.set_ylabel("coefficients")
    ax.set_title("Lasso Paths")
    ax.axis("tight")

    return alphas_lasso, coefs_lasso, ax
