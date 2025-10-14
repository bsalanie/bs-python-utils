"""
A Matplotlib utility program:

* `ax_text`: annotate an `ax` with text.
* `bs_mpl_plot_dcm_fit`: generates a boxplot of the predicted probas for a
  discrete choice model
"""

import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np

from bs_python_utils.bsutils import bs_error_abort


def ax_text(ax: axes.Axes, str_txt: str, x: float, y: float) -> axes.Axes:
    """
    Annotate an `ax` with text in Matplotlib.

    Args:
        ax: the axis we want to annotate
        str_txt: a string of text
        x: position in fraction of horizontal axis
        y: position in fraction of vertical axis

    Returns:
        the nnotated `ax`.
    """
    if not (isinstance(x, float) and 0 <= x <= 1):
        bs_error_abort("x should be a number between 0.0 and 1.0")
    if not (isinstance(y, float) and 0 <= y <= 1):
        bs_error_abort("y should be a number between 0.0 and 1.0")
    ax.text(
        x,
        y,
        str_txt,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    return ax


def bs_mpl_plot_dcm_fit(
    y_true: np.ndarray,
    probhat: np.ndarray,
    max_cols: int = 4,
    save_to: str | None = None,
) -> None:
    """generates a boxplot of the predicted probas for a discrete choice model
       by the actual value of the variable $y$.

    Args:
        y_true: an `(nobs)` vector with the true value of $y$
        probhat: the predicted probas for the values of $y$
        max_cols: the maximum number of columns in the plot. Defaults to 4.
        save_to: (maybe) where we save the plot, with `.png` extension.

    Returns:
        nothing.
    """
    n_vals_y = np.unique(y_true).size
    y_means = np.zeros(n_vals_y)
    for k in range(n_vals_y):
        y_means[k] = np.mean(probhat[:, k])

    if n_vals_y > max_cols:
        n_rows = (n_vals_y - 1) // max_cols + 1
        n_cols = max_cols
    else:
        n_rows = 1
        n_cols = n_vals_y

    fig, ax = plt.subplots(n_rows, n_cols, sharey=True)
    k = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax_k = ax[row, col] if n_rows > 1 else ax[col]
            ax_k.boxplot(probhat[y_true == k, k])
            ax_k.hlines(y_means[k], *ax_k.get_xlim(), linestyle="dashed", linewidth=1)
            ax_k.set_xticks([])
            # if col > 0:
            #     # suppress ticks on the vertical axis
            #     ax_k.set_yticks([])
            # ax_k.xaxis.set_major_locator(ticker.FixedLocator(positions))
            # ax_k.xaxis.set_major_formatter(
            #     ticker.FixedFormatter([f"y = {k+1}" for k in range(n_vals_y)])
            # )
            ax_k.set_xlabel(f"y = {k + 1}")
            ax_k.grid(axis="y", alpha=0.5)
            k += 1
            if k == n_vals_y:
                break
    if save_to:
        fig.savefig(f"{save_to}.png")
