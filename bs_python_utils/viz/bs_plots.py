"""General plotting utilities.

* `set_axis`: set the axis limits, with a margin
* `drawArrow_2dim`: draw an arrow between two points in a 2D Matplotlib plot.
"""

import matplotlib.axes as axes
import numpy as np


def set_axis(x: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """sets the axis limits with a margin

    Args:
        x: the values of the variable
        margin: the margin to add, a fraction of the range of the variable

    Returns:
        the min and max for the axis.
    """
    x_min, x_max = x.min(), x.max()
    scaled_diff = margin * (x_max - x_min)
    x_min -= scaled_diff
    x_max += scaled_diff
    return x_min, x_max


def drawArrow_2dim(
    ax: axes.Axes,
    xA: float,
    xB: float,
    yA: float,
    yB: float,
    c: str = "k",
    ls: str = "-",
) -> None:
    """Draw an arrow between two points in a 2D Matplotlib plot.

    Args:
        ax: the axis object
        xA: the x coordinate of the starting point
        xB: the x coordinate of the ending point
        yA: the y coordinate of the starting point
        yB: the y coordinate of the ending point
        c: the color of the arrow
        ls: the line style of the arrow

    Returns:
        nothing
    """
    n = 50
    x = np.linspace(xA, xB, 2 * n + 1)
    y = np.linspace(yA, yB, 2 * n + 1)
    ax.plot(x, y, color=c, linestyle=ls)
    ax.annotate(
        "",
        xy=(x[n], y[n]),
        xytext=(x[n - 1], y[n - 1]),
        arrowprops={"arrowstyle": "-|>", "color": c},
        size=15,
        # zorder=2,
    )
