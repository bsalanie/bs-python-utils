"""Some Seaborn plotting utilities:

* `bs_sns_get_legend`: get the `Legend` object of a Seaborn plot
* `bs_sns_bar_x_byf`: make a bar plot of `x` by `f`
* `bs_sns_bar_x_byfg`: make a bar plot of `x` by `f` and `g`
* `bs_sns_plot_density`: basic density plot
* `bs_sns_density_estimates`: plots the densities of estimates of several
  coefficients with several methods, superposed by methods and faceted by
  coefficients.
"""

from collections.abc import Callable
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def bs_regplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    line_color: str = "red",
    save: str | None = None,
) -> Axes:
    """Draw a seaborn regplot with title and legend.

    Args:
        data: dataframe, should contain columns `x` and `y`
        x: column name of x
        y: column name of y
        title: title of plot
        line_color: color of the regression line. Red by default
        save: where to save it, if requested

    Returns:
        the plot
    """
    sns.set_style("whitegrid")
    g = sns.regplot(
        data=data,
        x=x,
        y=y,
        label="Data",
        line_kws={"label": "Linear Fit (95% CI)", "color": line_color},
    )
    if title:
        plt.title(title)
    plt.legend(loc="best")
    if save:
        plt.savefig(f"{save}.png", dpi=400)
    return cast(Axes, g)


def bs_sns_bar_x_byf(
    df: pd.DataFrame,
    xstr: str,
    fstr: str,
    statistic: Callable = np.mean,
    label_x: str | None = None,
    label_f: str | None = None,
    title: str | None = None,
) -> Axes:
    """Make a bar plot of `x` by `f`.

    Args:
        df: dataframe, should contain columns `xstr` and `fstr`
        xstr: column name of x
        fstr: column name of f
        statistic: statistic to plot (by default, the mean)
        label_x: label of x
        label_f: label of f
        title: title of plot

    Returns:
        the plot.
    """
    fig, ax = plt.subplots()
    gbar = sns.barplot(
        x=fstr,
        y=xstr,
        data=df,
        estimator=statistic,
        errcolor="r",
        errwidth=0.75,
        capsize=0.2,
        ax=ax,
    )
    xlab = fstr if label_f is None else label_f
    ylab = xstr if label_x is None else label_x
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title is not None:
        ax.set_title(title)
    return cast(Axes, gbar)


def bs_sns_bar_x_byfg(
    df: pd.DataFrame,
    xstr: str,
    fstr: str,
    gstr: str,
    statistic: Callable = np.mean,
    label_x: str | None = None,
    label_f: str | None = None,
    label_g: str | None = None,
    title: str | None = None,
) -> Axes:
    """Make a bar plot of x by f and g

    Args:
        df: dataframe, should contain columns  `xstr`, `fstr`,  and `gstr`
        xstr: column name of x
        fstr: column name of f
        gstr: column name of g
        statistic: statistic to plot (by default, the mean)
        label_x: label of x
        label_f: label of f
        label_g: label of g in legend
        title: title of plot

    Returns:
        the plot.
    """
    _, ax = plt.subplots()
    gbar = sns.barplot(
        x=fstr,
        y=xstr,
        data=df,
        hue=gstr,
        estimator=statistic,
        errcolor="r",
        errwidth=0.75,
        capsize=0.2,
        ax=ax,
    )
    xlab = fstr if label_f is None else label_f
    ylab = xstr if label_x is None else label_x
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if label_g is not None:
        plt.gca().legend(title=label_g)
    if title is not None:
        ax.set_title(title)
    return cast(Axes, gbar)


def bs_sns_plot_density(
    df: pd.DataFrame, var_name: str, save_to: str | None = None
) -> None:
    """plots the density of a variable

    Args:
        df: dataframe, should contain column `var_name`
        var_name:  the name of a continuous variable
        save_to: (maybe) where we save the plot, with `.png` extension.
    """
    var_y = df[var_name].values
    var_fig = sns.kdeplot(var_y)
    var_fig.axvline(x=0, c="k", ls="dashed")
    var_fig.set_title(f"Density of the {var_name}")
    var_fig.set_xlabel(f"Value of {var_name}")
    var_fig.set_ylabel("Value of the density")
    if save_to:
        plt.savefig(f"{save_to}.png")


def bs_sns_density_estimates(
    df: pd.DataFrame,
    true_values: np.ndarray,
    method_string: str | None = "Estimator",
    coeff_string: str | None = "Parameter",
    estimate_string: str | None = "Estimate",
    max_cols: int = 3,
) -> sns.FacetGrid:
    """
    Plots the densities of estimates of several coefficients with several methods,
    superposed by methods and faceted by coefficients.

    Args:
        df: contains columns `method_string`, `coeff_name`, `estimate_value`
        true_values: the true values of the coefficients
        method_string: the name of the column that indicates the method
        coeff_string: the name of the column that indicates the coefficient
        estimate_string: the name of the column that gives the value of the estimate
        max_cols: we wrap after that

    Returns:
        the `FacetGrid` plot.

    """
    g = sns.FacetGrid(
        data=df,
        sharex=False,
        sharey=False,
        hue=method_string,
        col=coeff_string,
        col_wrap=max_cols,
    )
    g.map(sns.kdeplot, estimate_string)
    g.set_titles("{col_name}")
    for true_val, ax in zip(true_values, g.axes.ravel(), strict=True):
        ax.vlines(true_val, *ax.get_ylim(), color="k", linestyles="dashed")
    g.add_legend()

    return g
