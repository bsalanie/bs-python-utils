""" Some Seaborn plotting utilities:

* `bs_sns_get_legend`: get the `Legend` object of a Seaborn plot
* `bs_sns_bar_x_byf`: make a bar plot of `x` by `f`
* `bs_sns_bar_x_byfg`: make a bar plot of `x` by `f` and `g`
* `bs_sns_plot_density`: basic density plot
* `bs_sns_density_estimates`: plots the densities of estimates of several coefficients with several methods, superposed by methods and faceted by coefficients.
"""

from typing import Callable, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SeabornGraph = tuple[mpl.figure.Figure, mpl.axes.Axes]


def bs_sns_get_legend(g: SeabornGraph) -> mpl.legend.Legend:
    """
    Get the `Legend` object of a Seaborn plot.

    Args:
        g: the plot object

    Returns:
        leg: the associated `Legend` object.
    """
    # check axes and find which one has a legend
    axs = g[1]
    if isinstance(axs, mpl.axes.Axes):  # only one Axes
        leg = axs.get_legend()
    else:
        for ax in axs.flat():
            leg = ax.get_legend()
            if leg is not None:
                break
    # or legend may be on a figure
    if leg is None:
        leg = g._legend
    return leg


def bs_sns_bar_x_byf(
    df: pd.DataFrame,
    xstr: str,
    fstr: str,
    statistic: Callable = np.mean,
    label_x: str | None = None,
    label_f: str | None = None,
    title: str | None = None,
) -> SeabornGraph:
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
    return cast(SeabornGraph, gbar)


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
) -> SeabornGraph:
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
        gbar_legend = bs_sns_get_legend(gbar)
        gbar_legend.set_title(label_g)
    if title is not None:
        ax.set_title(title)
    return cast(SeabornGraph, gbar)


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
