""" a personal library of Altair plots
"""

from pathlib import Path
from typing import Callable, List, Union

import altair as alt
import numpy as np
import pandas as pd
from altair_saver import save as alt_save
from vega_datasets import data

from bs_python_utils.bsnputils import test_matrix, test_vector
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed


def _maybe_save(ch: alt.Chart, save: str = None):
    if save is not None:
        alt_save(ch, f"{save}.html")


def alt_scatterplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    time_series: bool = False,
    save: str = None,
    xlabel: str = None,
    ylabel: str = None,
    size: int = 30,
    title: str = None,
    color: str = None,
    aggreg: str = None,
    selection: bool = False,
) -> alt.Chart:
    """
    scatterplot of `df[str_x]` vs `df[str_y]`

    Args:
        df: the data with columns for x, y
        str_x: the name of a continuous x column
        str_y: the name of a continuous y column
        time_series: `True` if x is a time series
        xlabel: label for the horizontal axis
        ylabel: label for the vertical axis
        title: title for the graph
        size: radius of the circles
        color: variable that determines the color of the circles
        selection: if `True`, the user can select interactively from the `color` legend, if any
        save: the name of a file to save to (HTML extension will be added)
        aggreg: the name of an aggregating function for `y`

    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    var_x = alt.X(f"{str_x}:{type_x}")

    if xlabel is not None:
        if isinstance(xlabel, str):
            var_x = alt.X(f"{str_x}:{type_x}", axis=alt.Axis(title=xlabel))
        else:
            bs_error_abort(f"xlabel must be a string, not {xlabel}")

    if aggreg is not None:
        var_y = f"{aggreg}({str_y}):Q"
    else:
        var_y = str_y

    if ylabel is not None:
        if isinstance(ylabel, str):
            var_y = alt.Y(var_y, axis=alt.Axis(title=ylabel))
        else:
            bs_error_abort(f"ylabel must be a string, not {ylabel}")

    if isinstance(size, int):
        circles_size = size
    else:
        bs_error_abort(f"size must be an integer, not {size}")

    if color is not None:
        if isinstance(color, str):
            if selection:
                selection_criterion = alt.selection_multi(fields=[color], bind="legend")
                ch = (
                    alt.Chart(df)
                    .mark_circle(size=circles_size)
                    .encode(
                        x=var_x,
                        y=var_y,
                        color=color,
                        opacity=alt.condition(
                            selection_criterion, alt.value(1), alt.value(0.1)
                        ),
                    )
                    .add_selection(selection_criterion)
                )
            else:
                ch = (
                    alt.Chart(df)
                    .mark_circle(size=circles_size)
                    .encode(x=var_x, y=var_y, color=color)
                )
        else:
            bs_error_abort(f"color must be a string, not {color}")
    else:
        ch = alt.Chart(df).mark_circle(size=circles_size).encode(x=var_x, y=var_y)

    if title is not None:
        if isinstance(title, str):
            ch = ch.properties(title=title)
        else:
            bs_error_abort(f"title must be a string, not {title}")

    _maybe_save(ch, save)
    return ch


def alt_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    time_series: bool = False,
    save: str = None,
    aggreg: str = None,
    **kwargs,
) -> alt.Chart:
    """
    scatterplot of `df[str_x]` vs `df[str_y]`

    Args:
        df: the data with columns `str_x` and `str_y`
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        time_series: `True` if x is a time series
        save: the name of a file to save to (HTML extension will be added)
        aggreg: the name of an aggregating function for `y`

    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    if aggreg is not None:
        var_y = f"{aggreg}({str_y}):Q"
    else:
        var_y = str_y

    ch = alt.Chart(df).mark_line().encode(x=f"{str_x}:{type_x}", y=var_y)
    if "title" in kwargs:
        ch = ch.properties(title=kwargs["title"])
        _maybe_save(ch, save)
    return ch


def alt_plot_fun(
    f: Callable, start: float, end: float, npoints: int = 100, save: str = None
):
    """
    plots the function `f` from `start` to `end`

    Args:
        f: returns a Numpy array from a Numpy array
        start: first point on `x` axis
        end: last point on `x` axis
        npoints: number of points
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    step = (end - start) / npoints
    points = np.arange(start, end + step, step)
    fun_data = pd.DataFrame({"x": points, "y": f(points)})

    ch = (
        alt.Chart(fun_data)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )

    _maybe_save(ch, save)
    return ch


def alt_density(df: pd.DataFrame, str_x: str, save: str = None):
    """
    plots the density of `df[str_x]`

    Args:
        df: the data with the `str_x` variable
        str_x: the name of a continuous column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    ch = (
        alt.Chart(df)
        .transform_density(
            str_x,
            as_=[str_x, "Density"],
        )
        .mark_area(opacity=0.4)
        .encode(
            x=f"{str_x}:Q",
            y="Density:Q",
        )
    )

    _maybe_save(ch, save)
    return ch


def alt_linked_scatterplots(
    df: pd.DataFrame, str_x1: str, str_x2: str, str_y: str, str_f: str, save: str = None
):
    """
    two scatterplots: of `df[str_x1]` vs `df[str_y]` and of `df[str_x2]` vs `df[str_y]`,
    both with color as per `df[str_f]`. Selecting an interval in one shows up in the other.

    Args:
        df:
        str_x1: the name of a continuous column
        str_x2: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
          the `alt.Chart` object
    """
    interval = alt.selection_interval()

    base = (
        alt.Chart(df)
        .mark_point()
        .encode(
            y=f"{str_y}:Q", color=alt.condition(interval, str_f, alt.value("lightgray"))
        )
        .properties(selection=interval)
    )

    ch = base.encode(x=f"{str_x1}:Q") | base.encode(x=f"{str_x2}:Q")

    _maybe_save(ch, save)
    return ch


def alt_scatterplot_with_histo(
    df: pd.DataFrame, str_x: str, str_y: str, str_f: str, save: str = None
):
    """
    scatterplots  `df[str_x]` vs `df[str_y]` with colors as per `df[str_f]`
    allows to select an interval and histograns the counts of `df[str_f]` in the interval

    Args:
        df: the data with the `str_x` and `str_f` variables
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
          the `alt.Chart` object
    """
    interval = alt.selection_interval()

    points = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=f"{str_x}:Q",
            y=f"{str_y}:Q",
            color=alt.condition(interval, str_f, alt.value("lightgray")),
        )
        .properties(selection=interval)
    )

    histogram = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="count()",
            y=str_f,
            color=str_f,
        )
        .transform_filter(interval)
    )

    ch = points & histogram

    _maybe_save(ch, save)
    return ch


def alt_faceted_densities(
    df: pd.DataFrame,
    str_x: str,
    str_f: str,
    legend_title: str = None,
    save: str = None,
    max_cols: int = 4,
):
    """
    plots the density of `df[str_x]` by `df[str_f]` in column facets

    Args:
        df: the data with the `str_x` and `str_f` variables
        str_x: the name of a continuous column
        str_f: the name of a categorical column
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)
        max_cols: we wrap after that number of columns

    Returns:
        the `alt.Chart` object
    """
    our_legend_title = str_f if legend_title is None else legend_title
    ch = (
        alt.Chart(df)
        .transform_density(
            str_x,
            groupby=[str_f],
            as_=[str_x, "Density"],
        )
        .mark_area(opacity=0.4)
        .encode(
            x=f"{str_x}:Q",
            y="Density:Q",
            color=alt.Color(f"{str_f}:N", title=our_legend_title),
        )
        .facet(f"{str_f}:N", columns=max_cols)
    )

    _maybe_save(ch, save)
    return ch


def alt_superposed_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    time_series: bool = False,
    legend_title: str = None,
    save: str = None,
) -> alt.Chart:
    """
    plots `df[str_x]` vs `df[str_y]` by `df[str_f]` on one plot

    Args:
        df: the data with the `str_x`, `str_y`, and `str_f` variables
        str_x: the name of a continuous `x` column
        str_y: the name of a continuous `y` column
        str_f: the name of a categorical `f` column
        time_series: `True` if `str_x` is a time series
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    our_legend_title = str_f if legend_title is None else legend_title
    ch = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=f"{str_x}:{type_x}",
            y=f"{str_y}:Q",
            color=alt.Color(f"{str_f}:N", title=our_legend_title),
        )
    )
    _maybe_save(ch, save)
    return ch


def alt_superposed_faceted_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    str_g: str,
    time_series: bool = False,
    legend_title: str = None,
    max_cols: int = 5,
    save: str = None,
) -> alt.Chart:
    """
    plots `df[str_x]` vs `df[str_y]` superposed by `df[str_f]` and faceted by `df[str_g]`

    Args:
        df: the data with the `str_x`, `str_y`, and `str_f` variables
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        str_g: the name of a categorical column
        time_series: `True` if `str_x` is a time series
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)
        max_cols: we wrap after that number of columns


    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    our_title = str_f if legend_title is None else legend_title
    ch = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=f"{str_x}:{type_x}",
            y=f"{str_y}:Q",
            color=alt.Color(f"{str_f}:N", title=our_title),
            facet=alt.Facet(f"{str_g}:N", columns=max_cols),
        )
    )
    _maybe_save(ch, save)
    return ch


def alt_histogram_by(
    df: pd.DataFrame, str_x: str, str_y: str, str_agg: str = "mean", save: str = None
) -> alt.Chart:
    """
    plots a histogram of a statistic of `str_y` by `str_x`

    Args:
        df: a dataframe with columns `str_x` and `str_y`
        str_x: a categorical variable
        str_y: a continuous variable
        str_agg: how we aggregate the values of `str_y` by `str_x`
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the Altair chart
    """
    ch = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=str_x, y=f"{str_agg}({str_y}):Q")
        .properties(height=300, width=400)
    )
    _maybe_save(ch, save)
    return ch


def alt_histogram_continuous(
    df: pd.DataFrame, str_x: str, save: str = None
) -> alt.Chart:
    """
    histogram of a continuous variable `df[str_x]`

    Args:
        df: the data with the `str_x`, `str_y`, and `str_f` variables
        str_x: the name of a continuous column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    ch = alt.Chart(df).mark_bar().encode(alt.X(str_x, bin=True), y="count()")
    _maybe_save(ch, save)
    return ch


def alt_stacked_area(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    time_series=False,
    title: str = None,
    legend_title: str = None,
    save: str = None,
) -> alt.Chart:
    """
    normalized stacked lineplots of `df[str_x]` vs `df[str_y]` by `df[str_f]`

    Args:
        df: the data with columns for `str_x`, `str_y`, and `str_f`
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        time_series: `True` if `str_x` is a time series
        title: a title for the plot
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    ch = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=f"{str_x}:{type_x}",
            y=alt.Y(f"{str_y}:Q", stack="normalize"),
            color=f"{str_f}:N",
        )
    )
    if title is not None:
        ch = ch.properties(title=title)

    _maybe_save(ch, save)
    return ch


def alt_stacked_area_facets(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    str_g: str,
    time_series: bool = False,
    max_cols: int = 5,
    title: str = None,
    save: str = None,
) -> alt.Chart:
    """
    normalized stacked lineplots of `df[str_x]` vs `df[str_y]` by `df[str_f]`, faceted by `df[str_g]`

    Args:
        df: the data with columns for `str_x`, `str_y`, and `str_f`
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        str_g: the name of a categorical column
        time_series: `True` if `str_x` is a time series
        title: a title for the plot
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    type_x = "T" if time_series else "Q"
    ch = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=f"{str_x}:{type_x}",
            y=alt.Y(f"{str_y}:Q", stack="normalize"),
            color=f"{str_f}:N",
            facet=alt.Facet(f"{str_g}:N", columns=max_cols),
        )
    )
    _maybe_save(ch, save)
    return ch


def _stack_estimates(
    estimate_names: Union[str, List[str]], estimates: np.ndarray, df: pd.DataFrame
) -> pd.DataFrame:
    """
    adds to a dataframe `df` columns with names `estimate_names` for various `estimates of one coefficient

    Args:
        estimate_names: names of the n estimate columns to be added
        estimates: a matrix with n columns vectors
        df: a receiving data frame

    Returns:
        the dataframe, updated; and the names+['True value']
    """
    df1 = df.copy()
    n_estimates = 1 if isinstance(estimate_names, str) else len(estimate_names)
    if n_estimates == 1:
        size_est = test_vector(estimates, "_stack_estimates")
        if size_est != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and"
                f" {size_est} estimators"
            )
        df1[estimate_names] = estimates
        ordered_estimates = [estimate_names, "True value"]
    else:
        shape_est = test_matrix(estimates, "_stack_estimates")
        if shape_est[1] != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and"
                f" {shape_est[1]} estimators"
            )
        for i_est, est_name in enumerate(estimate_names):
            df1[est_name] = estimates[:, i_est]
        ordered_estimates = estimate_names + ["True value"]

    return df1, ordered_estimates


def plot_parameterized_estimates(
    parameter_name: str,
    parameter_values: np.ndarray,
    coeff_names: Union[str, List[str]],
    true_values: np.ndarray,
    estimate_names: Union[str, List[str]],
    estimates: np.ndarray,
    colors: List[str],
    save: str = None,
) -> alt.Chart:
    """
    plots estimates of coefficients, with the true values,  as a function of a parameter; one facet per coefficient

    Args:
        parameter_name: the name of the parameter
        parameter_values: a vector of `n_vals` values for the parameter
        coeff_names: the names of the `n_coeffs` coefficients
        true_values: their true values, depending on the parameter or not
        estimate_names: names of the estimates
        estimates: their values
        colors: colors for the various estimates
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    n_vals = test_vector(parameter_values)
    n_coeffs = 1 if isinstance(coeff_names, str) else len(coeff_names)
    if n_coeffs == 1:
        n_true = test_vector(true_values, "plot_parameterized_estimates")
        if n_true != n_vals:
            bs_error_abort(
                f"plot_parameterized_estimates: we have {n_true} values and"
                f" {n_vals} parameter values."
            )
        df = pd.DataFrame({parameter_name: parameter_values, "True value": true_values})
        df1, ordered_estimates = _stack_estimates(estimate_names, estimates, df)
        df1m = pd.melt(df1, parameter_name, var_name="Estimate")
        ch = (
            alt.Chart(df1m)
            .mark_line()
            .encode(
                x=f"{parameter_name}:Q",
                y="value:Q",
                strokeDash=alt.StrokeDash("Estimate:N", sort=ordered_estimates),
                color=alt.Color(
                    "Estimate:N",
                    sort=estimate_names,
                    scale=alt.Scale(domain=ordered_estimates, range=colors),
                ),
            )
        )
    else:
        n_true, n_c = test_matrix(true_values, "plot_parameterized_estimates")
        if n_true != n_vals:
            bs_error_abort(
                f"plot_parameterized_estimates: we have {n_true} true values and"
                f" {n_vals} parameter values."
            )
        if n_c != n_coeffs:
            bs_error_abort(
                f"plot_parameterized_estimates: we have {n_c} columns of true values"
                f" and {n_coeffs} coefficients."
            )
        df1 = [None] * n_coeffs
        for i_coeff, coeff in enumerate(coeff_names):
            df_i = pd.DataFrame(
                {
                    parameter_name: parameter_values,
                    "True value": true_values[:, i_coeff],
                }
            )
            df1[i_coeff], ordered_estimates = _stack_estimates(
                estimate_names, estimates[..., i_coeff], df_i
            )
            df1[i_coeff]["Coefficient"] = coeff

        df2 = pd.concat((df1[i_coeff] for i_coeff in range(n_coeffs)))
        ordered_colors = colors
        df2m = pd.melt(df2, [parameter_name, "Coefficient"], var_name="Estimate")
        ch = (
            alt.Chart(df2m)
            .mark_line()
            .encode(
                x=f"{parameter_name}:Q",
                y="value:Q",
                strokeDash=alt.StrokeDash("Estimate:N", sort=ordered_estimates),
                color=alt.Color(
                    "Estimate:N",
                    sort=ordered_estimates,
                    scale=alt.Scale(domain=ordered_estimates, range=ordered_colors),
                ),
            )
            .facet(alt.Facet("Coefficient:N", sort=coeff_names))
            .resolve_scale(y="independent")
        )

    _maybe_save(ch, save)

    return ch


def plot_true_sim_facets(
    parameter_name: str,
    parameter_values: np.ndarray,
    stat_names: List[str],
    stat_true: np.ndarray,
    stat_sim: np.ndarray,
    colors: List[str],
    stat_title: str = "Statistic",
    subtitle: str = "True vs estimated",
    ncols: int = 3,
    save: str = None,
) -> alt.Chart:
    """
    plots simulated and true values of statistics as a function of a parameter; one facet per coefficient

    Args:
        parameter_name: the name of the parameter
        parameter_values: a vector of `n_vals` values for the parameter
        stat_names: the names of the `n` statistics
        stat_true: their true values, `(n_vals, n)`
        stat_sim: their simulated values
        colors: colors for the various estimates
        stat_title: main title
        subtitle: subtitle
        ncols: wrap after `ncols` columns
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    n_stats = len(stat_names)
    nvals = test_vector(parameter_values, "plot_true_sim_facets")
    nv_true, n_stat_true = test_matrix(stat_true, "plot_true_sim_facets")
    if nv_true != nvals:
        bs_error_abort(
            f"plot_true_sim_facets: we have {nvals} parameter values and {nv_true} for"
            " stat_true."
        )
    nv_est, n_stat_est = test_matrix(stat_sim, "plot_true_sim_facets")
    if nv_est != nvals:
        bs_error_abort(
            f"plot_true_sim_facets: we have {nvals} parameter values and {nv_est} for"
            " stat_sim."
        )
    if n_stat_true != n_stats:
        bs_error_abort(
            f"plot_true_sim_facets: we have {n_stats} names for {n_stat_true} true"
            " statistics."
        )
    if n_stat_est != n_stats:
        bs_error_abort(
            f"plot_true_sim_facets: we have {n_stats} names for {n_stat_est} estimated"
            " statistics."
        )
    df = pd.DataFrame(
        {
            parameter_name: parameter_values,
            "True value": stat_true[:, 0],
            "Estimated": stat_sim[:, 0],
            stat_title: stat_names[0],
        }
    )
    for i_stat in range(1, n_stats):
        df_i = pd.DataFrame(
            {
                parameter_name: parameter_values,
                "True value": stat_true[:, i_stat],
                "Estimated": stat_sim[:, i_stat],
                stat_title: stat_names[i_stat],
            }
        )
        df = pd.concat((df, df_i))
    sub_order = ["True value", "Estimated"]
    dfm = pd.melt(df, [parameter_name, stat_title], var_name=subtitle)
    ch = (
        alt.Chart(dfm)
        .mark_line()
        .encode(
            x=f"{parameter_name}:Q",
            y="value:Q",
            strokeDash=alt.StrokeDash(f"{subtitle}:N", sort=sub_order),
            color=alt.Color(
                f"{subtitle}:N",
                sort=sub_order,
                scale=alt.Scale(domain=sub_order, range=colors),
            ),
            facet=alt.Facet(f"{stat_title}:N", sort=stat_names, columns=ncols),
        )
        .resolve_scale(y="independent")
    )

    _maybe_save(ch, save)

    return ch


def plot_true_sim2_facets(
    parameter_name: str,
    parameter_values: np.ndarray,
    stat_names: List[str],
    stat_true: np.ndarray,
    stat_sim1: np.ndarray,
    stat_sim2: np.ndarray,
    colors: List[str],
    stat_title: str = "Statistic",
    subtitle: str = "True vs estimated",
    ncols: int = 3,
    save: str = None,
) -> alt.Chart:
    """
    plots simulated values for two methods and true values of statistics as a function of a parameter;
    one facet per coefficient

    Args:
        parameter_name: the name of the parameter
        parameter_values: a vector of `n_vals` values for the parameter
        stat_names: the names of the `n` statistics
        stat_true: their true values, `(n_vals, n)`
        stat_sim1: their simulated values, method 1
        stat_sim2: their simulated values, method 2
        colors: colors for the various estimates
        stat_title: main title
        subtitle: subtitle
        ncols: wrap after `ncols` columns
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    n_stats = len(stat_names)
    nvals = test_vector(parameter_values, "plot_true_sim2_facets")
    nv_true, n_stat_true = test_matrix(stat_true, "plot_true_sim2_facets")
    if nv_true != nvals:
        bs_error_abort(f"we have {nvals} parameter values and {nv_true} for stat_true.")
    if n_stat_true != n_stats:
        bs_error_abort(f"we have {n_stats} names for {n_stat_true} true statistics.")

    nv_est1, n_stat_est1 = test_matrix(stat_sim1, "plot_true_sim2_facets")
    if nv_est1 != nvals:
        bs_error_abort(f"we have {nvals} parameter values and {nv_est1} for stat_sim1.")
    if n_stat_est1 != n_stats:
        bs_error_abort(
            f"we have {n_stats} names for {n_stat_est1} estimated statistics."
        )
    nv_est2, n_stat_est2 = test_matrix(stat_sim2, "plot_true_sim2_facets")
    if nv_est2 != nvals:
        bs_error_abort(f"we have {nvals} parameter values and {nv_est2} for stat_sim2.")
    if n_stat_est2 != n_stats:
        bs_error_abort(
            f"we have {n_stats} names for {n_stat_est2} estimated statistics."
        )

    df = pd.DataFrame(
        {
            parameter_name: parameter_values,
            "True value": stat_true[:, 0],
            "Estimated1": stat_sim1[:, 0],
            "Estimated2": stat_sim2[:, 0],
            stat_title: stat_names[0],
        }
    )
    for i_stat in range(1, n_stats):
        df_i = pd.DataFrame(
            {
                parameter_name: parameter_values,
                "True value": stat_true[:, i_stat],
                "Estimated1": stat_sim1[:, i_stat],
                "Estimated2": stat_sim2[:, i_stat],
                stat_title: stat_names[i_stat],
            }
        )
        df = pd.concat((df, df_i))
    sub_order = ["True value", "Estimated1", "Estimated2"]
    dfm = pd.melt(df, [parameter_name, stat_title], var_name=subtitle)
    ch = (
        alt.Chart(dfm)
        .mark_line()
        .encode(
            x=f"{parameter_name}:Q",
            y="value:Q",
            strokeDash=alt.StrokeDash(f"{subtitle}:N", sort=sub_order),
            color=alt.Color(
                f"{subtitle}:N",
                sort=sub_order,
                scale=alt.Scale(domain=sub_order, range=colors),
            ),
            facet=alt.Facet(f"{stat_title}:N", sort=stat_names, columns=ncols),
        )
        .resolve_scale(y="independent")
    )

    _maybe_save(ch, save)

    return ch


def alt_tick_plots(
    df: pd.DataFrame, list_vars: Union[str, list[str]], save: str = None
) -> alt.Chart:
    """
    ticks plot the `df` variables in `list_vars`, arranged vertically

    Args:
        df: a dataframe with the variables in `list_vars`
        list_vars: the name of a column of `df`, or a list of names
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object
    """
    if isinstance(list_vars, str):
        varname = list_vars
        ch = alt.Chart(df).encode(x=varname).mark_tick()
    else:
        ch = (
            alt.Chart(df)
            .encode(alt.X(alt.repeat("row"), type="quantitative"))
            .mark_tick()
            .repeat(row=list_vars)
            .resolve_scale(y="independent")
        )

    _maybe_save(ch, save)

    return ch


if __name__ == "__main__":
    _ = mkdir_if_needed(Path.cwd() / "altair_figs")

    cars = data.cars()

    ch = alt_superposed_lineplot(
        cars,
        "Horsepower",
        "Weight_in_lbs",
        "Origin",
        save="altair_figs/cars_superposed_lineplot",
    )

    ch = alt_superposed_faceted_lineplot(
        cars,
        "Horsepower",
        "Weight_in_lbs",
        "Origin",
        "Year",
        save="altair_figs/cars_superposed_faceted_lineplot",
    )

    ch = alt_histogram_continuous(
        cars, "Horsepower", save="altair_figs/cars_histo_cont"
    )

    ch = alt_histogram_by(
        cars, "Origin", "Horsepower", str_agg="median", save="altair_figs/cars_histo_by"
    )

    elec = data.iowa_electricity()

    ch = alt_stacked_area(
        elec,
        "year",
        "net_generation",
        "source",
        time_series=True,
        title="Generators",
        save="altair_figs/elec_stacked_areas",
    )

    ch = alt_stacked_area_facets(
        cars,
        "Year",
        "Displacement",
        "Name",
        "Origin",
        time_series=True,
        save="altair_figs/cars_stacked_areas_facets",
    )

    ch = alt_scatterplot(
        cars,
        "Year",
        "Displacement",
        time_series=True,
        title="Average car displacement",
        aggreg="average",
        save="altair_figs/cars_scatter",
    )

    ch = alt_scatterplot(
        cars,
        "Year",
        "Displacement",
        time_series=True,
        title="Average car displacement",
        aggreg="average",
        save="altair_figs/cars_scatter_labx",
        xlabel="Model year",
    )

    ch = alt_scatterplot(
        cars,
        "Horsepower",
        "Displacement",
        title="Car displacement",
        color="Origin",
        selection=True,
        save="altair_figs/cars_scatter_color",
        xlabel="Horsepower",
    )

    ch = alt_linked_scatterplots(
        cars,
        "Horsepower",
        "Displacement",
        "Miles_per_Gallon",
        "Origin",
        save="altair_figs/cars_linked_scatters",
    )

    ch = alt_scatterplot_with_histo(
        cars,
        "Horsepower",
        "Displacement",
        "Origin",
        save="altair_figs/cars_linked_scatter_histo",
    )

    ch = alt_density(cars, "Horsepower", save="altair_figs/horsepower_density")

    ch = alt_faceted_densities(
        cars, "Horsepower", "Origin", save="altair_figs/horsepower_distribs"
    )

    def fnp(x):
        return x * x - 4.0

    ch = alt_plot_fun(fnp, -2.0, 3.0, save="altair_figs/plot_function")

    # test plot_parameterized_estimates
    nvals = 50
    vals_p = np.arange(nvals) / (nvals - 1.0)
    true_vals = np.column_stack((vals_p, np.ones(nvals)))
    estimates_a = np.random.normal(size=((nvals, 2)), scale=0.2) + vals_p.reshape(
        (-1, 1)
    )
    estimates_b = np.random.normal(size=((nvals, 2)), scale=0.2) + np.ones((nvals, 2))
    estimates = np.zeros((nvals, 2, 2))
    estimates[..., 0] = estimates_a
    estimates[..., 1] = estimates_b

    ch = plot_parameterized_estimates(
        "Value of p",
        vals_p,
        ["a", "b"],
        true_vals,
        ["MLE", "MM"],
        estimates,
        colors=["black", "green", "blue"],
        save="altair_figs/ppe.html",
    )

    stats = np.reshape(estimates, (nvals, 4))
    true_vals = stats + np.random.normal(loc=-0.1, scale=0.2, size=stats.shape)
    ch = plot_true_sim_facets(
        "Value of p",
        vals_p,
        ["a", "b", "c", "d"],
        true_vals,
        stats,
        colors=["black", "red"],
        ncols=2,
        save="altair_figs/ptsf.html",
    )

    stats2 = stats + np.random.normal(loc=0.1, scale=0.2, size=stats.shape)
    ch = plot_true_sim2_facets(
        "Value of p",
        vals_p,
        ["a", "b", "c", "d"],
        true_vals,
        stats,
        stats2,
        colors=["black", "red", "green"],
        ncols=2,
        save="altair_figs/pts2f.html",
    )

    ch = alt_tick_plots(cars, "Weight_in_lbs", save="altair_figs/weight_ticks")

    ch = alt_tick_plots(
        cars, ["Horsepower", "Weight_in_lbs"], save="altair_figs/horse_weight_ticks"
    )
