"""Some Altair plots.

* `alt_lineplot`, `alt_superposed_lineplot`, `alt_superposed_faceted_lineplot`
* `alt_plot_fun`: plots a function
* `alt_density`, `alt_faceted_densities`: plots the density of `x`, or of `x`
  conditional on a category
* `alt_superposed_faceted_densities`: plots the density of `x` superposed by
  `f` and faceted by `g`
* `alt_scatterplot`, `alt_scatterplot_with_histo`, `alt-linked_scatterplots`:
  variants of scatter plots
* `alt_histogram_by`, `alt_histogram_continuous`: histograms of `x` by `y`,
  and of a continuous `x`
* `alt_stacked_area`,`alt_stacked_area_facets`: stacked area plots
* `plot_parameterized_estimates`: plots densities of estimates of
  coefficients, with the true values, as a function of a parameter
* `plot_true_sim_facets, plot_true_sim2_facets`: plot two simulated values
  and the true values of statistics as a function of a parameter
* `alt_tick_plots`: vertically arranged tick plots of variables
* `alt_matrix_heatmap`: plots a heatmap of a matrix.
"""

from collections.abc import Callable
from typing import cast

import altair as alt
import numpy as np
import pandas as pd
from altair_saver import save as alt_save

from bs_python_utils.bsnputils import check_matrix, check_vector
from bs_python_utils.bsutils import bs_error_abort


def _maybe_save(ch: alt.Chart, save: str | None = None):
    if save is not None:
        alt_save(ch, f"{save}.html")


def _add_title(ch: alt.Chart, title: str | None = None) -> alt.Chart:
    if title is not None:
        if not isinstance(title, str):
            raise TypeError(f"title must be a string, not {title!r}")
        ch = ch.properties(title=title)
    return cast(alt.Chart, ch)


def alt_scatterplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    time_series: bool = False,
    save: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size: int | None = 30,
    title: str | None = None,
    color: str | None = None,
    aggreg: str | None = None,
    selection: bool = False,
) -> alt.Chart:
    """Scatter ``df[str_y]`` against ``df[str_x]`` with optional coloring/selection.

    Args:
        df: Data frame holding the features to plot.
        str_x: Column name for the horizontal axis (continuous or time series).
        str_y: Column name for the vertical axis (continuous).
        time_series: If ``True`` encodes the x-axis as temporal.
        save: Optional basename to save the chart as HTML.
        xlabel: Optional label for the x-axis; must be a string when provided.
        ylabel: Optional label for the y-axis; must be a string when provided.
        size: Marker size (integer radius in pixels).
        title: Optional chart title.
        color: Column used for color encoding.
        aggreg: Optional aggregation function for ``str_y`` (e.g. ``"mean"``).
        selection: When ``True`` and a ``color`` is supplied, enable
            legend-based multi-selection.

    Returns:
        The Altair ``Chart`` for further composition or rendering.
    """
    type_x = "T" if time_series else "Q"
    var_x = alt.X(f"{str_x}:{type_x}")

    if xlabel is not None:
        if not isinstance(xlabel, str):
            raise TypeError(f"xlabel must be a string, not {xlabel!r}")
        var_x = alt.X(f"{str_x}:{type_x}", axis=alt.Axis(title=xlabel))

    var_y = f"{aggreg}({str_y}):Q" if aggreg is not None else str_y
    y_encoding: alt.Y | str

    if ylabel is not None:
        if not isinstance(ylabel, str):
            raise TypeError(f"ylabel must be a string, not {ylabel!r}")
        y_encoding = alt.Y(var_y, axis=alt.Axis(title=ylabel))
    else:
        y_encoding = var_y

    if not isinstance(size, int):
        raise TypeError(f"size must be an integer, not {size!r}")
    circles_size = size

    if color is not None:
        if not isinstance(color, str):
            raise TypeError(f"color must be a string, not {color!r}")
        if selection:
            selection_criterion = alt.selection_multi(fields=[color], bind="legend")
            ch = (
                alt.Chart(df)
                .mark_circle(size=circles_size)
                .encode(
                    x=var_x,
                    y=y_encoding,
                    color=color,
                    opacity=alt.condition(
                        selection_criterion, alt.value(1), alt.value(0.1)
                    ),
                )
                .add_params(selection_criterion)
            )
        else:
            ch = (
                alt.Chart(df)
                .mark_circle(size=circles_size)
                .encode(x=var_x, y=y_encoding, color=color)
            )
    else:
        ch = alt.Chart(df).mark_circle(size=circles_size).encode(x=var_x, y=y_encoding)

    ch = _add_title(ch, title)
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_boxes(
    df: pd.DataFrame,
    continuous_var: str,
    discrete_var: str,
    group_var: str,
    max_cols: int = 3,
    title: str | None = None,
    save: str | None = None,
) -> alt.Chart:
    """horizontal boxplots of `df[continuous_var]` by `df[discrete_var]` and
    `df[group_var]`

    Args:
        df: datframe with the three variables
        continuous_var: name of the continuous variable
        discrete_var: name of the discrete variable
        group_var: name of the grouping variable
        max_cols: maximum number of columns. Defaults to 3.
        title: a plot title. Defaults to None.
        save: the name of a file to save to (HTML extension will be added).
            Defaults to None.

    Returns:
        the chart.
    """
    boxes = (
        (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=f"{continuous_var}:Q", y=f"{discrete_var}:O", color=f"{group_var}:N"
            )
            .properties(width=180, height=180)
        )
        .facet(f"{group_var}:N", columns=max_cols)
        .resolve_scale(y="independent")
    )
    if title:
        boxes = boxes.properties(title=title)
    _maybe_save(boxes, save)
    return cast(alt.Chart, boxes)


def alt_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    time_series: bool = False,
    save: str | None = None,
    aggreg: str | None = None,
    **kwargs,
) -> alt.Chart:
    """
    Scatterplot of `df[str_x]` vs `df[str_y]`

    Args:
        df: the data with columns `str_x` and `str_y`
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        time_series: `True` if x is a time series
        save: the name of a file to save to (HTML extension will be added)
        aggreg: the name of an aggregating function for `y`

    Returns:
        the `alt.Chart` object.
    """
    type_x = "T" if time_series else "Q"
    var_y = f"{aggreg}({str_y}):Q" if aggreg is not None else str_y

    ch = alt.Chart(df).mark_line().encode(x=f"{str_x}:{type_x}", y=var_y)
    if "title" in kwargs:
        ch = ch.properties(title=kwargs["title"])
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_matrix_heatmap(
    mat: np.ndarray,
    str_format: str,
    multiple: float = 1.0,
    title: str | None = None,
    str_rows: str | None = "Row",
    str_cols: str | None = "Column",
    save: str | None = None,
) -> alt.Chart:
    """Plot a matrix heatmap using circle size and text annotations.

    Args:
        mat: Matrix to visualise; converted to a long-form data frame internally.
        str_format: Format specifier for the textual values (e.g. ``"d"`` or ``".2f"``).
        multiple: Multiplier applied to the circle size scale.
        title: Optional chart title.
        str_rows: Label used for the row coordinate in the long-form frame.
        str_cols: Label used for the column coordinate in the long-form frame.
        save: Optional basename to save the chart as HTML.

    Returns:
        The Altair ``Chart`` showing the heatmap.
    """
    n_rows, n_cols = check_matrix(mat)
    mat_df = (
        pd.DataFrame(mat)
        .stack()
        .rename_axis([str_rows, str_cols])
        .reset_index(name="Value")
    )
    if "d" in str_format:
        mat_df["Value"] = mat_df["Value"].round().astype(int)
    mat_min = mat_df["Value"].min()
    mat_df["Size"] = (mat_df["Value"] - mat_min + 1).astype(float)
    base = alt.Chart(mat_df).encode(
        x=f"{str_rows}:O", y=alt.Y(f"{str_cols}:O", sort="descending")
    )
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size(
            "Size:Q",
            legend=None,
            scale=alt.Scale(range=[1000 * multiple, 10000 * multiple]),
        )
    )
    text = base.mark_text(baseline="middle", fontSize=16).encode(
        text=alt.Text("Value:Q", format=str_format),
    )
    if title is None:
        both = (mat_map + text).properties(width=500, height=500)
    else:
        both = (mat_map + text).properties(title=title, width=400, height=400)

    _maybe_save(both, save)
    return cast(alt.Chart, both)


def alt_plot_fun(
    f: Callable,
    start: float,
    end: float,
    npoints: int = 100,
    save: str | None = None,
) -> alt.Chart:
    """Plot the scalar function ``f`` on ``[start, end]`` using ``npoints`` samples.

    Args:
        f: Callable mapping a NumPy array of x-values to an array of y-values.
        start: Lower bound of the plotting interval.
        end: Upper bound of the plotting interval.
        npoints: Number of sampling points (generated with ``np.linspace``).
        save: Optional basename to save the chart as HTML.

    Returns:
        The Altair ``Chart`` for further composition or rendering.
    """
    points = np.linspace(start, end, num=npoints)
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
    return cast(alt.Chart, ch)


def alt_density(df: pd.DataFrame, str_x: str, save: str | None = None) -> alt.Chart:
    """Plots the density of `df[str_x]`.

    Args:
        df: the data with the `str_x` variable
        str_x: the name of a continuous column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object.
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
    return cast(alt.Chart, ch)


def alt_superposed_faceted_densities(
    df: pd.DataFrame,
    str_x: str,
    str_f: str,
    str_g: str,
    max_cols: int | None = 4,
    save: str | None = None,
) -> alt.Chart:
    """
    Creates density plots of `df[str_x]` by `df[str_f]` and `df[str_g]`
    with color as per `df[str_f]` and faceted by `df[str_g]`.
    that is: facets by `str_g`, with densities conditional on `str_f` superposed.

    Args:
        df: a Pandas dataframe wity columns `str_x`, `str_f`, `str_g`
        str_x: the name of a continuous column
        str_f: the name of a categorical column
        str_g: the name of a categorical column
        max_cols: the number of columns after whcih we wrap
        save: the name of a file to save to (HTML extension will be added)

    Returns:
          the `alt.Chart` object.
    """
    densities = (
        alt.Chart(df)
        .transform_density(
            str_x,
            groupby=[str_f, str_g],
            as_=[str_x, "Density"],
        )
        .mark_line()
        .encode(
            x=f"{str_x}:Q",
            y="Density:Q",
            color=f"{str_f}:N",
        )
        .facet(column=f"{str_g}:N", columns=max_cols)
        .resolve_scale(x="independent", y="independent")
    )
    _maybe_save(densities, save)

    return cast(alt.Chart, densities)


def alt_linked_scatterplots(
    df: pd.DataFrame,
    str_x1: str,
    str_x2: str,
    str_y: str,
    str_f: str,
    save: str | None = None,
) -> alt.Chart:
    """
    Creates two scatterplots: of `df[str_x1]` vs `df[str_y]` and of
    `df[str_x2]` vs `df[str_y]`,
    both with color as per `df[str_f]`. Selecting an interval in one shows up
    in the other.

    Args:
        df:
        str_x1: the name of a continuous column
        str_x2: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
          the `alt.Chart` object.
    """
    interval = alt.selection_interval()

    base = (
        alt.Chart(df)
        .mark_point()
        .encode(
            y=f"{str_y}:Q", color=alt.condition(interval, str_f, alt.value("lightgray"))
        )
        .add_params(interval)
    )

    ch = base.encode(x=f"{str_x1}:Q") | base.encode(x=f"{str_x2}:Q")

    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_scatterplot_with_histo(
    df: pd.DataFrame, str_x: str, str_y: str, str_f: str, save: str | None = None
) -> alt.Chart:
    """
    Scatterplot of `df[str_x]` vs `df[str_y]` with colors as per `df[str_f]`
    allows to select an interval and histograns the counts of `df[str_f]` in
    the interval.

    Args:
        df: the data with the `str_x` and `str_f` variables
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
          the `alt.Chart` object.
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
        .add_params(interval)
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
    return cast(alt.Chart, ch)


def alt_faceted_densities(
    df: pd.DataFrame,
    str_x: str,
    str_f: str,
    legend_title: str | None = None,
    save: str | None = None,
    max_cols: int | None = 4,
) -> alt.Chart:
    """
    Plots the density of `df[str_x]` by `df[str_f]` in column facets

    Args:
        df: the data with the `str_x` and `str_f` variables
        str_x: the name of a continuous column
        str_f: the name of a categorical column
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)
        max_cols: we wrap after that number of columns

    Returns:
        the `alt.Chart` object.
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
    return cast(alt.Chart, ch)


def alt_superposed_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    time_series: bool = False,
    legend_title: str | None = None,
    save: str | None = None,
) -> alt.Chart:
    """
    Plots `df[str_x]` vs `df[str_y]` by `df[str_f]` on one plot

    Args:
        df: the data with the `str_x`, `str_y`, and `str_f` variables
        str_x: the name of a continuous `x` column
        str_y: the name of a continuous `y` column
        str_f: the name of a categorical `f` column
        time_series: `True` if `str_x` is a time series
        legend_title: a title for the legend
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object.
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
    return cast(alt.Chart, ch)


def alt_superposed_faceted_lineplot(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    str_g: str,
    time_series: bool = False,
    legend_title: str | None = None,
    max_cols: int | None = 5,
    save: str | None = None,
) -> alt.Chart:
    """
    Plots `df[str_x]` vs `df[str_y]` superposed by `df[str_f]` and faceted by
    `df[str_g]`

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
        the `alt.Chart` object.
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
            facet=(
                alt.Facet(f"{str_g}:N", columns=max_cols)
                if max_cols is not None
                else alt.Facet(f"{str_g}:N")
            ),
        )
    )
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_histogram_by(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_agg: str | None = "mean",
    save: str | None = None,
) -> alt.Chart:
    """
    Plots a histogram of a statistic of `str_y` by `str_x`

    Args:
        df: a dataframe with columns `str_x` and `str_y`
        str_x: a categorical variable
        str_y: a continuous variable
        str_agg: how we aggregate the values of `str_y` by `str_x`
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the Altair chart.
    """
    ch = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=str_x, y=f"{str_agg}({str_y}):Q")
        .properties(height=300, width=400)
    )
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_histogram_continuous(
    df: pd.DataFrame, str_x: str, save: str | None = None
) -> alt.Chart:
    """
    Histogram of a continuous variable `df[str_x]`

    Args:
        df: the data with the `str_x`, `str_y`, and `str_f` variables
        str_x: the name of a continuous column
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object.
    """
    ch = alt.Chart(df).mark_bar().encode(alt.X(str_x, bin=True), y="count()")
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def alt_stacked_area(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    time_series: bool = False,
    title: str | None = None,
    save: str | None = None,
) -> alt.Chart:
    """
    Normalized stacked lineplots of `df[str_x]` vs `df[str_y]` by `df[str_f]`

    Args:
        df: the data with columns for `str_x`, `str_y`, and `str_f`
        str_x: the name of a continuous column
        str_y: the name of a continuous column
        str_f: the name of a categorical column
        time_series: `True` if `str_x` is a time series
        title: a title for the plot
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object.
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
    return cast(alt.Chart, ch)


def alt_stacked_area_facets(
    df: pd.DataFrame,
    str_x: str,
    str_y: str,
    str_f: str,
    str_g: str,
    time_series: bool = False,
    max_cols: int | None = 5,
    title: str | None = None,
    save: str | None = None,
) -> alt.Chart:
    """
    Normalized stacked lineplots of `df[str_x]` vs `df[str_y]` by `df[str_f]`,
    faceted by `df[str_g]`

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
        the `alt.Chart` object.
    """
    type_x = "T" if time_series else "Q"
    ch = (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=f"{str_x}:{type_x}",
            y=alt.Y(f"{str_y}:Q", stack="normalize"),
            color=f"{str_f}:N",
            facet=(
                alt.Facet(f"{str_g}:N", columns=max_cols)
                if max_cols is not None
                else alt.Facet(f"{str_g}:N")
            ),
        )
    )
    _maybe_save(ch, save)
    return cast(alt.Chart, ch)


def _stack_estimates(
    estimate_names: str | list[str], estimates: np.ndarray, df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    adds to a dataframe `df` columns with names `estimate_names` for various
    `estimates of one coefficient

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
        size_est = check_vector(estimates, "_stack_estimates")
        if size_est != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and"
                f" {size_est} estimators"
            )
        df1[estimate_names] = estimates
        ordered_estimates = [estimate_names, "True value"]
    else:
        shape_est = check_matrix(estimates, "_stack_estimates")
        if shape_est[1] != n_estimates:
            bs_error_abort(
                f"_stack_estimates: we have {n_estimates} names of estimators and"
                f" {shape_est[1]} estimators"
            )
        for i_est, est_name in enumerate(estimate_names):
            df1[est_name] = estimates[:, i_est]
        ordered_estimates = [*estimate_names, "True value"]

    return df1, cast(list[str], ordered_estimates)


def plot_parameterized_estimates(
    parameter_name: str,
    parameter_values: np.ndarray,
    coeff_names: str | list[str],
    true_values: np.ndarray,
    estimate_names: str | list[str],
    estimates: np.ndarray,
    colors: list[str],
    save: str | None = None,
) -> alt.Chart:
    """
    Plots estimates of coefficients, with the true values, as a function of a
    parameter; one facet per coefficient

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
        the `alt.Chart` object.
    """
    n_vals = check_vector(parameter_values)
    n_coeffs = 1 if isinstance(coeff_names, str) else len(coeff_names)
    if n_coeffs == 1:
        n_true = check_vector(true_values, "plot_parameterized_estimates")
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
        n_true, n_c = check_matrix(true_values, "plot_parameterized_estimates")
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

        df2 = pd.concat(df1[i_coeff] for i_coeff in range(n_coeffs))
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

    return cast(alt.Chart, ch)


def plot_true_sim_facets(
    parameter_name: str,
    parameter_values: np.ndarray,
    stat_names: list[str],
    stat_true: np.ndarray,
    stat_sim: np.ndarray,
    colors: list[str],
    stat_title: str | None = "Statistic",
    subtitle: str | None = "True vs estimated",
    ncols: int | None = 3,
    save: str | None = None,
) -> alt.Chart:
    """
    Plots simulated and true values of statistics as a function of a parameter;
    one facet per coefficient

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
        the `alt.Chart` object.
    """
    n_stats = len(stat_names)
    nvals = check_vector(parameter_values, "plot_true_sim_facets")
    nv_true, n_stat_true = check_matrix(stat_true, "plot_true_sim_facets")
    if nv_true != nvals:
        bs_error_abort(
            f"plot_true_sim_facets: we have {nvals} parameter values and {nv_true} for"
            " stat_true."
        )
    nv_est, n_stat_est = check_matrix(stat_sim, "plot_true_sim_facets")
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
            facet=(
                alt.Facet(f"{stat_title}:N", sort=stat_names, columns=ncols)
                if ncols is not None
                else alt.Facet(f"{stat_title}:N", sort=stat_names)
            ),
        )
        .resolve_scale(y="independent")
    )

    _maybe_save(ch, save)

    return cast(alt.Chart, ch)


def plot_true_sim2_facets(
    parameter_name: str,
    parameter_values: np.ndarray,
    stat_names: list[str],
    stat_true: np.ndarray,
    stat_sim1: np.ndarray,
    stat_sim2: np.ndarray,
    colors: list[str],
    stat_title: str | None = "Statistic",
    subtitle: str | None = "True vs estimated",
    ncols: int | None = 3,
    save: str | None = None,
) -> alt.Chart:
    """
    Plots simulated values for two methods and true values of statistics as a
    function of a parameter;
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
        the `alt.Chart` object.
    """
    n_stats = len(stat_names)
    nvals = check_vector(parameter_values, "plot_true_sim2_facets")
    nv_true, n_stat_true = check_matrix(stat_true, "plot_true_sim2_facets")
    if nv_true != nvals:
        bs_error_abort(f"we have {nvals} parameter values and {nv_true} for stat_true.")
    if n_stat_true != n_stats:
        bs_error_abort(f"we have {n_stats} names for {n_stat_true} true statistics.")

    nv_est1, n_stat_est1 = check_matrix(stat_sim1, "plot_true_sim2_facets")
    if nv_est1 != nvals:
        bs_error_abort(f"we have {nvals} parameter values and {nv_est1} for stat_sim1.")
    if n_stat_est1 != n_stats:
        bs_error_abort(
            f"we have {n_stats} names for {n_stat_est1} estimated statistics."
        )
    nv_est2, n_stat_est2 = check_matrix(stat_sim2, "plot_true_sim2_facets")
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
            facet=(
                alt.Facet(f"{stat_title}:N", sort=stat_names, columns=ncols)
                if ncols is not None
                else alt.Facet(f"{stat_title}:N", sort=stat_names)
            ),
        )
        .resolve_scale(y="independent")
    )

    _maybe_save(ch, save)

    return cast(alt.Chart, ch)


def alt_tick_plots(
    df: pd.DataFrame, list_vars: str | list[str], save: str | None = None
) -> alt.Chart:
    """
    Creates a tick plot of the variables in `list_vars` of`df`, arranged vertically.

    Args:
        df: a dataframe with the variables in `list_vars`
        list_vars: the name of a column of `df`, or a list of names
        save: the name of a file to save to (HTML extension will be added)

    Returns:
        the `alt.Chart` object.
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

    return cast(alt.Chart, ch)
