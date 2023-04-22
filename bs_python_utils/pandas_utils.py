"""Utility functions for pandas """

from itertools import product
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bs_python_utils.bsnputils import bs_error_abort, test_vector_or_matrix
from bs_python_utils.bsutils import print_stars


def bspd_print(
    df: pd.DataFrame,
    s: Optional[str] = "",
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    precision: Optional[int] = None,
) -> None:
    """Pretty-prints a data frame

    Args:
        * df: any data frame
        * s: an optional title string
        * max_rows: maximum number of rows to print (all by default)
        * max_cols: maximum number of columns to print (all by default)
        * precision: of numbers. 3 digits by default.
    Returns: nothing.
    """

    print_stars(s)
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        max_cols,
        "display.precision",
        precision,
    ):
        print(df)


def bspd_cross_products(
    df: pd.DataFrame,
    l1: list[str],
    l2: Optional[list[str]] = None,
    with_squares: Optional[bool] = True,
) -> pd.DataFrame:
    """Returns a DataFrame with cross-products of the variables of `df`
    whose names are in `l1` and `l2`.

    Args:
        df: any data frame
        l1: a list of names of variables that belong to `df`
        l2: ibidem; `l1` by default
        with_squares: if `False`, we drop the squares. `True` by default.

    Returns:
        the data frame of cross-products with concatenated names.
    """

    lp2 = l1 if l2 is None else l2
    l12 = list(product(l1, lp2))
    cross_pairs = [[x[0], x[1]] for x in l12 if x[0] != x[1]]
    unique_pairs = []
    for i, c in enumerate(cross_pairs):
        print(c)
        c_ordered = c if c[0] < c[1] else list(reversed(c))
        print(c_ordered)
        if c_ordered not in unique_pairs:
            unique_pairs.append(c_ordered)
        print(unique_pairs)

    col_names = sorted([(x[0], x[1], f"{x[0]}*{x[1]}") for x in unique_pairs])

    if with_squares:
        col_names_squares = sorted(
            [(x[0], x[1], f"{x[0]}**2") for x in l12 if x[0] == x[1]]
        )
        col_names += col_names_squares

    df_cprods = pd.DataFrame(
        {col_name: df[x0] * df[x1] for (x0, x1, col_name) in col_names}
    )

    return df_cprods


def _list_str(names: Union[str, List[str]], suffix: str = None) -> List[str]:
    """make a list of strings with possibly the added suffix

    Args:
        names: a string or a list of strings
        suffix: a string, if any

    Returns:
        a list of the strings in names, with the suffix if specified
    """
    if isinstance(names, str):
        if suffix is not None:
            return [names + suffix]
        else:
            return [names]
    elif isinstance(names, list):
        if suffix is not None:
            return [name + suffix for name in names]
        else:
            return names
    else:
        bs_error_abort("names should be a string or a list of strings.")


def bspd_statsdf(
    T: Union[np.ndarray, List[np.ndarray]],
    col_names: Union[Union[str, List[str]], List[Union[str, List[str]]]],
) -> pd.DataFrame:
    """
    make a dataframe with columns from the array(s) in T and names from col_names

    Args:
        T: a list of n_T matrices or vectors with N rows, or a matrix or a vector with N rows
        col_names: a list of n_T name objects; a name object must be a string or a list of strings,
            with the names for the column(s) of the corresponding T matrix

    Returns:
        a dataframe with the named columns
    """
    if isinstance(T, list):
        n_T = len(T)
        if not isinstance(col_names, list):
            bs_error_abort("If T is a list, then col_names should be a list too.")
        elif len(col_names) != n_T:
            bs_error_abort(f"T has {n_T} elements but col_names has {len(col_names)}.")
        # ndims_T = np.ndarray(n_T, dtype=int)
        shape_T = []
        for i in range(n_T):
            # ndims_T[i] = test_vector_or_matrix(T[i])
            shape_T.append(T[i].shape)
        set_nrows = set([shape_i[0] for shape_i in shape_T])
        if len(set_nrows) > 1:
            bs_error_abort("All T arrays should have the same number of rows.")
        big_T = T[0]
        big_names = _list_str(col_names[0], suffix="_1")
        for i in range(1, n_T):
            big_T = np.column_stack((big_T, T[i]))
            big_names.extend(_list_str(col_names[i], suffix=f"_{i+1}"))

        df = pd.DataFrame(big_T, columns=big_names, copy=True)

    else:  # only one element in T
        ndims_T = test_vector_or_matrix(T)
        if ndims_T == 1:
            if not isinstance(col_names, str):
                bs_error_abort(f"T is a vector but col_names is {col_names}")
            df = pd.DataFrame(T, columns=[col_names], copy=True)
        elif ndims_T == 2:
            N, K = T.shape
            K2 = len(col_names)
            if K2 != K:
                bs_error_abort(f"T is {T.shape} but col_names has {K2} elements")
            df = pd.DataFrame(T, columns=col_names, copy=True)

    return df


def _test_names_n(col_names: List[str]) -> List[int]:
    """
    Tests that if a name in col_names ends with `_n`, with `n` an integer, then all names do

    Args:
        col_names: a list of names

    Returns:
        the list of the values of `n`  if all names in the list end in `_n`; `[]` if none of them does.
        Aborts otherwise.
    """
    underscore_n = []
    ending_integers = []
    for name in col_names:
        split_name = name.split("_")
        len_split = len(split_name)
        if len_split > 1:  # found at least one '_'
            last_bit = split_name[-1]
            try:
                ending_int = int(last_bit)
                underscore_n.append(True)  # ends with '_n'
                ending_integers.append(ending_int)
            except ValueError:
                underscore_n.append(False)
        else:
            underscore_n.append(False)

    values_integers = set(ending_integers)
    n_values_integers = len(values_integers)

    if n_values_integers == 0 or all(
        underscore_n
    ):  # none ends in '_n' or all end in '_n'
        return list(values_integers)
    else:
        bs_error_abort(
            "If a column name ends with '_n' where n is an integer,  then all should."
        )


def bspd_prepareplot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: any dataframe whose column names either all end in '_n' for n an integer, or none does

    Returns:
        a properly melted dataframe for plotting, with columns 'Sample', 'Statistic', 'Value',
        and 'Group' if there are several integers
    """
    # check the names of the columns
    values_integers = _test_names_n(df.columns)
    n_values_integers = len(values_integers)

    df2 = df.copy()
    df2["Sample"] = np.arange(df.shape[0])
    dfm = pd.melt(
        df2,
        id_vars="Sample",
        value_vars=list(df.columns),
        var_name="Statistic",
        value_name="Value",
    )
    if n_values_integers in [0, 1]:
        return dfm
    else:  # at least two different groups of statistics
        stat_group = dfm["Statistic"].str.split("_", n=1, expand=True)
        dfm.drop(columns=["Statistic"], inplace=True)
        dfm["Statistic"] = stat_group[0]
        dfm["Group"] = stat_group[1]
        return dfm
