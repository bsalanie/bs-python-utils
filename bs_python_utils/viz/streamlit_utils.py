"""
Contains various utility programs for Streamlit apps:

* `st_table_estimate`: to print a table of estimates with standard errors
* `st_download_numpy_as_csv`: button to download a Numpy array to a CSV file
* `st_download_dataframe_as_csv`: button to download a Pandas dataframe to a CSV file.
"""

from typing import cast

import numpy as np
import pandas as pd
import streamlit as st


def st_table_estimates(
    coeff_names: list[str],
    estimates: np.ndarray,
    stderrs: np.ndarray,
    true_coeffs: np.ndarray | None = None,
) -> pd.DataFrame:
    """returns a table of the estimates

    Args:
        coeff_names: the names of the coefficients
        estimates: the estimated values of the coefficients
        stderrs: the standard errors of the estimates
        true_coeffs: the true values of the coefficients, if available

    Returns:
        the table
    """
    st.write("The coefficients are:")
    if true_coeffs:
        df_coeffs_estimates = pd.DataFrame(
            {
                "True": true_coeffs,
                "Estimated": estimates,
                "Standard errors": stderrs,
            },
            index=coeff_names,
        )
    else:
        df_coeffs_estimates = pd.DataFrame(
            {
                "Estimated": estimates,
                "Standard errors": stderrs,
            },
            index=coeff_names,
        )

    return df_coeffs_estimates


def _convert_dataframe_to_csv(df: pd.DataFrame) -> str:
    """converts a DataFrame to a string in csv format"""
    return cast(str, df.to_csv().encode("utf-8"))


def _convert_arr_to_csv(arr: np.ndarray) -> str:
    """converts aa array to a string in csv format"""
    return _convert_dataframe_to_csv(pd.DataFrame(arr))


def st_download_numpy_as_csv(arr: np.ndarray, file_name: str) -> None:
    """button to download an array to a csv file

    Args:
        arr: a Numpy array
        file_name: the array will be downloaded in file_name.csv

    Returns:
        nothing
    """
    csv = _convert_arr_to_csv(arr)
    _ = st.download_button(
        label=f"Download the {file_name} as a CSV file",
        data=csv,
        file_name=f"{file_name}.csv",
        mime="text/csv",
    )


def st_download_dataframe_as_csv(df: pd.DataFrame, file_name: str) -> None:
    """button to download a DataFrame to a csv file

    Args:
        df: a Pandas DataFrame
        file_name: the datafrme will be downloaded in file_name.csv

    Returns:
        nothing
    """
    csv = _convert_dataframe_to_csv(df)
    _ = st.download_button(
        label=f"Download the {file_name} as a CSV file",
        data=csv,
        file_name=f"{file_name}.csv",
        mime="text/csv",
    )
