"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import os

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def list_parquet_files(path: str) -> list[str]:
    """Lists all parquet files in a given directory.

    Args:
        path (str): The folder to search for parquet files.

    Returns:
        parquet_files (list[str]): A list of parquet file paths.
    """

    parquet_files = []
    for root, _, files in os.walk(path):
        for file in files:
            # Check all the files in the folder
            if file.endswith('.parquet'):
                # Keep only the parquet files
                parquet_files.append(os.path.join(root, file))
    return parquet_files


def drop_unused_columns(df: pd.DataFrame, list_cols_to_remove: list[str]) -> pd.DataFrame:
    """Drop the unnecessary columns for the model

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        (pd.DataFrame): Output dataframe
    """
    return df.drop(columns=list_cols_to_remove)


def set_date_format(df: pd.DataFrame) -> pd.DataFrame:
    """Set the date format to the right type

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        df (pd.DataFrame): Output dataframe
    """
    df["duedate"] = pd.to_datetime(df["duedate"], format="%Y-%m-%d %H:%M:%S")
    return df


def add_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    df['date'] = df["duedate"].dt.date
    return df


def update_values_bool_columns(df: pd.DataFrame, list_bool_cols: list[str]) -> pd.DataFrame:
    """Update the values for the boolean columns

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        df (pd.DataFrame): Output dataframe
    """
    values_replace = {'OUI': 1, 'NON': 0}
    dict_replace = {el: values_replace for el in list_bool_cols}
    return df.replace(dict_replace)
