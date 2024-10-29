"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import gc
import logging
import os

import numpy as np
import pandas as pd

# Option
logger = logging.getLogger("__name__")

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
                logger.info("Dataset: {}".format(file))  # noqa: UP032
    return parquet_files


def merge_datasets(list_files: list[str]) -> pd.DataFrame:
    """Merge all datasets into one

    Args:
        list_files (list[str]): List of paths to all datasets in parquet files.
    Returns:
        df_final (pd.DataFrame): Output dataframe
    """
    df_final = pd.DataFrame()
    for file in list_files:
        df = pd.read_parquet(file)
        df_final = pd.concat([df_final, df])
        # Free memory
        del df
        gc.collect()
    return df_final


def create_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Create a specific index to each row based on the station code and the date of the data

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        df (pd.DataFrame): Output DataFrame with index column
    """
    # Set column date to datetime
    df["duedate"] = pd.to_datetime(df["duedate"])
    # Define index
    df.insert(0, "idx", df["stationcode"].astype(str) + (df["duedate"].values.astype(np.int64) // 10 ** 9).astype(str))
    return df


def create_feature_description() -> list[dict[str, str]]:
    """Create feature description for the feature store

    Args:
        None
    Returns:
        (list[dict[str, str]]): List of the features names and descriptions
    """
    return [
        {"name": "idx", "description": "Idx based on the station code and datetime as timestamp"},
        {"name": "stationcode", "description": "Code of the velib station"},
        {"name": "name", "description": "Name of the velib station"},
        {"name": "is_installed", "description": "Is the velib station available"},
        {"name": "capacity", "description": "Capacity of the velib station"},
        {"name": "numdocksavailable", "description": "Number of docks available at the velib station"},
        {"name": "numbikesavailable", "description": "Number of bikes available at the velib station"},
        {"name": "mechanical", "description": "Number of mechanical bikes available at the station"},
        {"name": "ebike", "description": "Number of ebikes available at the station"},
        {"name": "is_renting", "description": "Bikes available for renting"},
        {"name": "is_returning", "description": "Places available to return bikes"},
        {"name": "duedate", "description": "Date of the data info"},
        {"name": "coordonnees_geo", "description": "Geographical coordinates of the station"},
        {"name": "nom_arrondissement_communes", "description": "Name of the city where the station is located"},
        {"name": "code_insee_commune", "description": "Insee where the station is located"},
    ]


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
