"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

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


def update_values_bool_columns(df: pd.DataFrame, list_bool_cols: list[str]) -> pd.DataFrame:
    """Update the values for the boolean columns

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        df (pd.DataFrame): Output dataframe
    """
    values_replace = {'OUI': True, 'NON': False}
    dict_replace = {el: values_replace for el in list_bool_cols}
    return df.replace(dict_replace)
