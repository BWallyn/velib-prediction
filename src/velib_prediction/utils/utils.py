# =================
# ==== IMPORTS ====
# =================

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def rename_columns(df: pd.DataFrame, dict_to_rename: dict[str, str]) -> pd.DataFrame:
    """Rename columns of dataset

    Args:
        df (pd.DataFrame): Input DataFrame
        dict_to_rename (dict[str, str]): Dict of the columns to rename
    Returns:
        (pd.DataFrame): Output DataFrame
    """
    return df.rename(columns=dict_to_rename)


def sort_dataframe(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    """Sort dataset based on columns

    Args:
        df (pd.DataFrame): Input DataFrame
        list_cols (list[str]): List of the columns used to sort DataFrame
    Returns:
        (pd.DataFrame): Output DataFrame sorted
    """
    return df.sort_values(by=list_cols)


def drop_columns(df: pd.DataFrame, list_cols: list[str]) -> pd.DataFrame:
    """Drop columns

    Args:
        df (pd.DataFrame): Input DataFrame
        list_cols (list[str]): List of the columns to drop
    Returns:
        (pd.DataFrame): Output DataFrame
    """
    return df.drop(columns=list_cols)
