"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ===================
# ==== FUNCTIONS ====
# ===================

def add_lags_sma(  # noqa: PLR0913
    df: pd.DataFrame, list_lags: list[int], feat_id: str, feat_date: str, feat_target: str, shift_days: int,
) -> pd.DataFrame:
    """Add different lags to the dataset with a shift of shift_days.

    Args:
        df (pd.DataFrame): Input dataframe
        list_lags (list[int]): List of the lags to add
        feat_id (str): Name of the id column of the velib stations
        feat_date (str): Name of the date column
        feat_target (str): Name of the target column
        shift_days (int): Number of days to shift the results
    Returns:
        df (pd.DataFrame): Output dataframe with lags added
    """
    df = df.sort_values(by=[feat_date])
    for id in df[feat_id].unique():
        for lag in list_lags:
            df.loc[df[feat_id] == id, f'sma_{lag}_lag'] = df.loc[df[feat_id] == id].rolling(lag)[feat_target].mean().shift(shift_days).values
    return df


def get_split_train_val_cv(
    df: pd.DataFrame, target: pd.Series, n_splits: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split the time serie dataset for cross validation using expanding window

    Args:
        df (pd.DataFrame): Input dataframe
        target (pd.Series): Target
        n_splits (int): Number of splits to create
    Returns:
        list_train_valid (list[tuple[pd.DataFrame, pd.DataFrame]]): Split dataframe using time series split
    """
    list_train_valid = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, valid_index in tscv.split(df):
        list_train_valid.append((df.loc[train_index], df.loc[valid_index]))
    return list_train_valid
