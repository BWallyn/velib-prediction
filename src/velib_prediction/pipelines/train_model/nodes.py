"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import mlflow
import pandas as pd
from catboost import CatBoostRegressor
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
    df: pd.DataFrame, n_splits: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split the time serie dataset for cross validation using expanding window

    Args:
        df (pd.DataFrame): Input dataframe
        n_splits (int): Number of splits to create
    Returns:
        list_train_valid (list[tuple[pd.DataFrame, pd.DataFrame]]): Split dataframe using time series split
    """
    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Prepare
    list_train_valid = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, valid_index in tscv.split(df):
        list_train_valid.append((df.loc[train_index], df.loc[valid_index]))
    return list_train_valid


def mlflow_log_parameters(model: CatBoostRegressor) -> None:
    """Log the parameters of the Catboost regressor model to MLflow

    Args:
        model (CatBoostRegressor): Catboost regressor model trained
    """
    all_params = model.get_all_params()
    mlflow.log_param('depth', all_params['depth'])
    mlflow.log_param('iterations', all_params['iterations'])
    mlflow.log_param('loss_function', all_params['loss_function'])
    mlflow.log_param('learning_rate', all_params['learning_rate'])
    mlflow.log_param('l2_leaf_reg', all_params['l2_leaf_reg'])
    mlflow.log_param('random_strength', all_params['random_strength'])
    mlflow.log_param('border_count', all_params['border_count'])


def mlflow_log_model(model: CatBoostRegressor) -> None:
    """Log the Catboost regressor model to MLflow

    Args:
        model (CatBoostRegressor): Catboost regressor model trained
    """
    # model.save_model('../models/cb_classif')
    mlflow.catboost.log_model(cb_model=model, artifact_path='model')
