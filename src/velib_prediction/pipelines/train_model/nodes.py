"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
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


def mlflow_log_metrics_cv(
    l_rmse_train: list[float], l_rmse_eval: list[float], best_iteration: int
) -> None:
    """Log metrics to MLflow

    Args:
        l_rmse_train (list[float]): Lkist of the RMSE for the train set
        l_rmse_eval (list[float]): List of the RMSE for the evaluation set
        best_iteration (int): Best iteration for Catboost model
    """
    mlflow.log_metric('rmse_mean_train', np.mean(l_rmse_train))
    mlflow.log_metric('rmse_mean_eval', np.mean(l_rmse_eval))
    mlflow.log_metric('best_iteration', best_iteration)


def train_model(
    pool_train: Pool, pool_eval: Pool, plot_training: bool, verbose: int,
    **kwargs,
) -> CatBoostRegressor:
    """Train a catboost model

    Args:
        pool_train (Pool): train pool
        pool_eval (Pool): pool used to evaluate the model
        plot_training (bool): whether to plot the leaning curves
        verbose (int): verbose parameter while learning
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model (CatBoostRegressor): Catboost regressor model trained
    """
    # Create model
    model = CatBoostRegressor(
        random_seed=12,
        **kwargs
    )
    # Fit model
    model.fit(
        pool_train,
        eval_set=pool_eval,
        use_best_model=True,
        plot=plot_training,
        verbose=verbose,
    )
    return model


def train_model_cv_mlflow(  # noqa: PLR0913
    list_train_valid: list[tuple[pd.DataFrame, pd.DataFrame]],
    feat_cat: list[str], plot_training: bool=False, verbose: int=0,
    shap_max_disp: int=10, path_reports: str='../reports',
    **kwargs
) -> tuple[CatBoostRegressor, np.array, np.array]:
    """Using cross validation, train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    Create a MLflow run
    For each split of train and validation, using expanding window validation:
    - Create the catboost pools for the catboost model
    - Define a Catboost regressor model
    - Train the model on training set and use a validation set to keep the best model
    - Predict on train and evaluation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        list_train_valid: list of the different splits of train and validation
        feat_cat: list of categorical features
        plot_training: whether to plot the leaning curves
        verbose: verbose parameter while learning
        shap_max_display: top features to display on the shap values
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model: Catboost regressor model trained
        pred_train: predictions on the train set
        pred_valid: predictions on the validation set
    """
    # Options
    l_rmse_train = []
    l_rmse_eval = []
    # MLflow
    with mlflow.start_run():
        # Run over the different folds
        for i in range(len(list_train_valid)):
            df_train_, df_eval_ = list_train_valid[i]
            y_train_, y_eval_ = df_train_[""]
            # Create Pools for catboost model
            pool_train = Pool(df_train_, y_train_, feat_cat)
            pool_eval = Pool(df_eval_, y_eval_, feat_cat)
            # Train model
            model = train_model(pool_train, pool_eval, plot_training, verbose, **kwargs)
            # Predict
            # Force predictions as integers as the number of newspapers has to be int
            pred_train = model.predict(pool_train)
            pred_eval = model.predict(pool_eval)
        # Log parameters to mlflow
        mlflow_log_metrics_cv(
            l_rmse_train, l_rmse_eval, model.get_best_iteration()
        )
        # mlflow_log_shap(model, df_train_, shap_max_disp=shap_max_disp, path_reports=path_reports)
        mlflow_log_model(model)
        mlflow_log_parameters(model)
        return model, pred_train, pred_eval

