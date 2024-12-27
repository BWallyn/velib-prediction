"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import logging
from functools import partial
from typing import Any

import mlflow
import optuna
from optuna.trial import Trial
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from velib_prediction.pipelines.train_model.mlflow import (
    _create_mlflow_signature,
    _log_mlflow_catboost_parameters,
    _log_mlflow_metric,
    _log_mlflow_model_catboost,
    create_mlflow_experiment,
)

# Options
logger = logging.getLogger(__name__)

# ===================
# ==== FUNCTIONS ====
# ===================

def select_columns(df: pd.DataFrame, list_feat: list[str], target_name: str) -> pd.DataFrame:
    """Select columns of the dataframe

    Args:
        df (pd.DataFrame): Input DataFrame
        list_feat (list[str]): List of the features to select
        target_name (str): Name of the target
    Returns:
        (pd.DataFrame): Output DataFrame
    """
    return df[list_feat + [target_name]]


def add_lags_sma(  # noqa: PLR0913
    df: pd.DataFrame, list_lags: list[int], feat_id: str, feat_date: str, feat_target: str, n_shift: int,
) -> pd.DataFrame:
    """Add different lags to the dataset with a shift of n_shift

    Args:
        df (pd.DataFrame): Input dataframe
        list_lags (list[int]): List of the lags to add
        feat_id (str): Name of the id column of the velib stations
        feat_date (str): Name of the date column
        feat_target (str): Name of the target column
        n_shift (int): Number of rows to shift the results
    Returns:
        df (pd.DataFrame): Output dataframe with lags added
    """
    df = df.sort_values(by=[feat_date])
    for id in df[feat_id].unique():
        for lag in list_lags:
            df.loc[df[feat_id] == id, f'sma_{lag}_lag'] = df.loc[df[feat_id] == id].rolling(lag)[feat_target].mean().shift(n_shift).values
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


def _filter_last_hours(group, feat_date: str, n_hours: int=5):
    """
    """
    last_hours = group[feat_date].max() - pd.Timedelta(hours=n_hours)
    return group[group[feat_date] >= last_hours]


def split_train_valid_last_hours(df: pd.DataFrame, feat_date: str, n_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and validation sets

    Args:
        df (pd.DataFrame): Input DataFrame
        feat_date (str): Name of the date column
        n_hours (int): Number of hours to keep for validation set
    Returns:
        df_train (pd.DataFrame): Train DataFrame
        df_valid (pd.DataFrame): Valid DataFrame
    """
    # Order dataset by station and date
    df_sel = df.sort_values(["stationcode", feat_date], ascending=[True, True])
    df_valid_index = df_sel.groupby('stationcode').apply(_filter_last_hours, feat_date=feat_date, n_hours=n_hours).reset_index(drop=True)["idx"].values
    df_valid = df.loc[df["idx"].isin(df_valid_index)]
    df_train = df.loc[~df["idx"].isin(df_valid_index)]
    return df_train, df_valid


def create_mlflow_experiment_if_needed(
    experiment_folder_path: str | None=None, experiment_name: str | None=None, experiment_id: str | None=None
) -> str:
    """Create a MLflow experiment if needed.
    If experiment id is not None, then create a MLflow experiment using the folder path and the experiment name

    Args:
        experiment_folder_path (Optional[str]): Path to the folder to save the MLflow experiment
        experiment_name (Optional[str]): Name of the MLflow experiment to create
        experiment_id (Optional[str]): Id of the MLflow experiment if it exists
    Returns:
        experiment_id (str): Id of the MLflow experiment
    """
    if experiment_id is None:
        logger.info("Creating MLflow experiment...")
        experiment_id = create_mlflow_experiment(experiment_folder_path, experiment_name)
        logger.info("MLflow {} experiment created".format(experiment_id))  # noqa: UP032
    else:
        logger.info("{} experiment used".format(experiment_id))  # noqa: UP032
    return experiment_id



def _train_model(
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


def train_model_mlflow(  # noqa: PLR0913
    experiment_id: str,
    parent_run_id: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feat_cat: list[str],
    verbose: int=0,
    **kwargs
) -> tuple[CatBoostRegressor, float, float]:
    """Train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    - Define a Catboost regressor model
    - Train the model on training set and use a validation set to keep the best model
    - Predict on train and evaluation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        experiment_id (str): Id of the MLflow experiment
        parent_run_id (str): Id of the MLflow parent run
        df_train (pd.DataFrame): Train DataFrame
        df_valid (pd.DataFrame): Validation DataFrame
        verbose (int): verbose parameter while learning
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model: Catboost regressor model trained
    """
    # Select the target
    y_train = df_train["target"]
    y_valid = df_valid["target"]
    x_train = df_train.drop(columns=["target"])
    x_valid = df_valid.drop(columns=["target"])
    # Create pools for Catboost model
    pool_train = Pool(data=x_train, label=y_train, cat_features=feat_cat)
    pool_eval = Pool(data=x_valid, label=y_valid, cat_features=feat_cat)
    # Create MLflow child run
    with mlflow.start_run(
        experiment_id=experiment_id,
        nested=True,
        tags={mlflow.utils.mlflow_tags.MLFLOW_PARENT_RUN_ID: parent_run_id}
    ) as child_run:
        model = _train_model(pool_train, pool_eval, plot_training=False, verbose=verbose, **kwargs)
        # Predict
        pred_train = model.predict(pool_train)
        pred_eval = model.predict(pool_eval)
        # Log parameters to MLflow
        _log_mlflow_catboost_parameters(model=model)
        # Log metrics
        rmse_train = root_mean_squared_error(y_true=y_train, y_pred=pred_train)
        rmse_valid = root_mean_squared_error(y_true=y_valid, y_pred=pred_eval)
        dict_metrics = {
            'rmse_train': rmse_train,
            'rmse_valid': rmse_valid,
        }
        _log_mlflow_metric(dict_metrics=dict_metrics, run_id=child_run.info.run_id)
        # Log model
        # _log_mlflow_model_catboost(model=model, df=df_train)
    return model, rmse_train, rmse_valid


def train_model_cv_mlflow(  # noqa: PLR0913
    run_name: str,
    experiment_id: str,
    list_train_valid: list[tuple[pd.DataFrame, pd.DataFrame]],
    feat_cat: list[str],
    catboost_params: dict[str, Any],
    verbose: int=0,
) -> CatBoostRegressor:
    """Using cross validation, train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    Create a MLflow run
    For each split of train and validation, using expanding window validation:
    - Create the catboost pools for the catboost model
    - Define a Catboost regressor model
    - Train the model on training set and use a validation set to keep the best model
    - Predict on train and evaluation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        run_name (str): Name of the MLflow parent run
        experiment_id (str): Experiment id of the MLflow
        list_train_valid (list[tuple[pd.DataFrame, pd.DataFrame]]): Tuples of train and validation sets
        feat_cat (list[str]): List of the categorical features
        verbose (int): Verbose of the catboost training
    Returns:
        model (CatBoostregressor): Model trained
    """
    # MLflow
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as parent_run:
        # Iterate over the different tuples of datasets
        for i, (df_train, df_valid) in enumerate(list_train_valid):
            _, _, _ = train_model_mlflow(
                experiment_id=experiment_id,
                parent_run_id=parent_run.info.run_id,
                df_train=df_train,
                df_valid=df_valid,
                feat_cat=feat_cat,
                verbose=verbose,
                **catboost_params
            )


def _build_search_space(trial: Trial, hyperparams_search_space: dict[str, Any]):
    """
    """
    hyperparams = {}
    # Set the hyperparams
    for param_name, sampling_params in hyperparams_search_space.items():
        if sampling_params["sampling_type"] == "categorical":
            hyperparams[param_name] = eval(
                f"trial.suggest_{sampling_params['sampling_type']}('{param_name}', {sampling_params['choices']})"
            )
        else:
            hyperparams[param_name] = eval(
                f"trial.suggest_{sampling_params['sampling_type']}('{param_name}', {sampling_params['min']}, {sampling_params['max']})"
            )
    return hyperparams


def optimize_hyperparams(  # noqa: PLR0913
    trial: Trial,
    search_params: dict[str, Any],
    experiment_id: str,
    run_id: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feat_cat: list[str],
) -> float:
    """Bayesian optimization function

    Args:
        trial (): Trial for bayesian optimization
        experiment_id (str): Id of the MLflow experiment
        run_id (str): Id of the MLflow run
        search_params (dict[str, Any]): Search parameters for the hyperparameters
        df_train (pd.DataFrame): Train DataFrame
        df_valid (pd.DataFrame): Validation DataFrame
        feat_cat (list[str]): List of categorical features
    Returns:
        rmse_valid (float): Root Mean Squared Error on the validation set
    """
    optimize_params = {}
    # Set the hyperparams
    for param_name, sampling_params in search_params.items():
        optimize_params[param_name] = eval(
            f"trial.suggest_{sampling_params['sampling_type']}('{param_name}', {sampling_params['min']}, {sampling_params['max']})"
        )

    # Train model mlflow
    logger.info(f"Train Catboost model using {optimize_params}")
    model, _, rmse_valid = train_model_mlflow(
        experiment_id=experiment_id,
        parent_run_id=run_id,
        df_train=df_train,
        df_valid=df_valid,
        feat_cat=feat_cat,
        verbose=100,
        **optimize_params
    )
    logger.info(f"Catboost model trained with RMSE: {rmse_valid}")
    # Add the best iteration as an attribute
    trial.set_user_attr("best_iter", model.get_best_iteration())
    return rmse_valid


def train_model_bayesian_opti(  # noqa: PLR0913
    run_name: str,
    experiment_id: str,
    search_params: dict[str, Any],
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feat_cat: list[str],
    n_trials: int,
) -> dict[str, Any]:
    """Use Bayesian optimization to find the best hyperparameters

    Args:
        run_name (str): Name of the run
        experiment_id (str): Id of the MLflow experiment
        search_params (dict[str, Any]): Hyperparameters space to search
        df_train (pd.DataFrame): Train DataFrame
        df_valid (pd.DataFrame): Validation DataFrame
        feat_cat (list[str]): List of the categorical features
        n_trials (int): Number of trials for the bayesian optimization
    Returns:
        (dict[str, Any]): Best hyperparameters of the model
    """
    # Start a MLflow run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as parent_run:
        # Run Bayesian optimization using Optuna
        study = optuna.create_study(study_name="", direction="maximize")

        # Define objective function
        objective = partial(
            optimize_hyperparams,
            experiment_id=experiment_id,
            run_id=parent_run.info.run_id,
            search_params=search_params,
            df_train=df_train,
            df_valid=df_valid,
            feat_cat=feat_cat,
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=-1)
        logger.info(f"Best parameters found: {study.best_params}")
    # Get best parameters
    best_parameters = study.best_params
    # Set the number of iterations as the best iteration
    best_parameters["iterations"] = study.best_trial.user_attrs["best_iter"]
    return best_parameters


def train_final_model(  # noqa: PLR0913
    experiment_id: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feat_cat: list[str],
    best_parameters: dict[str, Any],
    verbose: int=0,
):
    """Train a Catboost regressor model and log the parameters, metrics, model and shap values to MLflow

    - Define a Catboost regressor model
    - Train the model on training set and use a validation set to keep the best model
    - Predict on train and evaluation sets
    Log parameters, model, metrics, shap and confusion matrix to MLflow

    Args:
        experiment_id (str): Id of the MLflow experiment
        parent_run_id (str): Id of the MLflow parent run
        df_train (pd.DataFrame): Train DataFrame
        df_valid (pd.DataFrame): Validation DataFrame
        verbose (int): verbose parameter while learning
        **kwargs: hyperparameters of the Catboost regressor
    Returns:
        model: Catboost regressor model trained
    """
    # Select the target
    y_train = df_train["target"]
    y_valid = df_valid["target"]
    x_train = df_train.drop(columns=["target"])
    x_valid = df_valid.drop(columns=["target"])
    # Create pools for Catboost model
    pool_train = Pool(data=x_train, label=y_train, cat_features=feat_cat)
    pool_eval = Pool(data=x_valid, label=y_valid, cat_features=feat_cat)
    with mlflow.start_run(experiment_id=experiment_id) as parent_run:
        # Set model
        model = _train_model(pool_train, None, plot_training=False, verbose=verbose, **best_parameters)
        # Predict
        pred_train = model.predict(pool_train)
        pred_eval = model.predict(pool_eval)
        # Log parameters to MLflow
        _log_mlflow_catboost_parameters(model=model)
        # Log metrics
        rmse_train = root_mean_squared_error(y_true=y_train, y_pred=pred_train)
        rmse_valid = root_mean_squared_error(y_true=y_valid, y_pred=pred_eval)
        dict_metrics = {
            'rmse_train': rmse_train,
            'rmse_valid': rmse_valid,
        }
        _log_mlflow_metric(dict_metrics=dict_metrics, run_id=parent_run.info.run_id)
        # Log model
        signature = _create_mlflow_signature()
        _log_mlflow_model_catboost(model=model, df=df_train, signature=signature)
    return model
