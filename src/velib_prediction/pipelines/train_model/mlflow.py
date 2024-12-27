# =================
# ==== IMPORTS ====
# =================

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

# Options
VERSION_FORMAT = "%Y_%m_%dT%H_%M_%S_%fZ"

# ===================
# ==== FUNCTIONS ====
# ===================

def generate_timestamp() -> str:
    """Generate timestamp to be used by versionning

    Args:
        None
    Returns:
        (str): String representation of the current timestamp
    """
    current_ts = datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)
    return current_ts[:-4] + current_ts[-1:] # Dont keep microseconds


def create_mlflow_experiment(
    experiment_folder_path: str, experiment_name: str,
) -> str:
    """Create a MLflow experiment

    Args:
        experiment_folder_path (str): Path to the MLflow experiment folder
        experiment_name (str): Name of the experiment
    Returns:
        (str): Id of the created experiment
    """
    # Create MLflow experiment
    experiment_id = mlflow.create_experiment(
        name=f"{experiment_name}",
        artifact_location=Path.cwd().joinpath(experiment_folder_path).as_uri(),
    )
    return experiment_id

def _create_mlflow_signature() -> ModelSignature:
    """Create a MLflow signature for the model

    Args:
        None
    Returns:
        (ModelSignature): MLflow signature for the model
    """
    input_schema = Schema(
        [
            ColSpec("string", "stationcode"),
            ColSpec("integer", "is_installed"),
            ColSpec("integer", "capacity"),
            ColSpec("integer", "numdocksavailable"),
            ColSpec("integer", "mechanical"),
            ColSpec("integer", "ebike"),
            ColSpec("integer", "is_renting"),
            ColSpec("integer", "is_returning"),
            ColSpec("string", "code_insee_commune"),
            ColSpec("integer", "duedate_year"),
            ColSpec("integer", "duedate_month"),
            ColSpec("integer", "duedate_day"),
            ColSpec("integer", "duedate_weekday"),
            ColSpec("integer", "duedate_weekend"),
            ColSpec("float", "sma_1_lag", required=False),
        ]
    )
    output_schema = Schema([ColSpec("integer", "target")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)

def _log_mlflow_model_catboost(
    model: CatBoostRegressor, df: pd.DataFrame, signature: ModelSignature,
) -> None:
    """Log model to MLflow

    Args:
        model (CatBoostRegressor): CatBoostRegressor model trained
        df (pd.DataFrame): DataFrame to use as example
        signature (ModelSignature): Signature of the model to log to MLflow
    Returns:
        None
    """
    mlflow.catboost.log_model(
        model,
        "model",
        signature=signature,
        input_example=df[model.feature_names_].sample(10, random_state=42),
    )


def _log_mlflow_metric(dict_metrics: dict[str, Any], run_id: str) -> None:
    """Log metrics to MLflow

    Args:
        dict_metrics (dict[str, Any]): Dict containing metrics
        run_id (str): Id of the MLflow run
    Returns:
        None
    """
    for metric_name, metric_value in dict_metrics.items():
        mlflow.log_metric(metric_name, metric_value, run_id=run_id)


def _log_mlflow_parameters(dict_params: dict[str, Any]) -> None:
    """Log parameters to MLflow

    Args:
        dict_params (dict[str, Any]): Dict containing parameters of the model
    Returns:
        None
    """
    mlflow.log_params(dict_params)


def _log_mlflow_catboost_parameters(model: CatBoostRegressor) -> None:
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


def _log_fig_in_artifacts(fig: plt.figure, figure_path: str) -> None:
    """Log figures in artifacts to MLflow

    Args:
        fig (str): Figure to log to MLflow
        figure_path (str): Path to the figure
    Returns:
        None
    """
    mlflow.log_figure(fig, figure_path)
