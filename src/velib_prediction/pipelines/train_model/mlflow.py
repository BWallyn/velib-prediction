# =================
# ==== IMPORTS ====
# =================

from datetime import datetime, timezone

import mlflow
import pandas as pd

from catboost import CatBoostRegressor

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
    # Generate timestamp for the MLflow logging
    timestamp = generate_timestamp()

    # Create MLflow experiment
    experiment_path = f"{experiment_folder_path}{experiment_name}_{timestamp}"
    experiment_id = mlflow.create_experiment(experiment_path)
    return experiment_id


def _log_mlflow_model_catboost(
    model: CatBoostRegressor, df: pd.DataFrame
) -> None:
    """Log model to MLflow

    Args:
        model (CatBoostRegressor): CatBoostRegressor model trained
        df (pd.DataFrame): DataFrame to use as example
    Returns:
        None
    """
    mlflow.catboost.log_model(
        model,
        "model",
        input_example=df[model.feature_names_].sample(10, random_state=42)
    )


def _log_mlflow_metric(dict_metric: str, run_id: str) -> None:
    """Log metrics to MLflow
    """
    for metric_name, metric_value in dict_metric.items():
        mlflow.log_metric(metric_name, metric_value, run_id=run_id)
