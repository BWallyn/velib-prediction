"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.10
"""
# =================
# ==== IMPORTS ====
# =================

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Options
logger = logging.getLogger(__name__)


# ===================
# ==== FUNCTIONS ====
# ===================


def model_predict(model: CatBoostRegressor, df: pd.DataFrame) -> np.array:
    """Predict the target using the trained model.

    Args:
        model (CatboostRegressor): Trained model
        df (pd.DataFrame): Data to predict
    Returns:
        (np.array): Predictions
    """
    return model.predict(df[model.feature_names_])
