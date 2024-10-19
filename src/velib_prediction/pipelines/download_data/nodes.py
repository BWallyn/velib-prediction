"""
This is a boilerplate pipeline 'download_data'
generated using Kedro 0.19.8
"""
# =================
# ==== IMPORTS ====
# =================

import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests

# Options
logger = logging.getLogger(__name__)

# ===================
# ==== FUNCTIONS ====
# ===================

def generate_timestamp() -> str:
    """Generate the timestamp to be used by versionning data
    """
    current_ts = datetime.now(tz=timezone.utc).strftime("%Y_%m_%dT%H_%M_%S_%fz")
    return current_ts[:-4] + current_ts[-1:] # Don't keep microseconds


def download_data(url: str) -> pd.DataFrame:
    """Download data from a website

    Args:
        url (str): Path to the dataset
    Returns:
        df (pd.DataFrame): Dataset downloaded and saved as a pandas DataFrame
    """
    response = requests.get(url)
    if response.status_code != 200:  # noqa: PLR2004
        logger.error("Error retriving data")
    df = pd.DataFrame(response.json()['results'])
    return df


def save_data(df: pd.DataFrame, path_data: str) -> None:
    """Save dataset as a parquet file

    Args:
        df (pd.DataFrame): Input dataframe
        path_data (str): Path to the folder to save file
    Returns:
        None
    """
    timestamp = generate_timestamp()
    logger.info('Saving dataset...')
    df.to_parquet(os.path.join(path_data, f"velib_{timestamp}.parquet"))
    logger.info('Dataset saved')
