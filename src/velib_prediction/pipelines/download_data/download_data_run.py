# =================
# ==== IMPORTS ====
# =================

import logging
import os
from datetime import UTC, datetime

import pandas as pd
import requests

# Options
logger = logging.getLogger(__name__)


# ===================
# ==== FUNCTIONS ====
# ===================


def generate_timestamp() -> str:
    """Generate the timestamp to be used by versionning data"""
    current_ts = datetime.now(tz=UTC).strftime("%Y_%m_%dT%H_%M_%S_%fz")
    return current_ts[:-4] + current_ts[-1:]  # Don't keep microseconds


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
    df = pd.DataFrame(response.json()["results"])
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
    logger.info("Saving dataset...")
    df.to_parquet(os.path.join(path_data, f"velib_{timestamp}.parquet"))
    logger.info("Dataset saved")


def main(url: str, path_data: str) -> None:
    """Download and save data

    Args:
        url (str): Path to the dataset online
        path_data (str): Path to the folder where to store the dataset downloaded
    Returns:
        None
    """
    df = download_data(url)
    save_data(df, path_data=path_data)


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    # Options
    URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records?refine=nom_arrondissement_communes%3A%22Paris%22&refine=nom_arrondissement_communes%3A%22Levallois-Perret%22&refine=nom_arrondissement_communes%3A%22Puteaux%22&refine=nom_arrondissement_communes%3A%22Suresnes%22&refine=nom_arrondissement_communes%3A%22Boulogne-Billancourt%22&refine=nom_arrondissement_communes%3A%22Clichy%22"
    PATH_DATA = "./data/01_raw/"

    # Run
    main(url=URL, path_data=PATH_DATA)
