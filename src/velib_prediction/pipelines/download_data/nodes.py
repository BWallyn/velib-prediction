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
URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records?refine=nom_arrondissement_communes%3A%22Paris%22&refine=nom_arrondissement_communes%3A%22Levallois-Perret%22&refine=nom_arrondissement_communes%3A%22Puteaux%22&refine=nom_arrondissement_communes%3A%22Suresnes%22&refine=nom_arrondissement_communes%3A%22Boulogne-Billancourt%22&refine=nom_arrondissement_communes%3A%22Clichy%22"


# ===================
# ==== FUNCTIONS ====
# ===================

def generate_timestamp() -> str:
    """Generate the timestamp to be used by versionning data
    """
    current_ts = datetime.now(tz=timezone.utc).strftime("%Y_%m_%dT%H_%M_%S_%fz")
    return current_ts[:-4] + current_ts[-1:] # Don't keep microseconds


def download_data(url: str) -> pd.DataFrame:
    """
    """
    response = requests.get(url)
    if response.status_code != 200:  # noqa: PLR2004
        logging.error("Error retriving data")
    df = pd.DataFrame(response.json()['results'])
    return df


def save_data(df: pd.DataFrame, path_data: str) -> None:
    """
    """
    timestamp = generate_timestamp()
    df.to_parquet(os.path.join(path_data, f"velib_{timestamp}.parquet"))
