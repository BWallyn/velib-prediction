"""
This is a boilerplate pipeline 'prepare_app'
generated using Kedro 0.19.10
"""
# =================
# ==== IMPORTS ====
# =================

import geopandas as gpd
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# ===================
# ==== FUNCTIONS ====
# ===================

def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the latitude and longitude from the coordinates column.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        (pd.DataFrame): Output DataFrame containing latitude and longitude
    """
    # Split into lat and lon
    df[['lat', 'lon']] = pd.DataFrame(df['coordonnees_geo'].tolist())
    # Drop the geocoordinates
    return df.drop("coordonnees_geo", axis=1)


def convert_to_geojson(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert the dataframe to a geojson using the coordinates column.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        (gpd.GeoDataFrame): Output GeoDataFrame containing latitude and longitude
    """
    # Transform to geopandas dataframe
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )


def extract_geo_points_by_station(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract the geographical points for each Velib station

    Args:
        gdf (gpd.GeoDataFrame): Input DataFrame
    Returns:
        (gpd.GeoDataFrame): Output DataFrame
    """
    return gdf.drop_duplicates(subset=["stationcode"])[["stationcode", "name", "geometry"]]


def add_geographical_info(df: pd.DataFrame, location_stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add geographical information of the Velib stations to the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        location_stations (gpd.GeoDataFrame): GeoDataFrame containing the location of the stations
    Returns:
        (gpd.GeoDataFrame): Output GeoDataFrame
    """
    df_coord = df.merge(location_stations, how="left", on="stationcode")
    return gpd.GeoDataFrame(df_coord, geometry="geometry")


def _model_predict(model: CatBoostRegressor, df: pd.DataFrame) -> np.array:
    """Predict the target using the trained model.

    Args:
        model (CatboostRegressor): Trained model
        df (pd.DataFrame): Data to predict
    Returns:
        (np.array): Predictions
    """
    return model.predict(df[model.feature_names_])


def prepare_data_to_plot_predictions(model: CatBoostRegressor, df_training: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Prepare data to plot predictions.

    Args:
        model (CatboostRegressor): Trained model
        df_training (pd.DataFrame): Training data
        df_test (pd.DataFrame): Test data
    Returns:
        (pd.DataFrame): DataFrame of training and test containing the predictions
    """
    # Predictions
    df_training["pred"] = _model_predict(model, df_training)
    df_test["pred"] = _model_predict(model, df_test)
    # Add the dataset info
    df_training["dataset"] = "training"
    df_test["dataset"] = "test"
    # Concatenate the data
    return pd.concat([df_training, df_test], axis=0)
