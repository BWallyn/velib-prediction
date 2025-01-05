"""
This is a boilerplate pipeline 'prepare_app'
generated using Kedro 0.19.10
"""
# =================
# ==== IMPORTS ====
# =================

import geopandas as gpd
import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def convert_to_geojson(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert the dataframe to a geojson using the coordinates column.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        (gpd.GeoDataFrame): Output GeoDataFrame containing latitude and longitude
    """
    # Split into lat and lon
    df[['lat', 'lon']] = pd.DataFrame(df['coordonnees_geo'].tolist())
    # Drop the geocoordinates
    df = df.drop("coordonnees_geo", axis=1)
    # Transform to geopandas dataframe
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )


def extract_geo_points_by_station(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract the geographical points for each Velib station

    Args:
        gdf (gpd.GeoDataFrame): Input DataFrame
    Returns:
        (gpd.GeoDataFrame): Output DataFrame
    """
    return gdf.drop_duplicates(subset=["stationcode"])[["stationcode", "name", "coordonnees_geo"]]
