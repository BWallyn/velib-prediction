# =================
# ==== IMPORTS ====
# =================

import geopandas as gpd
import pandas as pd
import streamlit as st

# ===================
# ==== FUNCTIONS ====
# ===================

# Load dataset
@st.cache_data
def _load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data
def _load_geo_data(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _create_header():
    """
    """
    st.title("Velib Data Analysis")
    st.image("reports/images/velib-velo-electrique.jpeg", caption="Electrical velib")
    st.write("""
        This app is used to analyze the Velib dataset. The goal is to predict the number of available bikes at a given station in the next 24 hours.
    """)


def _display_stations(station_coordinates: gpd.GeoDataFrame):
    """
    """
    st.subheader("Display Velib stations")
    st.map(station_coordinates, latitude="lat", longitude="lon", size="capacity")


# Main function to run the app
def main():
    # Set header
    _create_header()

    # Load data
    # df_train = _load_data('data/04_feature/df_feat_train.parquet')

    # Load geo data
    list_stations = _load_data("data/08_reporting/station_locations.parquet")
    # Display velib stations
    _display_stations(list_stations)
    st.write(list_stations)

    # # Display dataset
    # st.subheader("Dataset")
    # st.write(df_train)

    # # Display basic statistics
    # st.subheader("Basic Statistics")
    # st.write(df_train.describe())

    # # Display data types
    # st.subheader("Data Types")
    # st.write(df_train.dtypes)

    # # Display number of rows and columns
    # st.subheader("Number of Rows and Columns")
    # st.write(f"Rows: {df_train.shape[0]}, Columns: {df_train.shape[1]}")


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    main()
