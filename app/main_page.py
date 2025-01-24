# =================
# ==== IMPORTS ====
# =================

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Options
sns.set_style("whitegrid")

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


def _create_header() -> None:
    """Create header of the app

    Args:
        None
    Returns:
        None
    """
    st.title("Velib Data Analysis")
    st.image("reports/images/velib-velo-electrique.jpeg", caption="Electrical velib")
    st.write("""
        This app is used to analyze the Velib dataset. The goal is to predict the number of available bikes at a given station in the next 24 hours.
    """)


def _display_stations(station_coordinates: gpd.GeoDataFrame) -> None:
    """Display the Velib stations on a map

    Args:
        station_coordinates (gpd.GeoDataFrame): GeoDataFrame containing the coordinates of the Velib stations
    Returns:
        None
    """
    st.subheader("Display Velib stations")
    st.map(station_coordinates, latitude="lat", longitude="lon", size="capacity")


def _create_selectbox(df: pd.DataFrame, column: str) -> str:
    """Create a selectbox to choose a station.

    Args:
        df (pd.DataFrame): DataFrame containing the station names
        column (str): Column containing the station names
    Returns:
        (str): Selected station
    """
    return st.selectbox("Select the station", df[column].unique())


def _plot_predictions(df: pd.DataFrame, station_name: str) -> None:
    """Plot the predictions

    Args:
        df (pd.DataFrame): DataFrame containing the predictions
        station_name (str): Name of the station to plot
    Returns:
        None
    """
    # Filter the station
    df_station = df[df["name"] == station_name]
    df_station_training = df_station[df_station["dataset"] == "training"]
    df_station_test = df_station[df_station["dataset"] == "test"]
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_station_training["duedate"], y=df_station_training["target"], mode="lines+markers", name="Available bikes")
    )
    fig.add_trace(
        go.Scatter(x=df_station_test["duedate"], y=df_station_test["target"], mode="lines+markers", name="Available bikes")
    )
    fig.add_trace(
        go.Scatter(x=df_station_test["duedate"], y=df_station_test["pred"], mode="lines+markers", name="Predictions")
    )
    # Edit the layout
    fig.update_layout(
        title=dict(text=f'Number of available bikes at station {station_name}'),
        xaxis=dict(title=dict(text='Date')),
        yaxis=dict(title=dict(text='Number of available bikes')),
    )
    # Plot the predictions
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


# Main function to run the app
def main():
    """Main function to run the app

    Args:
        None
    Returns:
        None
    """
    # Set header
    _create_header()

    # Load data
    # df_train = _load_data('data/04_feature/df_feat_train.parquet')

    # Load geo data
    list_stations = _load_data("data/08_reporting/station_locations.parquet")
    # Display velib stations
    _display_stations(list_stations)

    # Display prediction ov available bikes
    st.subheader("Predictions for a specific station")
    station_name = _create_selectbox(list_stations, "name")
    df_pred = _load_data("data/08_reporting/predictions_to_plot.parquet")
    _plot_predictions(df_pred, station_name)

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
