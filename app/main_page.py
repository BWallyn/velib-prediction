# =================
# ==== IMPORTS ====
# =================

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yaml

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


def _set_parameters() -> None:
    """Set the parameters for the app
    """
    st.set_page_config(layout="wide")


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


def _plot_capacity_stations(df: pd.DataFrame) -> None:
    """Plot the capacity of stations on a histogram plot

    Args:
        df (pd.DataFrame): Input DataFrame containing the stations coded and names and capacity
    Returns:
        None
    """
    df_plot = df.drop_duplicates(subset=["stationcode"])[["name", "capacity"]]
    fig = px.bar(df_plot, x="name", y="capacity", title="Capacity of Velib stations")
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def _display_stations(station_coordinates: gpd.GeoDataFrame) -> None:
    """Display the Velib stations on a map

    Args:
        station_coordinates (gpd.GeoDataFrame): GeoDataFrame containing the coordinates of the Velib stations
    Returns:
        None
    """
    st.subheader("Display Velib stations")
    # Get mapbox token
    with open("./conf/local/credentials.yml") as file:
        data = yaml.safe_load(file)
    token = data["mapbox"]["token"]
    # Get unique row by station
    station_coordinates = station_coordinates.drop_duplicates(subset=["stationcode"])
    # Create plot
    fig = px.scatter_mapbox(
        station_coordinates,
        lat="lat",
        lon="lon",
        color="capacity",
        size="capacity",
        color_continuous_scale=px.colors.cyclical.IceFire,
    )
    # Edit the layout
    fig.update_layout(
        title_text='Velib stations in Paris',
        showlegend=True,
        width=800,
        height=800,
    )
    fig.update_layout(
        mapbox_style="light",
        mapbox_accesstoken=token,
        mapbox_zoom=10,
        mapbox_center={"lat": 48.85, "lon": 2.33},
    )
    # Plot the map
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def _create_selectbox(df: pd.DataFrame, column: str) -> str:
    """Create a selectbox to choose a station.

    Args:
        df (pd.DataFrame): DataFrame containing the station names
        column (str): Column containing the station names
    Returns:
        (str): Selected station
    """
    return st.selectbox("Select the station", df[column].unique())


def _plot_bikes_type_over_time(df: pd.DataFrame, station_name: str) -> None:
    """Plot the number of each type of bikes over time for a specific station

    Args:
        df (pd.DataFrame): Input DetaFrame containing the number of bikes over time
        station_name (str): Name of the station to plot
    Returns:
        None
    """
    # Filter the station
    df_station = df[df["name"] == station_name]
    # Create the plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_station["duedate"],
            y=df_station["mechanical"],
            hoverinfo="x+y",
            mode="lines+markers",
            name="Available mechanical bikes",
            stackgroup="one"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_station["duedate"],
            y=df_station["ebike"],
            hoverinfo="x+y",
            mode="lines+markers",
            name="Available electrical bikes",
            stackgroup="one"
        )
    )
    # Edit the layout
    fig.update_layout(
        title=dict(text=f'Number of available types of bikes at station {station_name}'),
        xaxis=dict(title=dict(text='Date')),
        yaxis=dict(title=dict(text='Number of available bikes')),
    )
    # Plot the predictions
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


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


def _create_sidebar() -> None:
    """Create a sidebar for the app.

    Args:
        None
    Returns:
        None
    """
    st.sidebar.header("About")
    st.sidebar.markdown(
        "This app is used to analyze the Velib dataset. The goal is to predict the number of available bikes at a given station in the next 24 hours."
    )


# Main function to run the app
def main():
    """Main function to run the app

    Args:
        None
    Returns:
        None
    """
    # Set parameters of the app
    _set_parameters()

    # Set header
    _create_header()

    # Set sidebar
    _create_sidebar()

    # Load data
    # df_train = _load_data('data/04_feature/df_feat_train.parquet')

    # Load geo data
    list_stations = _load_data("data/08_reporting/station_locations.parquet")
    # Display velib stations
    _display_stations(list_stations)

    # Load predictions
    df_pred = _load_data("data/08_reporting/predictions_to_plot.parquet")
    # Display the capacity
    _plot_capacity_stations(df_pred)

    # Create selection box to select the station to display info
    st.subheader("Analyze a specific station")
    station_name = _create_selectbox(list_stations, "name")

    # Display the number of available bikes
    _plot_bikes_type_over_time(df_pred, station_name)
    # Display prediction ov available bikes
    _plot_predictions(df_pred, station_name)


# =============
# ==== RUN ====
# =============

if __name__ == "__main__":
    main()
