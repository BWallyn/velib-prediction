"""
This is a boilerplate pipeline 'prepare_app'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.prepare_app.nodes import (
    add_geographical_info,
    convert_to_geojson,
    extract_geo_points_by_station,
    extract_lat_lon,
    prepare_data_to_plot_predictions,
)
from velib_prediction.utils.utils import save_geojson


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=extract_lat_lon,
                inputs="df_raw",
                outputs="df_w_lat_lon",
                name="Extract_latitudes_and_longitudes",
            ),
            node(
                func=convert_to_geojson,
                inputs="df_w_lat_lon",
                outputs="gdf_with_coordinates",
                name="Convert_to_geodataframe"
            ),
            node(
                func=extract_geo_points_by_station,
                inputs="gdf_with_coordinates",
                outputs="list_coordinates",
                name="Extract_list_of_station_coordinates"
            ),
            node(
                func=add_geographical_info,
                inputs=["df_training_feat_engineered", "list_coordinates"],
                outputs="df_train_with_coordinates",
                name="Add_geographical_info_to_train",
            ),
            node(
                func=save_geojson,
                inputs=["df_train_with_coordinates", "params:path_save_gdf"],
                outputs=None,
                name="Save_geodataframe_to_geojson",
            ),
            node(
                func=prepare_data_to_plot_predictions,
                inputs=["model_velib", "df_training_sorted", "df_test_sorted"],
                outputs="df_pred_to_plot",
                name="Prepare_data_to_plot_predictions",
            )
        ],
        inputs=["df_raw", "df_training_feat_engineered", "df_training_sorted", "df_test_sorted", "model_velib"],
        outputs=["df_w_lat_lon", "df_train_with_coordinates", "df_pred_to_plot"],
        namespace="prepare_app"
    )
