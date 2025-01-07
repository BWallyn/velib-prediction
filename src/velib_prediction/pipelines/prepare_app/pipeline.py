"""
This is a boilerplate pipeline 'prepare_app'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.prepare_app.nodes import (
    add_geographical_info,
    convert_to_geojson,
    extract_geo_points_by_station,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=convert_to_geojson,
                inputs="df_raw",
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
                inputs=["df_train_prepared", "list_coordinates"],
                outputs="df_train_with_coordinates",
                name="Add_geographical_info_to_train",
            ),
        ],
        inputs=["df_raw", "df_train_prepared"],
        namespace="prepare_app"
    )
