"""
This is a boilerplate pipeline 'prepare_app'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.prepare_app.nodes import (
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
        ]
    )
