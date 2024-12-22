"""
This is a boilerplate test file for pipeline 'data_engineering'
generated using Kedro 0.19.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import logging

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from velib_prediction.pipelines.data_engineering.pipeline import (
    create_pipeline as create_de_pipeline,
)


def test_data_science_pipeline(caplog):    # Note: caplog is passed as an argument
    # Arrange pipeline
    pipeline = create_de_pipeline().from_nodes("data_engineering.merge_all_datasets").to_nodes("data_engineering.reset_values_in_boolean_columns")

    # Arrange data catalog
    catalog = DataCatalog()

    dummy_data = {
        "list_files": [
            "data/01_raw/velib_2024_10_19T16_12_14_282z.parquet",
            "data/01_raw/velib_2024_10_19T17_11_41_188z.parquet",
            "data/01_raw/velib_2024_10_19T18_15_02_973z.parquet"
        ]
    }

    dummy_parameters = {
        "path_data": "data/01_raw/",
        "cols_to_remove": ["name", "nom_arrondissement_communes"],
        "boolean_columns": ["is_installed", "is_renting", "is_returning"]
    }

    catalog.add_feed_dict(
        {
            "data_engineering.list_files": dummy_data["list_files"],
            "params:data_engineering.path_data": dummy_parameters["path_data"],
            "params:data_engineering.cols_to_remove": dummy_parameters["cols_to_remove"],
            "params:data_engineering.boolean_columns": dummy_parameters["boolean_columns"],
        }
    )

    # Arrange the log testing setup
    caplog.set_level(logging.DEBUG, logger="kedro") # Ensure all logs produced by Kedro are captured
    successful_run_msg = "Pipeline execution completed successfully."

    # Act
    SequentialRunner().run(pipeline, catalog)

    # Assert
    assert successful_run_msg in caplog.text
