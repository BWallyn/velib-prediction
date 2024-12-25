"""
This is a boilerplate test file for pipeline 'feature_engineering'
generated using Kedro 0.19.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import logging

import pandas as pd
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from velib_prediction.pipelines.feature_engineering.pipeline import (
    create_pipeline as create_fe_pipeline,
)


def test_data_science_pipeline(caplog):    # Note: caplog is passed as an argument
    # Arrange pipeline
    pipeline = create_fe_pipeline().from_nodes("feature_engineering.split_train_test").to_nodes("feature_engineering.Drop_unused_columns")

    # Arrange data catalog
    catalog = DataCatalog()

    dummy_data = {
        "df_with_bool_cols_upd": pd.read_parquet("tests/data/df_data_engineered.parquet")
    }

    dummy_parameters = {
        "feat_date": "duedate",
        "delta_days": 2,
        "cols_to_drop": ["coordonnees_geo", "date"]
    }

    catalog.add_feed_dict(
        {
            "df_with_bool_cols_upd": dummy_data["df_with_bool_cols_upd"],
            "params:feature_engineering.feat_date": dummy_parameters["feat_date"],
            "params:feature_engineering.delta_days": dummy_parameters["delta_days"],
            "params:feature_engineering.cols_to_drop": dummy_parameters["cols_to_drop"],
        }
    )

    # Arrange the log testing setup
    caplog.set_level(logging.DEBUG, logger="kedro") # Ensure all logs produced by Kedro are captured
    successful_run_msg = "Pipeline execution completed successfully."

    # Act
    SequentialRunner().run(pipeline, catalog)

    # Assert
    assert successful_run_msg in caplog.text
