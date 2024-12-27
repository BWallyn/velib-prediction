"""
This is a boilerplate test file for pipeline 'train_model'
generated using Kedro 0.19.7.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import logging

import pandas as pd
import pytest
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner

from velib_prediction.pipelines.train_model.pipeline import (
    create_pipeline as create_tm_pipeline,
)


@pytest.fixture
def sample_dataframe_train():
    return pd.read_parquet("tests/data/df_train.parquet")

def test_data_science_pipeline(caplog, sample_dataframe_train: pd.DataFrame):    # Note: caplog is passed as an argument
    # Arrange pipeline
    pipeline = (
        create_tm_pipeline()
        .from_nodes("train_model.Rename_target_column")
        .to_nodes("train_model.Select_columns_train", "train_model.Select_columns_valid")
    )

    # Arrange data catalog
    catalog = DataCatalog()

    dummy_data = {
        "df_train_prepared": sample_dataframe_train,
    }

    dummy_parameters = {
        "dict_rename_target": {"numbikesavailable": "target"},
        "feat_date": "duedate",
        "n_hours": 24,
        "feat_target": "target",
        "model_features": [
            "stationcode",
            "is_installed",
            "capacity",
            "numdocksavailable",
            "mechanical",
            "ebike",
            "is_renting",
            "is_returning",
            "code_insee_commune",
            "duedate_year",
            "duedate_month",
            "duedate_day",
            "duedate_weekday",
            "duedate_weekend",
        ],
    }

    catalog.add_feed_dict(
        {
            "df_train_prepared": dummy_data["df_train_prepared"],
            "params:train_model.dict_rename_target": dummy_parameters["dict_rename_target"],
            "params:train_model.feat_date": dummy_parameters["feat_date"],
            "params:train_model.n_hours": dummy_parameters["n_hours"],
            "params:train_model.feat_target": dummy_parameters["feat_target"],
            "params:train_model.model_features": dummy_parameters["model_features"],
        }
    )

    # Arrange the log testing setup
    caplog.set_level(logging.DEBUG, logger="kedro") # Ensure all logs produced by Kedro are captured
    successful_run_msg = "Pipeline execution completed successfully."

    # Act
    SequentialRunner().run(pipeline, catalog)

    # Assert
    assert successful_run_msg in caplog.text
