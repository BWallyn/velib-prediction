from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from velib_prediction.pipelines.train_model.nodes import (
    _filter_last_hours,
    add_lags_sma,
    create_mlflow_experiment_if_needed,
    get_split_train_val_cv,
    select_columns,
    split_train_valid_last_hours,
)

# ==== Select columns ====


def test_select_columns_basic():
    """Test basic functionality with simple DataFrame"""
    # Create sample DataFrame
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": ["a", "b", "c"],
            "feature3": [1.1, 2.2, 3.3],
            "target": [0, 1, 0],
        }
    )
    list_feat = ["feature1", "feature2"]
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    # Check correct columns are present
    expected_columns = ["feature1", "feature2", "target"]
    assert list(result.columns) == expected_columns
    # Check DataFrame shape
    assert result.shape == (3, 3)
    # Check data integrity
    assert (result["feature1"] == df["feature1"]).all()
    assert (result["feature2"] == df["feature2"]).all()
    assert (result["target"] == df["target"]).all()


def test_select_columns_empty_features():
    """Test with empty feature list"""
    df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
    list_feat = []
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    assert list(result.columns) == ["target"]
    assert result.shape == (2, 1)


def test_select_columns_single_feature():
    """Test with single feature"""
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]})
    list_feat = ["feature1"]
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    assert list(result.columns) == ["feature1", "target"]
    assert result.shape == (2, 2)


def test_select_columns_all_features():
    """Test selecting all features"""
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]})
    list_feat = ["feature1", "feature2"]
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    assert result.equals(df)


def test_select_columns_invalid_feature():
    """Test with non-existent feature name"""
    df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
    list_feat = ["non_existent_feature"]
    target_name = "target"
    with pytest.raises(KeyError):
        select_columns(df, list_feat, target_name)


def test_select_columns_invalid_target():
    """Test with non-existent target name"""
    df = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
    list_feat = ["feature1"]
    target_name = "non_existent_target"
    with pytest.raises(KeyError):
        select_columns(df, list_feat, target_name)


def test_select_columns_empty_dataframe():
    """Test with empty DataFrame"""
    df = pd.DataFrame(columns=["feature1", "target"])
    list_feat = ["feature1"]
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    assert result.empty
    assert list(result.columns) == ["feature1", "target"]


def test_select_columns_different_dtypes():
    """Test with different data types"""
    df = pd.DataFrame(
        {
            "int_feat": [1, 2],
            "float_feat": [1.1, 2.2],
            "str_feat": ["a", "b"],
            "bool_feat": [True, False],
            "target": [0, 1],
        }
    )
    list_feat = ["int_feat", "float_feat", "str_feat", "bool_feat"]
    target_name = "target"
    result = select_columns(df, list_feat, target_name)
    assert result.shape == (2, 5)
    assert (result.dtypes == df.dtypes).all()


# ==== Add lags ====


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(start="2024-01-01", periods=5)
    df = pd.DataFrame(
        {
            "station_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "date": dates.tolist() * 2,
            "bikes": [10, 20, 15, 25, 30, 5, 8, 12, 15, 18],
        }
    )
    return df


def test_add_lags_sma_basic(sample_df):
    """Test basic functionality with simple parameters"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    # Check if new column was created
    assert "sma_2_lag" in result.columns
    # Check if original columns are preserved
    assert all(col in result.columns for col in sample_df.columns)
    # Check if DataFrame is sorted by date
    assert result["date"].equals(result["date"].sort_values())
    # Verify SMA calculation for first station
    station_1_values = result[result["station_id"] == 1]["sma_2_lag"].tolist()
    expected_values = [np.nan, np.nan, 15.0, 17.5, 20.0]
    np.testing.assert_almost_equal(station_1_values, expected_values, decimal=2)


def test_add_lags_sma_multiple_lags(sample_df):
    """Test with multiple lag values"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2, 3],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    # Check if both lag columns were created
    assert "sma_2_lag" in result.columns
    assert "sma_3_lag" in result.columns
    # Verify calculations for both lags
    station_1 = result[result["station_id"] == 1]
    assert pd.isna(station_1["sma_2_lag"].iloc[0])  # First values should be NaN
    assert pd.isna(station_1["sma_3_lag"].iloc[0])


def test_add_lags_sma_different_shift(sample_df):
    """Test with different shift values"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=2,
    )
    # Check if shift is applied correctly
    station_1_values = result[result["station_id"] == 1]["sma_2_lag"].tolist()
    assert pd.isna(station_1_values[0])  # First values should be NaN due to shift
    assert pd.isna(station_1_values[1])


# def test_add_lags_sma_empty_lags(sample_df):
#     """Test with empty lag list"""
#     result = add_lags_sma(
#         df=sample_df.copy(),
#         list_lags=[],
#         feat_id='station_id',
#         feat_date='date',
#         feat_target='bikes',
#         n_shift=1
#     )
#     # Check if DataFrame remains unchanged
#     assert result.equals(sample_df)


def test_add_lags_sma_single_station(sample_df):
    """Test with single station"""
    single_station_df = sample_df[sample_df["station_id"] == 1].copy()
    result = add_lags_sma(
        df=single_station_df,
        list_lags=[2],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    # Check if calculations are correct for single station
    expected_values = [np.nan, np.nan, 15.0, 17.5, 20.0]
    np.testing.assert_almost_equal(
        result["sma_2_lag"].tolist(), expected_values, decimal=2
    )


def test_add_lags_sma_missing_values():
    """Test with missing values in the target column"""
    df = pd.DataFrame(
        {
            "station_id": [1, 1, 1, 1],
            "date": pd.date_range(start="2024-01-01", periods=4),
            "bikes": [10, np.nan, 15, 20],
        }
    )
    result = add_lags_sma(
        df=df.copy(),
        list_lags=[2],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    # Check if NaN values are handled correctly
    assert pd.isna(result["sma_2_lag"].iloc[0])
    assert pd.isna(result["sma_2_lag"].iloc[1])


def test_add_lags_sma_invalid_columns():
    """Test with invalid column names"""
    df = pd.DataFrame(
        {"invalid_id": [1, 2], "date": ["2024-01-01", "2024-01-02"], "bikes": [10, 20]}
    )
    with pytest.raises(KeyError):
        add_lags_sma(
            df=df,
            list_lags=[2],
            feat_id="station_id",  # Non-existent column
            feat_date="date",
            feat_target="bikes",
            n_shift=1,
        )


def test_add_lags_sma_empty_dataframe():
    """Test with empty DataFrame"""
    empty_df = pd.DataFrame(columns=["station_id", "date", "bikes"])
    result = add_lags_sma(
        df=empty_df,
        list_lags=[2],
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    assert result.empty


def test_add_lags_sma_large_lag(sample_df):
    """Test with lag larger than the group size"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[10],  # Larger than number of rows per station
        feat_id="station_id",
        feat_date="date",
        feat_target="bikes",
        n_shift=1,
    )
    # Check if large lag results in NaN values
    station_1_sma = result[result["station_id"] == 1]["sma_10_lag"]
    assert station_1_sma.isna().all()


# ==== Split train and validation ====


@pytest.fixture
def sample_df_split_cv():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame(
        {"date": pd.date_range(start="2024-01-01", periods=10), "value": range(10)}
    )


def test_basic_split_cv(sample_df_split_cv):
    """Test basic functionality with simple parameters"""
    n_splits = 3
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits)
    # Check correct number of splits
    assert len(result) == n_splits
    # Check each split is a tuple of DataFrames
    for train, val in result:
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
    # Check that splits are expanding (train set grows)
    train_sizes = [len(train) for train, _ in result]
    assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))
    # Check validation sets are same size (except possibly last one)
    val_sizes = [len(val) for _, val in result]
    assert (
        len(set(val_sizes[:-1])) == 1
    )  # All validation sets except last should be same size


def test_index_reset(sample_df_split_cv):
    """Test that index is properly reset"""
    # Add random index
    df_with_index = sample_df_split_cv.copy()
    df_with_index.index = range(100, 110)
    result = get_split_train_val_cv(df_with_index, n_splits=3)
    # Check that indices in resulting splits start from 0
    for train, val in result:
        assert train.index[0] == 0
        assert val.index[0] > 0
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing


def test_data_integrity_cv(sample_df_split_cv):
    """Test that no data is lost or duplicated in the splits"""
    n_splits = 3
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits)
    # Check last split contains all data
    final_train, final_val = result[-1]
    total_rows = len(final_train) + len(final_val)
    assert total_rows == len(sample_df_split_cv)
    # Check data values are preserved
    combined_last_split = pd.concat([final_train, final_val])
    assert combined_last_split["value"].sum() == sample_df_split_cv["value"].sum()


def test_chronological_order(sample_df_split_cv):
    """Test that splits maintain chronological order"""
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits=3)
    for train, val in result:
        # Check train dates are before validation dates
        assert train["date"].max() < val["date"].min()
        # Check dates are ordered within each split
        assert train["date"].is_monotonic_increasing
        assert val["date"].is_monotonic_increasing


def test_single_split(sample_df_split_cv):
    """Test with single split"""
    with pytest.raises(ValueError):
        get_split_train_val_cv(sample_df_split_cv.copy(), n_splits=1)


def test_max_splits(sample_df_split_cv):
    """Test with maximum possible number of splits"""
    n_splits = len(sample_df_split_cv) - 1  # Maximum possible splits
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits)
    assert len(result) == n_splits
    for train, val in result:
        assert len(train) >= 1
        assert len(val) >= 1


def test_empty_dataframe_cv():
    """Test with empty DataFrame"""
    empty_df = pd.DataFrame(columns=["date", "value"])
    with pytest.raises(ValueError):
        get_split_train_val_cv(empty_df, n_splits=3)


def test_invalid_splits():
    """Test with invalid number of splits"""
    df = pd.DataFrame({"value": range(5)})
    # Test with n_splits = 0
    with pytest.raises(ValueError):
        get_split_train_val_cv(df, n_splits=0)
    # Test with n_splits > len(df)
    with pytest.raises(ValueError):
        get_split_train_val_cv(df, n_splits=10)


def test_small_dataframe():
    """Test with minimal size DataFrame"""
    small_df = pd.DataFrame(
        {"date": pd.date_range(start="2024-01-01", periods=2), "value": range(2)}
    )
    with pytest.raises(ValueError):
        get_split_train_val_cv(small_df.copy(), n_splits=3)


def test_column_preservation_cv(sample_df_split_cv):
    """Test that all columns are preserved in splits"""
    # Add extra columns
    df = sample_df_split_cv.copy()
    df["category"] = "A"
    df["extra"] = 1.0
    result = get_split_train_val_cv(df, n_splits=3)
    for train, val in result:
        assert all(col in train.columns for col in df.columns)
        assert all(col in val.columns for col in df.columns)


# ==== Split last hours ====


@pytest.fixture
def sample_group():
    """Create a sample DataFrame group for testing"""
    dates = pd.date_range(
        start="2024-01-01 00:00:00", end="2024-01-01 23:00:00", freq="h"
    )
    return pd.DataFrame({"timestamp": dates, "value": range(len(dates))})


def test_basic_filtering(sample_group):
    """Test basic functionality with default hours"""
    result = _filter_last_hours(sample_group.copy(), feat_date="timestamp")
    # Check if result has correct number of hours (5 hours + last hour = 6 entries)
    assert len(result) == 6  # noqa: PLR2004
    # Check if filtered dates are within last 5 hours
    max_date = result["timestamp"].max()
    min_date = result["timestamp"].min()
    assert (max_date - min_date) <= pd.Timedelta(hours=5)


def test_custom_hours(sample_group):
    """Test with custom number of hours"""
    n_hours = 3
    result = _filter_last_hours(
        sample_group.copy(), feat_date="timestamp", n_hours=n_hours
    )
    # Check if result has correct number of hours
    assert len(result) == 4  # 3 hours + last hour  # noqa: PLR2004
    # Verify time span
    time_span = result["timestamp"].max() - result["timestamp"].min()
    assert time_span <= pd.Timedelta(hours=n_hours)


def test_zero_hours_cv(sample_group):
    """Test with zero hours"""
    result = _filter_last_hours(sample_group.copy(), feat_date="timestamp", n_hours=0)
    # Should only return the last timestamp
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == sample_group["timestamp"].max()


def test_empty_group():
    """Test with empty DataFrame"""
    empty_group = pd.DataFrame(columns=["timestamp", "value"])
    result = _filter_last_hours(empty_group, feat_date="timestamp")
    assert len(result) == 0
    assert result.empty


def test_single_timestamp():
    """Test with single timestamp"""
    single_time = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-01 00:00:00")], "value": [1]}
    )
    result = _filter_last_hours(single_time, feat_date="timestamp")
    assert len(result) == 1
    assert result.equals(single_time)


def test_irregular_timestamps_cv(sample_group):
    """Test with irregular time intervals"""
    irregular_times = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 18:00:00",
                "2024-01-01 18:30:00",
                "2024-01-01 19:15:00",
                "2024-01-01 20:45:00",
                "2024-01-01 23:00:00",
            ],
            "value": range(5),
        }
    )
    irregular_times["timestamp"] = pd.to_datetime(irregular_times["timestamp"])
    result = _filter_last_hours(irregular_times, feat_date="timestamp", n_hours=3)
    # Check if timestamps within last 3 hours are included
    last_time = irregular_times["timestamp"].max()
    expected_min_time = last_time - pd.Timedelta(hours=3)
    assert all(result["timestamp"] >= expected_min_time)


def test_exact_boundary():
    """Test with timestamps exactly on the boundary"""
    times = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 18:00:00",
                "2024-01-01 20:00:00",  # Exactly 3 hours before last
                "2024-01-01 23:00:00",
            ],
            "value": range(3),
        }
    )
    times["timestamp"] = pd.to_datetime(times["timestamp"])
    result = _filter_last_hours(times, feat_date="timestamp", n_hours=3)
    # Should include boundary timestamp
    assert len(result) == 2  # noqa: PLR2004
    assert min(result["timestamp"]) == pd.to_datetime("2024-01-01 20:00:00")


def test_different_column_name():
    """Test with different date column name"""
    df = pd.DataFrame(
        {
            "date_col": pd.date_range(
                start="2024-01-01 20:00:00", end="2024-01-01 23:00:00", freq="h"
            ),
            "value": range(4),
        }
    )
    result = _filter_last_hours(df, feat_date="date_col", n_hours=2)
    assert len(result) == 3  # Last 2 hours + last entry  # noqa: PLR2004
    assert (result["date_col"].max() - result["date_col"].min()) <= pd.Timedelta(
        hours=2
    )


def test_preserve_other_columns():
    """Test that non-datetime columns are preserved"""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2024-01-01 20:00:00", end="2024-01-01 23:00:00", freq="h"
            ),
            "value": range(4),
            "category": ["A", "B", "C", "D"],
            "metric": [1.1, 2.2, 3.3, 4.4],
        }
    )
    result = _filter_last_hours(df, feat_date="timestamp", n_hours=2)
    # Check all columns are preserved
    assert all(col in result.columns for col in df.columns)
    # Check data integrity
    assert all(result["value"] == df.iloc[-3:]["value"])
    assert all(result["category"] == df.iloc[-3:]["category"])
    assert all(result["metric"] == df.iloc[-3:]["metric"])


# ==== Split last hours ====


@pytest.fixture
def sample_df_split():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(
        start="2024-01-01 00:00:00", end="2024-01-01 23:00:00", freq="h"
    )
    # Create DataFrame with two stations
    df = pd.DataFrame(
        {
            "timestamp": dates.tolist() * 2,
            "stationcode": [1] * len(dates) + [2] * len(dates),
            "value": list(range(len(dates))) * 2,
        }
    )
    # Add index column
    df["idx"] = range(len(df))
    return df.sort_values(["stationcode", "timestamp"])


def test_basic_split(sample_df_split):
    """Test basic functionality with simple parameters"""
    n_hours = 5
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=n_hours
    )
    # Check that all data is preserved
    assert len(df_train) + len(df_valid) == len(sample_df_split)
    # Check that each station has correct number of validation hours
    for station in sample_df_split["stationcode"].unique():
        station_valid = df_valid[df_valid["stationcode"] == station]
        max_time = station_valid["timestamp"].max()
        min_time = station_valid["timestamp"].min()
        assert (max_time - min_time) <= pd.Timedelta(hours=n_hours)


def test_data_ordering(sample_df_split):
    """Test that data remains properly ordered"""
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=5
    )
    # Check ordering in train set
    assert df_train.groupby("stationcode")["timestamp"].is_monotonic_increasing.all()
    # Check ordering in validation set
    assert df_valid.groupby("stationcode")["timestamp"].is_monotonic_increasing.all()


def test_temporal_split(sample_df_split):
    """Test that temporal split is correct"""
    n_hours = 5
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=n_hours
    )
    # For each station, check that validation data comes after training data
    for station in sample_df_split["stationcode"].unique():
        train_max = df_train[df_train["stationcode"] == station]["timestamp"].max()
        valid_min = df_valid[df_valid["stationcode"] == station]["timestamp"].min()
        assert train_max < valid_min


def test_zero_hours(sample_df_split):
    """Test with zero validation hours"""
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=0
    )
    # Validation set should only contain last timestamp for each station
    assert len(df_valid) == len(sample_df_split["stationcode"].unique())
    # Train set should contain all other data
    assert len(df_train) == len(sample_df_split) - len(
        sample_df_split["stationcode"].unique()
    )


def test_all_hours(sample_df_split):
    """Test with validation hours larger than available data"""
    n_hours = 24  # More hours than available in the data
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=n_hours
    )
    # All data should be in validation set
    assert len(df_valid) == len(sample_df_split)
    assert len(df_train) == 0


def test_single_station():
    """Test with single station"""
    dates = pd.date_range(
        start="2024-01-01 00:00:00", end="2024-01-01 23:00:00", freq="h"
    )
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "stationcode": [1] * len(dates),
            "value": range(len(dates)),
            "idx": range(len(dates)),
        }
    )
    df_train, df_valid = split_train_valid_last_hours(
        df, feat_date="timestamp", n_hours=5
    )
    # Check correct split
    assert len(df_valid) == 6  # 5 hours plus last hour  # noqa: PLR2004
    assert len(df_train) == len(df) - 6


def test_irregular_timestamps():
    """Test with irregular time intervals"""
    df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 18:00:00",
                "2024-01-01 19:30:00",
                "2024-01-01 20:45:00",
                "2024-01-01 23:00:00",
            ]
            * 2,
            "stationcode": [1, 1, 1, 1, 2, 2, 2, 2],
            "value": range(8),
            "idx": range(8),
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_train, df_valid = split_train_valid_last_hours(
        df, feat_date="timestamp", n_hours=3
    )
    # Check that irregular timestamps are handled correctly
    for station in df["stationcode"].unique():
        valid_times = df_valid[df_valid["stationcode"] == station]["timestamp"]
        time_span = valid_times.max() - valid_times.min()
        assert time_span <= pd.Timedelta(hours=3)


def test_empty_dataframe():
    """Test with empty DataFrame"""
    empty_df = pd.DataFrame(columns=["timestamp", "stationcode", "value", "idx"])
    df_train, df_valid = split_train_valid_last_hours(
        empty_df, feat_date="timestamp", n_hours=5
    )
    assert df_train.empty
    assert df_valid.empty


def test_column_preservation(sample_df_split):
    """Test that all columns are preserved"""
    # Add extra columns
    df = sample_df_split.copy()
    df["category"] = "A"
    df["extra"] = 1.0
    df_train, df_valid = split_train_valid_last_hours(
        df, feat_date="timestamp", n_hours=5
    )
    # Check all columns are preserved in both splits
    assert all(col in df_train.columns for col in df.columns)
    assert all(col in df_valid.columns for col in df.columns)


def test_data_integrity(sample_df_split):
    """Test that no data is lost or duplicated"""
    df_train, df_valid = split_train_valid_last_hours(
        sample_df_split.copy(), feat_date="timestamp", n_hours=5
    )
    # Check no duplicates in either set
    assert not df_train["idx"].duplicated().any()
    assert not df_valid["idx"].duplicated().any()
    # Check no overlap between sets
    assert not set(df_train["idx"]).intersection(set(df_valid["idx"]))
    # Check all data is accounted for
    assert set(df_train["idx"]).union(set(df_valid["idx"])) == set(
        sample_df_split["idx"]
    )


# ==== Create MLflow experiment ====


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_mlflow():
    """Mock MLflow functions"""
    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.create_experiment") as mock_create,
    ):
        mock_create.return_value = "12092276"
        yield mock_create


def test_create_new_experiment(mock_logger, mock_mlflow):
    """Test creating a new experiment when no experiment_id is provided"""
    experiment_folder = "/path/to/experiment"
    experiment_name = "test_experiment"
    result = create_mlflow_experiment_if_needed(
        experiment_folder_path=experiment_folder, experiment_name=experiment_name
    )
    # Check if experiment was created
    mock_mlflow.assert_called_once_with(
        name=experiment_name, artifact_location=f"file://{experiment_folder}"
    )
    # Check logger calls
    # TODO: fix logger calls
    # assert mock_logger.info.call_args_list == [
    #     call("Creating MLflow experiment..."),
    #     call("MLflow 12092276 experiment created")
    # ]
    # Check return value
    assert isinstance(result, str)
    assert result == "12092276"


# def test_use_existing_experiment(mock_logger):
#     """Test using existing experiment when experiment_id is provided"""
#     existing_experiment_id = "existing_id"
#     result = create_mlflow_experiment_if_needed(
#         experiment_id=existing_experiment_id
#     )
#     # Check if correct log was created
#     mock_logger.info.assert_called_once_with("existing_id experiment used")
#     # Check return value
#     assert result == existing_experiment_id

# def test_create_experiment_with_all_params(mock_logger, mock_mlflow):
#     """Test creating experiment with all parameters provided"""
#     experiment_folder = "/path/to/experiment"
#     experiment_name = "test_experiment"
#     experiment_id = None
#     result = create_mlflow_experiment_if_needed(
#         experiment_folder_path=experiment_folder,
#         experiment_name=experiment_name,
#         experiment_id=experiment_id
#     )
#     # Check if experiment was created
#     mock_mlflow.assert_called_once_with(
#         experiment_name,
#         artifact_location=experiment_folder
#     )
#     # Check return value
#     assert result == "test_experiment_id"

# def test_none_parameters(mock_logger, mock_mlflow):
#     """Test behavior with None parameters"""
#     result = create_mlflow_experiment_if_needed(
#         experiment_folder_path=None,
#         experiment_name=None,
#         experiment_id=None
#     )
#     # Check if experiment was created with None values
#     mock_mlflow.assert_called_once_with(
#         None,
#         artifact_location=None
#     )
#     assert result == "test_experiment_id"

# @pytest.mark.integration
# def test_integration_with_mlflow():
#     """Integration test with actual MLflow"""
#     # Set up temporary tracking URI
#     temp_dir = "temp_mlflow"
#     os.makedirs(temp_dir, exist_ok=True)
#     mlflow.set_tracking_uri(f"file://{temp_dir}")
#     try:
#         # Test creating new experiment
#         experiment_name = "test_integration"
#         result = create_mlflow_experiment_if_needed(
#             experiment_folder_path=temp_dir,
#             experiment_name=experiment_name
#         )
#         # Verify experiment exists
#         experiment = mlflow.get_experiment(result)
#         assert experiment.name == experiment_name
#     finally:
#         # Clean up
#         import shutil
#         shutil.rmtree(temp_dir)

# def test_experiment_creation_error(mock_logger, mock_mlflow):
#     """Test handling of experiment creation error"""
#     mock_mlflow.side_effect = Exception("Creation failed")
#     with pytest.raises(Exception) as exc_info:
#         create_mlflow_experiment_if_needed(
#             experiment_folder_path="/path",
#             experiment_name="test"
#         )
#     assert str(exc_info.value) == "Creation failed"
#     mock_logger.info.assert_called_once_with("Creating MLflow experiment...")

# @pytest.mark.parametrize("folder_path,name", [
#     ("/valid/path", "valid_name"),
#     ("/path/with/spaces", "name with spaces"),
#     ("/path/with/特殊字符", "特殊字符"),
#     ("", ""),
# ])
# def test_various_input_formats(folder_path, name, mock_logger, mock_mlflow):
#     """Test different input formats for folder path and experiment name"""
#     result = create_mlflow_experiment_if_needed(
#         experiment_folder_path=folder_path,
#         experiment_name=name
#     )
#     mock_mlflow.assert_called_once_with(
#         name,
#         artifact_location=folder_path
#     )
#     assert result == "test_experiment_id"

# def test_experiment_id_precedence(mock_logger, mock_mlflow):
#     """Test that experiment_id takes precedence over other parameters"""
#     result = create_mlflow_experiment_if_needed(
#         experiment_folder_path="/path",
#         experiment_name="test",
#         experiment_id="existing_id"
#     )
#     # Check that create_experiment was not called
#     mock_mlflow.assert_not_called()
#     # Check that existing ID was used
#     assert result == "existing_id"
