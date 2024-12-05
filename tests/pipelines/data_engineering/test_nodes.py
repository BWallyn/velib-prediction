import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

from velib_prediction.pipelines.data_engineering.nodes import (
    create_idx,
    list_parquet_files,
    merge_datasets,
)


def test_list_parquet_files_empty_directory():
    """
    Test that the function returns an empty list when no parquet files exist.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the logger to prevent actual logging
        with mock.patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock.Mock()
            mock_get_logger.return_value = mock_logger

            result = list_parquet_files(temp_dir)
            assert result == [], "Function should return an empty list for an empty directory"

def test_list_parquet_files_single_level():
    """
    Test listing parquet files in a single-level directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        parquet_files = [
            os.path.join(temp_dir, 'test1.parquet'),
            os.path.join(temp_dir, 'test2.parquet')
        ]
        non_parquet_files = [
            os.path.join(temp_dir, 'test.txt'),
            os.path.join(temp_dir, 'test.csv')
        ]

        # Create the files
        for file_path in parquet_files + non_parquet_files:
            open(file_path, 'w').close()

        # Mock the logger
        with mock.patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock.Mock()
            mock_get_logger.return_value = mock_logger

            # Call the function
            result = list_parquet_files(temp_dir)

            # Assert that only parquet files are returned
            assert set(result) == set(parquet_files), "Function should only return .parquet files"
            assert len(result) == 2, "Should find exactly 2 parquet files"  # noqa: PLR2004

def test_list_parquet_files_nested_directory():
    """
    Test listing parquet files in a nested directory structure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory structure with parquet files
        nested_dir1 = os.path.join(temp_dir, 'subdir1')
        nested_dir2 = os.path.join(temp_dir, 'subdir2')
        os.makedirs(nested_dir1)
        os.makedirs(nested_dir2)

        # Create parquet files in different directories
        parquet_files = [
            os.path.join(temp_dir, 'test1.parquet'),
            os.path.join(nested_dir1, 'test2.parquet'),
            os.path.join(nested_dir2, 'test3.parquet')
        ]
        non_parquet_files = [
            os.path.join(temp_dir, 'test.txt'),
            os.path.join(nested_dir1, 'test.csv')
        ]

        # Create the files
        for file_path in parquet_files + non_parquet_files:
            open(file_path, 'w').close()

        # Mock the logger
        with mock.patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock.Mock()
            mock_get_logger.return_value = mock_logger

            # Call the function
            result = list_parquet_files(temp_dir)

            # Assert that all parquet files are found, including those in subdirectories
            assert set(result) == set(parquet_files), "Function should find all .parquet files in nested directories"
            assert len(result) == 3, "Should find exactly 3 parquet files"  # noqa: PLR2004

def test_list_parquet_files_invalid_path():
    """
    Test behavior with an invalid directory path.
    Assumes the function returns an empty list instead of raising an error.
    """
    # Mock the logger
    with mock.patch('logging.getLogger') as mock_get_logger:
        mock_logger = mock.Mock()
        mock_get_logger.return_value = mock_logger
        # Check that an empty list is returned for an invalid path
        result = list_parquet_files('/path/that/does/not/exist')
        assert result == [], "Function should return an empty list for an invalid path"


@pytest.fixture
def sample_dataframes():
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    return [df1, df2]

@pytest.fixture
def mock_parquet_files(tmp_path, sample_dataframes):
    file_paths = []
    for i, df in enumerate(sample_dataframes):
        file_path = tmp_path / f"data_{i}.parquet"
        df.to_parquet(file_path)
        file_paths.append(str(file_path))
    return file_paths

def test_merge_datasets(mock_parquet_files, sample_dataframes):
    expected_df = pd.concat(sample_dataframes)
    actual_df = merge_datasets(mock_parquet_files)

    pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "stationcode": ["A", "B", "A"],
        "duedate": ["2023-11-22", "2023-11-23", "2023-11-22"]
    })

def test_create_idx(sample_df):
    # Call the function
    result_df = create_idx(sample_df)

    # Assertions
    # 1. Check if the index column is added
    assert "idx" in result_df.columns

    # 2. Check if the index is created correctly
    expected_df = pd.DataFrame({
        "idx": ["A1700611200", "B1700697600", "A1700611200"],
        "stationcode": ["A", "B", "A"],
        "duedate": ["2023-11-22", "2023-11-23", "2023-11-22"]
    })
    expected_df["duedate"] = pd.to_datetime(expected_df["duedate"])
    pd.testing.assert_frame_equal(result_df, expected_df)

    # 3. Check if the `duedate` column is converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(result_df["duedate"])
