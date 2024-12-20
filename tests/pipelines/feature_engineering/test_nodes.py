from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from velib_prediction.pipelines.feature_engineering.nodes import (
    drop_columns,
    extract_date_features,
    # get_holidays,
    get_weekend,
    split_train_test,
)

# ==== split train and test =====

@pytest.fixture
def sample_dataframe():
    data = {'date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20']),
            'value': [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)

def test_split_train_test_basic(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 10)
    assert len(df_train) == 2  # noqa: PLR2004
    assert len(df_test) == 3  # noqa: PLR2004
    assert df_train['date'].max() < pd.to_datetime('2024-01-10')
    assert df_test['date'].min() >= pd.to_datetime('2024-01-10')

def test_split_train_test_empty_test(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 0)
    assert len(df_train) == 4  # noqa: PLR2004
    assert len(df_test) == 1
    assert df_test['date'].iloc[0] == sample_dataframe['date'].max()

def test_split_train_test_empty_train(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 30)
    assert len(df_train) == 0
    assert len(df_test) == 5  # noqa: PLR2004

def test_split_train_test_different_date_column_name(sample_dataframe):
    sample_dataframe = sample_dataframe.rename(columns={'date': 'my_date'})
    df_train, df_test = split_train_test(sample_dataframe, 'my_date', 10)
    assert len(df_train) == 2  # noqa: PLR2004
    assert len(df_test) == 3  # noqa: PLR2004

def test_split_train_test_same_date_cutoff(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 5)
    assert len(df_train) == 3  # noqa: PLR2004
    assert len(df_test) == 2  # noqa: PLR2004
    assert df_test['date'].iloc[0] == pd.to_datetime('2024-01-15')

def test_split_train_test_one_row_df():
    data = {'date': pd.to_datetime(['2024-01-01']), 'value': [1]}
    df = pd.DataFrame(data)
    df_train, df_test = split_train_test(df, 'date', 10)
    assert len(df_train) == 0
    assert len(df_test) == 1

def test_split_train_test_delta_days_is_float():
    data = {'date': pd.to_datetime(['2024-01-01', '2024-01-05']), 'value': [1,2]}
    df = pd.DataFrame(data)
    with pytest.raises(AssertionError):
        split_train_test(df, 'date', 10.5)


# ==== Get holidays =====

# Mock pandas.read_csv to avoid downloading actual data
@pytest.fixture
def mock_read_csv():
    with patch("pandas.read_csv") as mock_csv:
        mock_csv.return_value = MagicMock()  # Simulate a DataFrame
        yield mock_csv

# TODO: fix the test
# def test_get_holidays(mock_read_csv):
#     df_holidays = get_holidays()

#     # Assert mocked function was called with the correct URL
#     mock_read_csv.assert_called_once_with(
#         "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
#     )

#     # Assert basic data manipulation steps are performed (specific assertions depend on your data format)
#     assert mock_read_csv.return_value.sort_values.called  # Assert sorting by "Date de début"

#     # You can further assert on specific data cleaning steps
#     # e.g., assert number of rows after filtering zones
#     # assert len(df_holidays) == X (expected number after filtering)

#     # Assert date format conversion
#     assert pd.api.types.is_datetime64_dtype(df_holidays["date_begin"])
#     assert pd.api.types.is_datetime64_dtype(df_holidays["date_end"])

#     # Assert year filtering
#     assert df_holidays["date_end"].dt.year.min() >= 2020  # noqa: PLR2004


# ==== Merge holidays ====

# TODO: Add tests for the merge holidays function


# ==== Get weekend ====

@pytest.fixture
def sample_dataframe_weekend():
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08']) #Mon, Tue, Wed, Thu, Fri, Sat, Sun, Mon
    df = pd.DataFrame({'date': dates})
    df['date_weekday'] = df['date'].dt.weekday
    return df

def test_get_weekend_basic(sample_dataframe_weekend):
    df_with_weekend = get_weekend(sample_dataframe_weekend.copy(), 'date')  # Important: copy the DataFrame
    expected_weekend = np.array([0, 0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(df_with_weekend['date_weekend'].values, expected_weekend)

def test_get_weekend_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        get_weekend(df, 'date')

def test_get_weekend_no_weekday_column(sample_dataframe_weekend):
    df = sample_dataframe_weekend.drop(columns='date_weekday')
    with pytest.raises(KeyError):
        get_weekend(df, 'date')

def test_get_weekend_different_date_column_name(sample_dataframe_weekend):
    df = sample_dataframe_weekend.rename(columns={'date': 'my_date'})
    df['my_date_weekday'] = df['my_date'].dt.weekday
    df_with_weekend = get_weekend(df.copy(), 'my_date')
    expected_weekend = np.array([0, 0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(df_with_weekend['my_date_weekend'].values, expected_weekend)

def test_get_weekend_already_existing_column(sample_dataframe_weekend):
    sample_dataframe_weekend['date_weekend'] = 999
    df_with_weekend = get_weekend(sample_dataframe_weekend.copy(), 'date')
    expected_weekend = np.array([0, 0, 0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(df_with_weekend['date_weekend'].values, expected_weekend)


# ==== Extract date features ====

@pytest.fixture
def sample_dataframe_date():
    data = {'date_str': ['20231215', '20231218', '20240102']}
    return pd.DataFrame(data)

def test_extract_date_features(sample_dataframe_date):
    df_with_features = extract_date_features(sample_dataframe_date.copy(), 'date_str')

    # Assert presence of new features
    assert 'date_str_year' in df_with_features.columns
    assert 'date_str_month' in df_with_features.columns
    assert 'date_str_day' in df_with_features.columns
    assert 'date_str_weekday' in df_with_features.columns
    assert 'date_str_weekend' in df_with_features.columns

    # Assert data types (optional)
    assert pd.api.types.is_datetime64_dtype(df_with_features['date_str'])
    assert df_with_features['date_str_year'].dtype == np.int32
    assert df_with_features['date_str_month'].dtype == np.int32
    assert df_with_features['date_str_day'].dtype == np.int32
    assert df_with_features['date_str_weekday'].dtype == np.int32
    assert df_with_features['date_str_weekend'].dtype == np.int64

    # Assert specific feature values (optional)
    assert df_with_features['date_str_year'].iloc[0] == 2023  # noqa: PLR2004
    assert df_with_features['date_str_month'].iloc[1] == 12  # noqa: PLR2004
    assert df_with_features['date_str_weekday'].iloc[2] == 1  # Tuesday (assuming Tuesday is weekday 1)


# ==== Drop columns ====

@pytest.fixture
def sample_dataframe_drop():
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
    return pd.DataFrame(data)

def test_drop_columns_basic(sample_dataframe_drop):
    df_dropped = drop_columns(sample_dataframe_drop.copy(), ['col1', 'col3'])
    assert list(df_dropped.columns) == ['col2']
    assert len(df_dropped) == 3  # noqa: PLR2004

def test_drop_columns_empty_list(sample_dataframe_drop):
    df_dropped = drop_columns(sample_dataframe_drop.copy(), [])
    assert list(df_dropped.columns) == ['col1', 'col2', 'col3']
    assert len(df_dropped) == 3  # noqa: PLR2004

def test_drop_columns_all_columns(sample_dataframe_drop):
    df_dropped = drop_columns(sample_dataframe_drop.copy(), ['col1', 'col2', 'col3'])
    assert df_dropped.empty

# TODO: fix test
# def test_drop_columns_nonexistent_column(sample_dataframe_drop):
#     with pytest.raises(ValueError, match="KeyError: \"['col4'] not found in axis\""): #Check the content of the message
#         drop_columns(sample_dataframe_drop.copy(), ['col4'])

# TODO: fix test
# def test_drop_columns_mix_existing_nonexistent(sample_dataframe_drop):
#     with pytest.raises(ValueError, match="KeyError: \"['col3', 'col4'] not found in axis\""): #Check the content of the message
#         drop_columns(sample_dataframe_drop.copy(), ['col2', 'col3', 'col4'])

def test_drop_columns_duplicate_columns_to_drop(sample_dataframe_drop):
    df_dropped = drop_columns(sample_dataframe_drop.copy(), ['col1', 'col1'])
    assert list(df_dropped.columns) == ['col2', 'col3']
    assert len(df_dropped) == 3  # noqa: PLR2004

def test_drop_columns_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        drop_columns(df, ['col1'])
