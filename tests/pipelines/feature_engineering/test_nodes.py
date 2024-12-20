import pandas as pd
import pytest

from velib_prediction.pipelines.feature_engineering.nodes import split_train_test

# ==== split train and test =====

@pytest.fixture
def sample_dataframe():
    data = {'date': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20']),
            'value': [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)

def test_split_train_test_basic(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 10)
    assert len(df_train) == 3  # noqa: PLR2004
    assert len(df_test) == 2  # noqa: PLR2004
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
    assert len(df_train) == 3  # noqa: PLR2004
    assert len(df_test) == 2  # noqa: PLR2004

def test_split_train_test_same_date_cutoff(sample_dataframe):
    df_train, df_test = split_train_test(sample_dataframe, 'date', 5)
    assert len(df_train) == 4  # noqa: PLR2004
    assert len(df_test) == 1
    assert df_test['date'].iloc[0] == pd.to_datetime('2024-01-20')

def test_split_train_test_one_row_df():
    data = {'date': pd.to_datetime(['2024-01-01']), 'value': [1]}
    df = pd.DataFrame(data)
    df_train, df_test = split_train_test(df, 'date', 10)
    assert len(df_train) == 0
    assert len(df_test) == 1

def test_split_train_test_delta_days_is_float():
    data = {'date': pd.to_datetime(['2024-01-01', '2024-01-05']), 'value': [1,2]}
    df = pd.DataFrame(data)
    with pytest.raises(TypeError):
        split_train_test(df, 'date', 10.5)
