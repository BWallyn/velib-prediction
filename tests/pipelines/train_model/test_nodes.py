import numpy as np
import pandas as pd
import pytest

from velib_prediction.pipelines.train_model.nodes import (
    add_lags_sma,
    get_split_train_val_cv,
    select_columns,
)

# ==== Select columns ====

def test_select_columns_basic():
    """Test basic functionality with simple DataFrame"""
    # Create sample DataFrame
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'feature3': [1.1, 2.2, 3.3],
        'target': [0, 1, 0]
    })
    list_feat = ['feature1', 'feature2']
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    # Check correct columns are present
    expected_columns = ['feature1', 'feature2', 'target']
    assert list(result.columns) == expected_columns
    # Check DataFrame shape
    assert result.shape == (3, 3)
    # Check data integrity
    assert (result['feature1'] == df['feature1']).all()
    assert (result['feature2'] == df['feature2']).all()
    assert (result['target'] == df['target']).all()

def test_select_columns_empty_features():
    """Test with empty feature list"""
    df = pd.DataFrame({
        'feature1': [1, 2],
        'target': [0, 1]
    })
    list_feat = []
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    assert list(result.columns) == ['target']
    assert result.shape == (2, 1)

def test_select_columns_single_feature():
    """Test with single feature"""
    df = pd.DataFrame({
        'feature1': [1, 2],
        'feature2': [3, 4],
        'target': [0, 1]
    })
    list_feat = ['feature1']
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    assert list(result.columns) == ['feature1', 'target']
    assert result.shape == (2, 2)

def test_select_columns_all_features():
    """Test selecting all features"""
    df = pd.DataFrame({
        'feature1': [1, 2],
        'feature2': [3, 4],
        'target': [0, 1]
    })
    list_feat = ['feature1', 'feature2']
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    assert result.equals(df)

def test_select_columns_invalid_feature():
    """Test with non-existent feature name"""
    df = pd.DataFrame({
        'feature1': [1, 2],
        'target': [0, 1]
    })
    list_feat = ['non_existent_feature']
    target_name = 'target'
    with pytest.raises(KeyError):
        select_columns(df, list_feat, target_name)

def test_select_columns_invalid_target():
    """Test with non-existent target name"""
    df = pd.DataFrame({
        'feature1': [1, 2],
        'target': [0, 1]
    })
    list_feat = ['feature1']
    target_name = 'non_existent_target'
    with pytest.raises(KeyError):
        select_columns(df, list_feat, target_name)

def test_select_columns_empty_dataframe():
    """Test with empty DataFrame"""
    df = pd.DataFrame(columns=['feature1', 'target'])
    list_feat = ['feature1']
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    assert result.empty
    assert list(result.columns) == ['feature1', 'target']

def test_select_columns_different_dtypes():
    """Test with different data types"""
    df = pd.DataFrame({
        'int_feat': [1, 2],
        'float_feat': [1.1, 2.2],
        'str_feat': ['a', 'b'],
        'bool_feat': [True, False],
        'target': [0, 1]
    })
    list_feat = ['int_feat', 'float_feat', 'str_feat', 'bool_feat']
    target_name = 'target'
    result = select_columns(df, list_feat, target_name)
    assert result.shape == (2, 5)
    assert (result.dtypes == df.dtypes).all()


# ==== Add lags ====

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(start='2024-01-01', periods=5)
    df = pd.DataFrame({
        'station_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'date': dates.tolist() * 2,
        'bikes': [10, 20, 15, 25, 30, 5, 8, 12, 15, 18]
    })
    return df

def test_add_lags_sma_basic(sample_df):
    """Test basic functionality with simple parameters"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    # Check if new column was created
    assert 'sma_2_lag' in result.columns
    # Check if original columns are preserved
    assert all(col in result.columns for col in sample_df.columns)
    # Check if DataFrame is sorted by date
    assert result['date'].equals(result['date'].sort_values())
    # Verify SMA calculation for first station
    station_1_values = result[result['station_id'] == 1]['sma_2_lag'].tolist()
    expected_values = [np.nan, np.nan, 15.0, 17.5, 20.0]
    np.testing.assert_almost_equal(station_1_values, expected_values, decimal=2)

def test_add_lags_sma_multiple_lags(sample_df):
    """Test with multiple lag values"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2, 3],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    # Check if both lag columns were created
    assert 'sma_2_lag' in result.columns
    assert 'sma_3_lag' in result.columns
    # Verify calculations for both lags
    station_1 = result[result['station_id'] == 1]
    assert pd.isna(station_1['sma_2_lag'].iloc[0])  # First values should be NaN
    assert pd.isna(station_1['sma_3_lag'].iloc[0])

def test_add_lags_sma_different_shift(sample_df):
    """Test with different shift values"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[2],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=2
    )
    # Check if shift is applied correctly
    station_1_values = result[result['station_id'] == 1]['sma_2_lag'].tolist()
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
    single_station_df = sample_df[sample_df['station_id'] == 1].copy()
    result = add_lags_sma(
        df=single_station_df,
        list_lags=[2],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    # Check if calculations are correct for single station
    expected_values = [np.nan, np.nan, 15.0, 17.5, 20.0]
    np.testing.assert_almost_equal(
        result['sma_2_lag'].tolist(),
        expected_values,
        decimal=2
    )

def test_add_lags_sma_missing_values():
    """Test with missing values in the target column"""
    df = pd.DataFrame({
        'station_id': [1, 1, 1, 1],
        'date': pd.date_range(start='2024-01-01', periods=4),
        'bikes': [10, np.nan, 15, 20]
    })
    result = add_lags_sma(
        df=df.copy(),
        list_lags=[2],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    # Check if NaN values are handled correctly
    assert pd.isna(result['sma_2_lag'].iloc[0])
    assert pd.isna(result['sma_2_lag'].iloc[1])

def test_add_lags_sma_invalid_columns():
    """Test with invalid column names"""
    df = pd.DataFrame({
        'invalid_id': [1, 2],
        'date': ['2024-01-01', '2024-01-02'],
        'bikes': [10, 20]
    })
    with pytest.raises(KeyError):
        add_lags_sma(
            df=df,
            list_lags=[2],
            feat_id='station_id',  # Non-existent column
            feat_date='date',
            feat_target='bikes',
            n_shift=1
        )

def test_add_lags_sma_empty_dataframe():
    """Test with empty DataFrame"""
    empty_df = pd.DataFrame(columns=['station_id', 'date', 'bikes'])
    result = add_lags_sma(
        df=empty_df,
        list_lags=[2],
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    assert result.empty

def test_add_lags_sma_large_lag(sample_df):
    """Test with lag larger than the group size"""
    result = add_lags_sma(
        df=sample_df.copy(),
        list_lags=[10],  # Larger than number of rows per station
        feat_id='station_id',
        feat_date='date',
        feat_target='bikes',
        n_shift=1
    )
    # Check if large lag results in NaN values
    station_1_sma = result[result['station_id'] == 1]['sma_10_lag']
    assert station_1_sma.isna().all()


# ==== Split train and validation ====

@pytest.fixture
def sample_df_split_cv():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10),
        'value': range(10)
    })

def test_basic_split(sample_df_split_cv):
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
    assert all(train_sizes[i] < train_sizes[i+1] for i in range(len(train_sizes)-1))
    # Check validation sets are same size (except possibly last one)
    val_sizes = [len(val) for _, val in result]
    assert len(set(val_sizes[:-1])) == 1  # All validation sets except last should be same size

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

def test_data_integrity(sample_df_split_cv):
    """Test that no data is lost or duplicated in the splits"""
    n_splits = 3
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits)
    # Check last split contains all data
    final_train, final_val = result[-1]
    total_rows = len(final_train) + len(final_val)
    assert total_rows == len(sample_df_split_cv)
    # Check data values are preserved
    combined_last_split = pd.concat([final_train, final_val])
    assert combined_last_split['value'].sum() == sample_df_split_cv['value'].sum()

def test_chronological_order(sample_df_split_cv):
    """Test that splits maintain chronological order"""
    result = get_split_train_val_cv(sample_df_split_cv.copy(), n_splits=3)
    for train, val in result:
        # Check train dates are before validation dates
        assert train['date'].max() < val['date'].min()
        # Check dates are ordered within each split
        assert train['date'].is_monotonic_increasing
        assert val['date'].is_monotonic_increasing

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

def test_empty_dataframe():
    """Test with empty DataFrame"""
    empty_df = pd.DataFrame(columns=['date', 'value'])
    with pytest.raises(ValueError):
        get_split_train_val_cv(empty_df, n_splits=3)

def test_invalid_splits():
    """Test with invalid number of splits"""
    df = pd.DataFrame({'value': range(5)})
    # Test with n_splits = 0
    with pytest.raises(ValueError):
        get_split_train_val_cv(df, n_splits=0)
    # Test with n_splits > len(df)
    with pytest.raises(ValueError):
        get_split_train_val_cv(df, n_splits=10)

def test_small_dataframe():
    """Test with minimal size DataFrame"""
    small_df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=2),
        'value': range(2)
    })
    with pytest.raises(ValueError):
        get_split_train_val_cv(small_df.copy(), n_splits=3)

def test_column_preservation(sample_df_split_cv):
    """Test that all columns are preserved in splits"""
    # Add extra columns
    df = sample_df_split_cv.copy()
    df['category'] = 'A'
    df['extra'] = 1.0
    result = get_split_train_val_cv(df, n_splits=3)
    for train, val in result:
        assert all(col in train.columns for col in df.columns)
        assert all(col in val.columns for col in df.columns)
