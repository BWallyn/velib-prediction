"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.7
"""
# =================
# ==== IMPORTS ====
# =================

import datetime
import logging

import numpy as np
import pandas as pd

from velib_prediction.pipelines.feature_engineering.dataclasses import (
    ValidatedDataFrame,
)

# Options
logger = logging.getLogger(__name__)

# ===================
# ==== FUNCTIONS ====
# ===================

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        df (pd.DataFrame): Output DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype

        if col in ["coordonnees_geo", "date"]:
            pass
        elif (not isinstance(col_type, object)):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # noqa: PLR5501
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
    logger.info(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


def split_train_test(df: pd.DataFrame, feat_date: str, delta_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets.

    Args:
        df (pd.DataFrame): Input dataframe
        feat_date (str): Name of the date column
        delta_days (int): Number of days that will be kept for the test set
    Returns:
        df_train (pd.DataFrame): Train set
        df_test (pd.DataFrame): Test set
    """
    # Set the right type for date feature
    df[feat_date] = pd.to_datetime(df[feat_date].cat.as_ordered(), format='%Y-%m-%d').dt.date
    # Check number of days in integer:
    assert isinstance(delta_days, int), "delta_days should be an integer"
    # Get the max date
    max_date = df[feat_date].max()
    train_cutoff = max_date - datetime.timedelta(days=delta_days)
    # Split
    df_train = df[df[feat_date] < train_cutoff]
    df_test = df[df[feat_date] >= train_cutoff]
    return df_train, df_test


def get_holidays() -> pd.DataFrame:
    """Get holidays dataset from data.education.gouv

    - Download the info of the holidays from data.education.gouv
    - Keep only the french metropolitan zones
    - Drop duplicates to keep just one row by zone and holiday period
    - Select only years 2020 and after

    Returns:
        df_holidays (pd.DataFrame): Dataframe of the holiday dates
    """
    path_data = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
    df_holidays = pd.read_csv(path_data, sep=";").sort_values(by="Date de début")
    # Select only metropolitan France dates
    df_holidays = df_holidays[df_holidays["Zones"].isin(["Zone A", "Zone B", "Zone C"])]
    # Drop duplicates between same zones and dates
    df_holidays.drop_duplicates(subset=["Zones", "Date de début"], inplace=True)
    # Change types
    df_holidays["Date de début"] = pd.to_datetime(df_holidays["Date de début"].str[:10], format="%Y-%m-%d")
    df_holidays["Date de fin"] = pd.to_datetime(df_holidays["Date de fin"].str[:10], format="%Y-%m-%d")
    df_holidays.rename(columns={"Date de début": "date_begin", "Date de fin": "date_end"}, inplace=True)
    # df_holidays["Date de début"] = df_holidays["Date de début"].dt.date
    # Select only after 2020
    df_holidays = df_holidays[df_holidays["date_end"] >= datetime.datetime(2020, 1, 1)]
    return df_holidays


def add_holidays_period(df: pd.DataFrame, df_holidays: pd.DataFrame, feat_date: str, zone: str="Zone A") -> pd.DataFrame:
    """Add the holidays periods to the dataset

    Args:
        df (pd.DataFrame): Input dataframe
        df_holidays (pd.DataFrame): Holidays dataframe
        feat_date (str): Name of the date feature
    Returns:
        df_final (pd.DataFrame): DataFrame with the holidays indicator
    """
    # Options
    zone_name = zone.replace(" ", "")
    # Set zone
    df_holidays_zone = df_holidays[df_holidays["Zones"] == zone]
    # Set the right type
    df[feat_date] = pd.to_datetime(df[feat_date], format="%Y-%m-%d")
    # Reset index
    # TODO fix left index problem
    df = df.reset_index(drop=True)
    # Merge closest holiday date
    merged_df = pd.merge_asof(
        df, df_holidays_zone[["date_begin", "date_end", "Description"]], left_on=feat_date, right_on='date_begin', direction='backward'
    )
    # Set the right index as they are lost during merge asof
    merged_df.index = df.index
    # Filter out rows where the Date is before the begining or after the 'end' date
    merged_df = merged_df[(merged_df[feat_date] >= merged_df['date_begin']) & (merged_df[feat_date] <= merged_df['date_end'])]
    merged_df.drop(columns=["date_begin", "date_end"], inplace=True)
    merged_df.rename(columns={"Description": f"Description_{zone_name}"}, inplace=True)
    # Select rows without holidays
    df_no_holidays = df.drop(index=merged_df.index)
    df_no_holidays[f"Description_{zone_name}"] = "None"
    df_final = pd.concat([df_no_holidays, merged_df]).sort_values(by=feat_date)
    return df_final


def get_weekend(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Get if the day is weekend or not
    - 0 if the day is a weekday
    - 1 if the day is weekend

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns:
        df: dataset with the feature weekend or not
    """
    df.loc[:, f"{feat_date}_weekend"] = 0
    df.loc[df[f"{feat_date}_weekday"] >= 5, f"{feat_date}_weekend"] = 1  # noqa: PLR2004
    return df


def extract_date_features(df: pd.DataFrame, feat_date: str) -> pd.DataFrame:
    """Create features from a datetime feature

    Args:
        df: dataset
        feat_date: name of the date feature
    Returns
        df: dataset with features extracted from the date
    """
    df[feat_date] = pd.to_datetime(df[feat_date], format='%Y%m%d')
    df[f'{feat_date}_year'] = df[feat_date].dt.year
    df[f'{feat_date}_month'] = df[feat_date].dt.month
    df[f'{feat_date}_day'] = df[feat_date].dt.day
    df[f'{feat_date}_weekday'] = df[feat_date].dt.weekday
    df = get_weekend(df, feat_date=feat_date)
    return df


def drop_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """Drop specific columns

    Args:
        df (pd.DataFrame): Input DataFrame
        cols_to_drop (list[str]): List of columns to drop
    Returns:
        (pd.DataFrame): Output DataFrame
    """
    return df.drop(columns=cols_to_drop)


def validate_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the input DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame
    Returns:
        df (pd.DataFrame): Output DataFrame validated
    """
    # Perform dataframe validation
    validated_df = ValidatedDataFrame(df=df)
    return validated_df.df
