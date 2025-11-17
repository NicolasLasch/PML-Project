import pandas as pd
import numpy as np

def parse_timestamps(df):
    """
    Convert 'start_time' and 'end_time' columns to datetime objects for
    further time-based operations
    """
    # Start and End Time
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    return df


def extract_temporal_features(df):
    """
    Extract relevant temporal features from 'start_time':
    hour, day, month, day of week, rush hour category
    """
    # Hours, Days and Months
    df['hour'] = df['start_time'].dt.hour   # Hour of day (0-23)
    df['day'] = df['start_time'].dt.day     # Day of month (1-31)
    df['month'] = df['start_time'].dt.month # Month of yaer (1-12)

    # Day of Week
    df['dayofweek'] = df['start_time'].dt.dayofweek # Day of week (0-6, MON - SUN)

    # Is rush hour
    df['is_rush_hour'] = df['hour'].isin(
        [7, 8, 9, 16, 17, 18]
    ).astype(int)
    return df


def clean_missing_values(df):
    """
    Dealing with missing values :
    column = numerical columns
    """
    # Identify numerical columns for imputation
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns
    # Impute missing values in numeric columns with median
    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )
    return df


def remove_outliers(df, column, multiplier = 3.0):
    """
    Removing outliers from dataframe of specific column
    """
    # Interquartile Range (IQR) for outlier detection
    q1 = df[column].quantile(0.25)  # 25th percentile
    q3 = df[column].quantile(0.75)  # 75th percentile
    iqr = q3 - q1   # Interquartile Range

    # Lower and upper bounds for outliers with multiplier
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    # Filter dataframe within bounds, excluding outliers
    mask = (df[column] >= lower) & (df[column] <= upper)
    return df[mask]


def preprocess_pipeline(df):
    """
    Preprocessing pipeline applying all steps sequentially
    """
    df = parse_timestamps(df)
    df = extract_temporal_features(df)
    df = clean_missing_values(df)
    df = remove_outliers(df, 'duration')
    # Fully processed dataframe
    return df
