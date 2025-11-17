import pandas as pd
import numpy as np

def aggregate_station_hourly(df):
    """
    Aggregate data hourly for each start station, summarizing
    trip counts and weather conditions
    """
    hourly = df.groupby([
        'start_station_id',
        'start_station_name',
        df['start_time'].dt.date.rename('date'),    # Group by date (day precision)
        'hour'
    ]).agg({
        'duration': 'count',                        # Count trips per group
        'temperature': 'mean',                      # Average temperature, max/min
        'max_temperature': 'mean',
        'min_temperature': 'mean',
        'wind_speed': 'mean',                       # Avg wind speed, precipitation, humidity, sunshine
        'precipitation': 'mean',
        'humidity': 'mean',
        'sunshine': 'mean',
        'season': 'first',                          # Static fields (first value in the group)
        'is_holiday': 'first',
        'is_weekend': 'first',
        'start_station_latitude': 'first',
        'start_station_longitude': 'first',
        'is_rush_hour': 'first',
        'dayofweek': 'first',
        'month': 'first'
    }).reset_index()
    
    # Rename 'duration' count to 'trip_count' for clarity
    hourly = hourly.rename(columns={'duration': 'trip_count'})
    return hourly


def create_lag_features(df, lags=[1, 2, 3, 24]):
    """
    Creates lag features for 'trip_count' representing past hourly bike usage.
    Adds columns 'trip_count_lag_X' where X are lag hours corresponding to previous periods.
    """
    # Sort data to maintain temporal order for meaningful lag features
    df = df.sort_values(['start_station_id', 'date', 'hour'])
    
    # Lag Features representing trip counts from previous hours
    for lag in lags:
        df[f'trip_count_lag_{lag}'] = df.groupby('start_station_id')['trip_count'].shift(lag)
    
    return df


def create_rolling_features(df, windows=[3, 6, 24]):
    """
    """
    # Sort data to maintain temporal order for meaningful rolling averages
    df = df.sort_values(['start_station_id', 'date', 'hour'])
    
    # Create rolling window mean features for trip counts to capture recent trends
    for window in windows:
        df[f'trip_count_rolling_{window}h'] = df.groupby('start_station_id')['trip_count'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    return df


def create_weather_interactions(df):
    """
    Creates interaction terms and binary flags derived from weather variables.
    Models non-linear effects (temp * humidity) and simple indicators (rainy, hot, cold)
    """
    # Generate interaction terms between weather features
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['temp_wind'] = df['temperature'] * df['wind_speed']
    df['rain_wind'] = df['precipitation'] * df['wind_speed']

    # Create binary indicators to simplify complec weather conditions
    df['is_rainy'] = (df['precipitation'] > 0).astype(int)
    df['is_hot'] = (df['temperature'] > 20).astype(int)
    df['is_cold'] = (df['temperature'] < 5).astype(int)
    return df


def encode_cyclic_features(df):
    """
    Encodes cyclical temporal features (hour, day of week, month) using
    sins and cosine transforms (continuity in cyclic variables)
    """
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def calculate_station_popularity(df):
    """
    Compute station-level popularity statistics :
    - mean number of trips per hour
    - standard deviation to quantify variability
    """
    station_stats = df.groupby('start_station_id')['trip_count'].agg(
        ['mean', 'std']
    ).reset_index()
    station_stats.columns = ['start_station_id', 'station_avg_trips', 'station_std_trips']

    df = df.merge(station_stats, on='start_station_id', how='left')
    return df


def feature_engineering_pipeline(df):
    """
    Feature engineering workflow
    """
    df = aggregate_station_hourly(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_weather_interactions(df)
    df = encode_cyclic_features(df)
    df = calculate_station_popularity(df)
    df = df.dropna()
    return df


def prepare_train_test_split(df, test_ratio = 0.2):
    """
    Split Dataframe into training and test sets for time series data
    - Sort data by 'date' and 'hour' to preserve temporal order
    - Split index based on specified test_ratio
    """
    df = df.sort_values(['date', 'hour'])
    split_idx = int(len(df) * (1 - test_ratio))
    # Split without shuffling
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def get_feature_columns():
    """
    Feature column names used in the modeling pipeline
    """
    return [
        # Weather features
        'temperature', 'max_temperature', 'min_temperature',
        'wind_speed', 'precipitation', 'humidity', 'sunshine',

        # Calendar and temporal flags
        'season', 'is_holiday', 'is_weekend', 'is_rush_hour',

        # Cyclic temporal encodings to preserve periodicity
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'month_sin', 'month_cos',

        # Lag features capturing recent station usage history
        'trip_count_lag_1', 'trip_count_lag_2', 'trip_count_lag_3', 'trip_count_lag_24',
        # Rolling average features for smoothing trends
        'trip_count_rolling_3h', 'trip_count_rolling_6h', 'trip_count_rolling_24h',

        # Weather interaction terms and binary flags
        'temp_humidity', 'temp_wind', 'rain_wind',
        'is_rainy', 'is_hot', 'is_cold',

        # Station popularity metrics
        'station_avg_trips', 'station_std_trips',

        # Geospatial station coordinates
        'start_station_latitude', 'start_station_longitude'
    ]