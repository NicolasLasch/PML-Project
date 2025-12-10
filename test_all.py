import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    parse_timestamps,
    extract_temporal_features,
    clean_missing_values,
    remove_outliers,
    preprocess_pipeline
)
from src.feature_engineering import (
    aggregate_station_hourly,
    create_lag_features,
    create_rolling_features,
    create_weather_interactions,
    encode_cyclic_features,
    calculate_station_popularity,
    get_feature_columns
)
from src.model_train_eval import (
    train_random_forest,
    train_xgboost,
    evaluate_model,
    get_feature_importance
)
from src.station_reco import (
    calculate_distance,
    get_nearby_stations,
    calculate_recommendation_score
)


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'start_time': ['2024-01-01 08:30:00', '2024-01-02 17:45:00', '2024-01-03 12:00:00'],
        'end_time': ['2024-01-01 09:00:00', '2024-01-02 18:15:00', '2024-01-03 12:30:00'],
        'duration': [1800, 1800, 1800],
        'temperature': [15.5, np.nan, 20.0],
        'max_temperature': [18.0, 14.0, 22.0],
        'min_temperature': [12.0, 10.0, 18.0],
        'wind_speed': [5.0, 6.0, 4.0],
        'precipitation': [0.0, 2.5, 0.0],
        'humidity': [70.0, 80.0, 65.0],
        'sunshine': [5.0, 2.0, 6.0],
        'season': [0, 0, 0],
        'is_holiday': [0, 0, 0],
        'is_weekend': [0, 0, 0]
    })


@pytest.fixture
def processed_data():
    dates = pd.date_range('2024-01-01', periods=72, freq='h')
    return pd.DataFrame({
        'start_station_id': [1] * 72,
        'start_station_name': ['Station A'] * 72,
        'start_time': dates,
        'duration': np.random.randint(10, 50, 72),
        'temperature': np.random.uniform(10, 25, 72),
        'max_temperature': np.random.uniform(15, 30, 72),
        'min_temperature': np.random.uniform(5, 15, 72),
        'wind_speed': np.random.uniform(0, 20, 72),
        'precipitation': np.random.uniform(0, 5, 72),
        'humidity': np.random.uniform(50, 90, 72),
        'sunshine': np.random.uniform(0, 10, 72),
        'season': [0] * 72,
        'is_holiday': [0] * 72,
        'is_weekend': [0] * 72,
        'start_station_latitude': [60.39] * 72,
        'start_station_longitude': [5.32] * 72,
        'is_rush_hour': [0] * 72,
        'dayofweek': [0] * 72,
        'month': [1] * 72,
        'hour': list(range(24)) * 3
    })


class TestDataProcessing:
    
    def test_parse_timestamps(self, sample_raw_data):
        result = parse_timestamps(sample_raw_data)
        assert pd.api.types.is_datetime64_any_dtype(result['start_time'])
        assert pd.api.types.is_datetime64_any_dtype(result['end_time'])
    
    def test_extract_temporal_features(self, sample_raw_data):
        df = parse_timestamps(sample_raw_data)
        result = extract_temporal_features(df)
        
        assert 'hour' in result.columns
        assert 'day' in result.columns
        assert 'month' in result.columns
        assert 'dayofweek' in result.columns
        assert 'is_rush_hour' in result.columns
        
        assert result['hour'].iloc[0] == 8
        assert result['is_rush_hour'].iloc[0] == 1
        assert result['is_rush_hour'].iloc[2] == 0
    
    def test_clean_missing_values(self, sample_raw_data):
        result = clean_missing_values(sample_raw_data)
        assert result['temperature'].isna().sum() == 0
    
    def test_remove_outliers(self):
        df = pd.DataFrame({'duration': [10, 20, 30, 40, 1000]})
        result = remove_outliers(df, 'duration', multiplier=1.5)
        assert len(result) < len(df)
        assert 1000 not in result['duration'].values
    
    def test_preprocess_pipeline(self, sample_raw_data):
        result = preprocess_pipeline(sample_raw_data)
        
        assert 'hour' in result.columns
        assert 'is_rush_hour' in result.columns
        assert result['temperature'].isna().sum() == 0
        assert pd.api.types.is_datetime64_any_dtype(result['start_time'])


class TestFeatureEngineering:
    
    def test_aggregate_station_hourly(self, processed_data):
        result = aggregate_station_hourly(processed_data)
        
        assert 'trip_count' in result.columns
        assert 'start_station_id' in result.columns
        assert len(result) <= len(processed_data)
    
    def test_create_lag_features(self, processed_data):
        hourly = aggregate_station_hourly(processed_data)
        result = create_lag_features(hourly, lags=[1, 2])
        
        assert 'trip_count_lag_1' in result.columns
        assert 'trip_count_lag_2' in result.columns
    
    def test_create_rolling_features(self, processed_data):
        hourly = aggregate_station_hourly(processed_data)
        result = create_rolling_features(hourly, windows=[3])
        
        assert 'trip_count_rolling_3h' in result.columns
    
    def test_create_weather_interactions(self, processed_data):
        result = create_weather_interactions(processed_data)
        
        assert 'temp_humidity' in result.columns
        assert 'temp_wind' in result.columns
        assert 'rain_wind' in result.columns
        assert 'is_rainy' in result.columns
        assert 'is_hot' in result.columns
        assert 'is_cold' in result.columns
    
    def test_encode_cyclic_features(self, processed_data):
        result = encode_cyclic_features(processed_data)
        
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        assert 'dayofweek_sin' in result.columns
        assert 'dayofweek_cos' in result.columns
        assert 'month_sin' in result.columns
        assert 'month_cos' in result.columns
        
        assert result['hour_sin'].between(-1, 1).all()
        assert result['hour_cos'].between(-1, 1).all()
    
    def test_calculate_station_popularity(self, processed_data):
        hourly = aggregate_station_hourly(processed_data)
        result = calculate_station_popularity(hourly)
        
        assert 'station_avg_trips' in result.columns
        assert 'station_std_trips' in result.columns
    
    def test_get_feature_columns(self):
        feature_cols = get_feature_columns()
        
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert 'temperature' in feature_cols
        assert 'trip_count_lag_1' in feature_cols


class TestModelTraining:
    
    @pytest.fixture
    def sample_ml_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randint(0, 50, 100))
        return X, y
    
    def test_train_random_forest(self, sample_ml_data):
        X, y = sample_ml_data
        model = train_random_forest(X, y)
        
        assert hasattr(model, 'predict')
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_train_xgboost(self, sample_ml_data):
        X, y = sample_ml_data
        model = train_xgboost(X, y)
        
        assert hasattr(model, 'predict')
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    
    def test_evaluate_model(self, sample_ml_data):
        X, y = sample_ml_data
        model = train_random_forest(X, y)
        metrics = evaluate_model(model, X, y)
        
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_get_feature_importance(self, sample_ml_data):
        X, y = sample_ml_data
        model = train_random_forest(X, y)
        importance_df = get_feature_importance(model, X.columns.tolist())
        
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == X.shape[1]
        assert importance_df['importance'].sum() > 0


class TestStationRecommendation:
    
    def test_calculate_distance(self):
        lat1, lon1 = 60.3913, 5.3221
        lat2, lon2 = 60.3950, 5.3250
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        
        assert isinstance(distance, float)
        assert distance > 0
        assert distance < 1
    
    def test_calculate_distance_same_point(self):
        lat, lon = 60.3913, 5.3221
        distance = calculate_distance(lat, lon, lat, lon)
        
        assert distance == 0
    
    @pytest.fixture
    def stations_df(self):
        return pd.DataFrame({
            'start_station_id': [1, 2, 3, 4],
            'start_station_name': ['Station A', 'Station B', 'Station C', 'Station D'],
            'start_station_latitude': [60.3913, 60.3950, 60.4000, 60.5000],
            'start_station_longitude': [5.3221, 5.3250, 5.3300, 5.4000]
        })
    
    def test_get_nearby_stations(self, stations_df):
        user_lat, user_lon = 60.3913, 5.3221
        
        nearby = get_nearby_stations(user_lat, user_lon, stations_df, radius_km=2.0)
        
        assert 'distance_km' in nearby.columns
        assert len(nearby) <= len(stations_df)
        assert (nearby['distance_km'] <= 2.0).all()
    
    def test_calculate_recommendation_score(self):
        score1 = calculate_recommendation_score(0.5, 10, max_distance=2.0)
        score2 = calculate_recommendation_score(1.5, 10, max_distance=2.0)
        
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        assert score1 > score2


class TestFlaskApp:
    
    @pytest.fixture
    def check_models_exist(self):
        models_dir = Path('models')
        required_models = ['XGB_Tuned.joblib', 'RF_Tuned.joblib', 'XGB_Baseline.joblib', 'RF_Baseline.joblib']
        
        if not models_dir.exists():
            pytest.skip("Models directory not found")
        
        for model_name in required_models:
            if not (models_dir / model_name).exists():
                pytest.skip(f"Model {model_name} not found")
    
    def test_models_loaded(self, check_models_exist):
        import joblib
        model = joblib.load('models/XGB_Tuned.joblib')
        assert model is not None


class TestIntegration:
    
    def test_full_pipeline_sample(self, sample_raw_data):
        processed = preprocess_pipeline(sample_raw_data)
        assert len(processed) > 0
        
        features = create_weather_interactions(processed)
        features = encode_cyclic_features(features)
        
        assert 'temp_humidity' in features.columns
        assert 'hour_sin' in features.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])