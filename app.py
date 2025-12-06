import joblib
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from src.feature_engineering import get_feature_columns

app = Flask(__name__)

best_model = joblib.load("models/XGB_Tuned.joblib")
feature_cols = get_feature_columns()

sample_df = pd.read_csv("data/processed/sample_features.csv")
features_df = pd.read_csv("data/processed/feature_engineered_bergen_merged.csv")
comparison_df = pd.read_csv("output/model_comparison_results.csv", index_col=0)

raw_df = pd.read_csv("data/raw/bergen_merged.csv")
raw_df['start_time'] = pd.to_datetime(raw_df['start_time'])
raw_df['date'] = raw_df['start_time'].dt.date.astype(str)

models_cache = {}
for model_name in ['RF_Baseline', 'RF_Tuned', 'XGB_Baseline', 'XGB_Tuned']:
    model_path = Path(f"models/{model_name}.joblib")
    if model_path.exists():
        models_cache[model_name] = joblib.load(model_path)
        print(f"✓ Loaded {model_name}")


@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/api/init", methods=["GET"])
def initialize_dashboard():
    best_model_name = comparison_df['RMSE'].idxmin()
    
    samples = []
    for i, row in sample_df.head(100).iterrows():
        samples.append({
            'index': int(i),
            'label': f"{row['start_station_name']} | {row['date']} {int(row['hour'])}h | {row['trip_count']} trips",
            'station': str(row['start_station_name']),
            'date': str(row['date']),
            'hour': int(row['hour']),
            'trips': float(row['trip_count']),
            'lat': float(row['start_station_latitude']),
            'lon': float(row['start_station_longitude'])
        })
    
    latest_date = features_df['date'].max()
    daily_totals = features_df[features_df['date'] == latest_date].groupby('start_station_id').agg({
        'start_station_name': 'first',
        'start_station_latitude': 'first',
        'start_station_longitude': 'first',
        'trip_count': 'sum'
    }).reset_index()
    
    total_bikes_out = int(daily_totals['trip_count'].sum())
    avg_bikes_per_station = round(float(daily_totals['trip_count'].mean()), 1)
    
    station_geojson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [float(row['start_station_longitude']), float(row['start_station_latitude'])]
                },
                'properties': {
                    'id': int(row['start_station_id']),
                    'name': str(row['start_station_name']),
                    'daily_bikes_out': int(row['trip_count'])
                }
            }
            for _, row in daily_totals.iterrows()
        ]
    }
    
    return jsonify({
        'metrics': {
            'best_model': best_model_name,
            'total_stations': int(sample_df['start_station_id'].nunique()),
            'total_bikes_out': total_bikes_out,
            'avg_bikes_per_station': avg_bikes_per_station,
            'models': comparison_df.to_dict('index')
        },
        'samples': samples,
        'stations': station_geojson
    })

@app.route("/api/predict", methods=["POST"])
def predict_compare():
    try:
        data = request.get_json()
        row_idx = int(data.get("row_index"))
        
        row_data = sample_df.iloc[row_idx]
        X = row_data[feature_cols].values.reshape(1, -1)
        
        results = {}
        actual = float(row_data['trip_count'])
        
        selected_station_id = row_data['start_station_id']
        selected_date = row_data['date']
        
        station_day_data = features_df[
            (features_df['start_station_id'] == selected_station_id) &
            (features_df['date'].astype(str) == str(selected_date))
        ].sort_values('hour')
        
        if len(station_day_data) < 24:
            station_day_data = features_df[
                features_df['start_station_id'] == selected_station_id
            ].tail(24).sort_values('hour')
        
        timeseries_all = {}
        
        for name, model in models_cache.items():
            pred = float(model.predict(X)[0])
            error = pred - actual
            error_pct = (error / actual * 100) if actual != 0 else 0
            
            results[name] = {
                'prediction': round(pred, 2),
                'error': round(error, 2),
                'error_pct': round(error_pct, 2),
                'mae': round(abs(error), 2),
                'accuracy': round(100 - abs(error_pct), 2)
            }
            
            predictions = model.predict(station_day_data[feature_cols])
            timeseries_all[name] = [
                {
                    'time': f"{row['date']} {int(row['hour']):02d}:00",
                    'actual': float(row['trip_count']),
                    'predicted': float(predictions[idx])
                }
                for idx, (_, row) in enumerate(station_day_data.iterrows())
            ]
        
        return jsonify({
            'actual': actual,
            'models': results,
            'timeseries': timeseries_all,
            'selected_station': {
                'name': str(row_data['start_station_name']),
                'lat': float(row_data['start_station_latitude']),
                'lon': float(row_data['start_station_longitude'])
            },
            'context': {
                'station': str(row_data['start_station_name']),
                'date': str(row_data['date']),
                'hour': int(row_data['hour']),
                'temperature': float(row_data['temperature']),
                'precipitation': float(row_data['precipitation']),
                'wind_speed': float(row_data['wind_speed']),
                'is_rush_hour': bool(row_data['is_rush_hour']),
                'is_weekend': bool(row_data['is_weekend'])
            }
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/analyze_station_day", methods=["POST"])
def analyze_station_day():
    try:
        data = request.get_json()
        station_name = data.get("station_name")
        target_date = data.get("date")
        
        station_day_data = features_df[
            (features_df['start_station_name'] == station_name) &
            (features_df['date'].astype(str) == target_date)
        ].sort_values('hour')
        
        if station_day_data.empty:
            return jsonify({'error': f'No data for {station_name} on {target_date}'}), 404
        
        results = {}
        timeseries_all = {}
        total_actual = station_day_data['trip_count'].sum()
        
        for name, model in models_cache.items():
            predictions = model.predict(station_day_data[feature_cols])
            total_predicted = predictions.sum()
            error = total_predicted - total_actual
            error_pct = (error / total_actual * 100) if total_actual != 0 else 0
            
            results[name] = {
                'prediction': round(float(total_predicted), 2),
                'error': round(float(error), 2),
                'error_pct': round(float(error_pct), 2),
                'mae': round(float(abs(error)), 2),
                'accuracy': round(100 - abs(float(error_pct)), 2)
            }
            
            timeseries_all[name] = [
                {
                    'time': f"{int(row['hour']):02d}:00",
                    'actual': float(row['trip_count']),
                    'predicted': float(predictions[idx])
                }
                for idx, (_, row) in enumerate(station_day_data.iterrows())
            ]
        
        avg_temp = station_day_data['temperature'].mean()
        avg_precip = station_day_data['precipitation'].mean()
        avg_wind = station_day_data['wind_speed'].mean()
        is_weekend = bool(station_day_data.iloc[0]['is_weekend'])
        
        return jsonify({
            'actual': float(total_actual),
            'models': results,
            'timeseries': timeseries_all,
            'selected_station': {
                'name': station_name,
                'lat': float(station_day_data.iloc[0]['start_station_latitude']),
                'lon': float(station_day_data.iloc[0]['start_station_longitude'])
            },
            'context': {
                'station': station_name,
                'date': target_date,
                'total_hours': len(station_day_data),
                'temperature': round(float(avg_temp), 1),
                'precipitation': round(float(avg_precip), 1),
                'wind_speed': round(float(avg_wind), 1),
                'is_weekend': is_weekend
            }
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend_station():
    try:
        data = request.get_json()
        user_lat = float(data.get("lat", 60.3913))
        user_lon = float(data.get("lon", 5.3221))
        target_hour = int(data.get("hour", 8))
        target_date = data.get("date", str(features_df['date'].max()))
        
        from src.station_reco import calculate_distance
        
        unique_stations = features_df.groupby('start_station_id').agg({
            'start_station_name': 'first',
            'start_station_latitude': 'first',
            'start_station_longitude': 'first'
        }).reset_index()
        
        recommendations = []
        
        for _, station in unique_stations.iterrows():
            distance = calculate_distance(
                user_lat, user_lon,
                station['start_station_latitude'],
                station['start_station_longitude']
            )
            
            if distance > 2.0:
                continue
            
            station_id = station['start_station_id']
            
            station_data = features_df[
                (features_df['start_station_id'] == station_id) &
                (features_df['date'].astype(str) == target_date)
            ]
            
            if station_data.empty:
                station_data = features_df[
                    features_df['start_station_id'] == station_id
                ].tail(24)
            
            bikes_available = 1
            total_out_since_6am = 0
            total_in_since_6am = 0
            
            historical_in = raw_df[raw_df['end_station_id'] == station_id].shape[0]
            historical_out = raw_df[raw_df['start_station_id'] == station_id].shape[0]
            in_out_ratio = historical_in / historical_out if historical_out > 0 else 1.0
            
            for hour in range(6, target_hour + 1):
                hour_data = station_data[station_data['hour'] == hour]
                
                if not hour_data.empty:
                    X = hour_data.iloc[0][feature_cols].values.reshape(1, -1)
                else:
                    if len(station_data) > 0:
                        template = station_data.iloc[-1].copy()
                    else:
                        continue
                    
                    template['hour'] = hour
                    template['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                    template['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                    X = template[feature_cols].values.reshape(1, -1)
                
                pred_out = float(best_model.predict(X)[0])
                pred_in = pred_out * in_out_ratio
                
                total_out_since_6am += pred_out
                total_in_since_6am += pred_in
                bikes_available += pred_in - pred_out
            
            if bikes_available <= 0:
                availability_score = 0
            else:
                availability_score = min(bikes_available / 5.0, 1.0)
            
            distance_score = 1 - (distance / 2.0)
            
            combined_score = (0.7 * distance_score) + (0.3 * availability_score)
            
            recommendations.append({
                'station_id': int(station_id),
                'station_name': str(station['start_station_name']),
                'lat': float(station['start_station_latitude']),
                'lon': float(station['start_station_longitude']),
                'distance_km': round(distance, 2),
                'bikes_available': round(bikes_available, 1),
                'total_out_since_6am': round(total_out_since_6am, 1),
                'total_in_since_6am': round(total_in_since_6am, 1),
                'score': round(combined_score, 3)
            })
        
        if not recommendations:
            return jsonify({'error': 'No stations found nearby'}), 404
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = recommendations[:5]
        
        best_station = top_recommendations[0]
        
        return jsonify({
            'user_location': {'lat': user_lat, 'lon': user_lon},
            'target_hour': target_hour,
            'best_station': {
                'name': best_station['station_name'],
                'lat': best_station['lat'],
                'lon': best_station['lon'],
                'distance_km': best_station['distance_km'],
                'bikes_available': best_station['bikes_available'],
                'total_out_since_6am': best_station['total_out_since_6am'],
                'total_in_since_6am': best_station['total_in_since_6am'],
                'score': best_station['score']
            },
            'recommendations': top_recommendations
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/station_rankings", methods=["POST"])
def get_station_rankings():
    try:
        data = request.get_json()
        target_date = data.get("date")
        target_hour = int(data.get("hour", 8))
        
        filtered_df = features_df[
            (features_df['date'].astype(str) == target_date) &
            (features_df['hour'] == target_hour)
        ]
        
        if filtered_df.empty:
            return jsonify({'error': 'No data for selected date/hour'}), 404
        
        rankings = filtered_df.sort_values('trip_count', ascending=False).head(20)
        
        results = []
        for idx, row in rankings.iterrows():
            X = row[feature_cols].values.reshape(1, -1)
            predicted = float(best_model.predict(X)[0])
            
            results.append({
                'rank': len(results) + 1,
                'station_name': str(row['start_station_name']),
                'actual_trips': float(row['trip_count']),
                'predicted_trips': round(predicted, 1),
                'lat': float(row['start_station_latitude']),
                'lon': float(row['start_station_longitude']),
                'temperature': float(row['temperature']),
                'is_rush_hour': bool(row['is_rush_hour'])
            })
        
        return jsonify({
            'date': target_date,
            'hour': target_hour,
            'rankings': results
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/daily_station_flow", methods=["POST"])
def get_daily_station_flow():
    try:
        data = request.get_json()
        target_date = data.get("date")
        custom_weather = data.get("weather")
        
        daily_data = features_df[features_df['date'].astype(str) == target_date]
        has_actual_data = not daily_data.empty
        
        if has_actual_data:
            raw_daily = raw_df[raw_df['date'] == target_date]
            
            bikes_out = raw_daily.groupby('start_station_id').agg({
                'start_station_name': 'first',
                'start_station_latitude': 'first',
                'start_station_longitude': 'first',
                'duration': 'count'
            }).reset_index()
            bikes_out.columns = ['station_id', 'station_name', 'latitude', 'longitude', 'bikes_out']
            
            bikes_in = raw_daily.groupby('end_station_id').agg({
                'duration': 'count'
            }).reset_index()
            bikes_in.columns = ['station_id', 'bikes_in']
            
            all_stations = bikes_out.merge(bikes_in, on='station_id', how='outer').fillna(0)
            
            predictions_out = {}
            for _, row in all_stations.iterrows():
                station_id = row['station_id']
                station_daily = daily_data[daily_data['start_station_id'] == station_id]
                
                total_pred_out = 0
                for hour in range(24):
                    hour_data = station_daily[station_daily['hour'] == hour]
                    if not hour_data.empty:
                        X = hour_data.iloc[0][feature_cols].values.reshape(1, -1)
                        pred = float(best_model.predict(X)[0])
                        total_pred_out += pred
                
                predictions_out[station_id] = total_pred_out
            
            total_actual_in = all_stations['bikes_in'].sum()
            total_pred_out_sum = sum(predictions_out.values())
            
            predictions_in = {}
            for station_id in all_stations['station_id']:
                actual_in = all_stations[all_stations['station_id'] == station_id]['bikes_in'].values[0]
                if total_actual_in > 0:
                    in_proportion = actual_in / total_actual_in
                    predictions_in[station_id] = in_proportion * total_pred_out_sum
                else:
                    predictions_in[station_id] = predictions_out.get(station_id, 0)
            
        else:
            all_stations = features_df.groupby('start_station_id').agg({
                'start_station_name': 'first',
                'start_station_latitude': 'first',
                'start_station_longitude': 'first'
            }).reset_index()
            all_stations.columns = ['station_id', 'station_name', 'latitude', 'longitude']
            
            historical_in = raw_df.groupby('end_station_id').size().to_dict()
            
            predictions_out = {}
            for _, row in all_stations.iterrows():
                station_id = row['station_id']
                station_features = features_df[
                    features_df['start_station_id'] == station_id
                ].tail(24)
                
                if len(station_features) == 0:
                    predictions_out[station_id] = 0
                    continue
                
                total_pred_out = 0
                for hour in range(24):
                    template = station_features.iloc[-1].copy()
                    template['hour'] = hour
                    template['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                    template['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                    
                    if custom_weather:
                        template['temperature'] = custom_weather.get('temperature', template['temperature'])
                        template['max_temperature'] = custom_weather.get('max_temperature', template['max_temperature'])
                        template['min_temperature'] = custom_weather.get('min_temperature', template['min_temperature'])
                        template['precipitation'] = custom_weather.get('precipitation', template['precipitation'])
                        template['wind_speed'] = custom_weather.get('wind_speed', template['wind_speed'])
                        template['humidity'] = custom_weather.get('humidity', template['humidity'])
                        template['sunshine'] = custom_weather.get('sunshine', template['sunshine'])
                        
                        template['temp_humidity'] = template['temperature'] * template['humidity']
                        template['temp_wind'] = template['temperature'] * template['wind_speed']
                        template['rain_wind'] = template['precipitation'] * template['wind_speed']
                        template['is_rainy'] = 1 if template['precipitation'] > 0 else 0
                        template['is_hot'] = 1 if template['temperature'] > 20 else 0
                        template['is_cold'] = 1 if template['temperature'] < 5 else 0
                    
                    X = template[feature_cols].values.reshape(1, -1)
                    pred = float(best_model.predict(X)[0])
                    total_pred_out += pred
                
                predictions_out[station_id] = total_pred_out
            
            total_hist_in = sum(historical_in.values())
            total_pred_out_sum = sum(predictions_out.values())
            
            predictions_in = {}
            for station_id in all_stations['station_id']:
                hist_in = historical_in.get(station_id, 0)
                if total_hist_in > 0:
                    in_proportion = hist_in / total_hist_in
                    predictions_in[station_id] = in_proportion * total_pred_out_sum
                else:
                    predictions_in[station_id] = predictions_out.get(station_id, 0)
        
        results = []
        for _, row in all_stations.iterrows():
            station_id = int(row['station_id'])
            
            result = {
                'station_id': station_id,
                'station_name': str(row['station_name']),
                'lat': float(row['latitude']),
                'lon': float(row['longitude']),
                'predicted_bikes_out': round(predictions_out.get(station_id, 0), 1),
                'predicted_bikes_in': round(predictions_in.get(station_id, 0), 1)
            }
            
            if has_actual_data:
                result['bikes_out'] = int(row['bikes_out'])
                result['bikes_in'] = int(row['bikes_in'])
                result['net_flow'] = int(row['bikes_in'] - row['bikes_out'])
                result['predicted_net_flow'] = round(result['predicted_bikes_in'] - result['predicted_bikes_out'], 1)
            else:
                result['net_flow'] = round(result['predicted_bikes_in'] - result['predicted_bikes_out'], 1)
            
            results.append(result)
        
        if has_actual_data:
            results.sort(key=lambda x: x['bikes_out'], reverse=True)
        else:
            results.sort(key=lambda x: x['predicted_bikes_out'], reverse=True)
        
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return jsonify({
            'date': target_date,
            'has_actual_data': has_actual_data,
            'has_custom_weather': custom_weather is not None,
            'stations': results
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/predict_future", methods=["POST"])
def predict_future():
    try:
        data = request.get_json()
        station_id = int(data.get("station_id"))
        target_date = data.get("date")
        
        station_data = features_df[
            features_df['start_station_id'] == station_id
        ].iloc[-1]
        
        predictions = []
        for hour in range(24):
            feature_row = station_data[feature_cols].copy()
            feature_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            feature_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            X = feature_row.values.reshape(1, -1)
            pred = float(best_model.predict(X)[0])
            
            predictions.append({
                'hour': hour,
                'predicted_trips': round(pred, 1)
            })
        
        return jsonify({
            'station_name': str(station_data['start_station_name']),
            'date': target_date,
            'predictions': predictions
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route("/api/daily_report", methods=["POST"])
def get_daily_report():
    try:
        data = request.get_json()
        target_date = data.get("date")
        
        morning_df = features_df[
            (features_df['date'].astype(str) == target_date) &
            (features_df['hour'] == 6)
        ]
        
        evening_df = features_df[
            (features_df['date'].astype(str) == target_date) &
            (features_df['hour'] == 22)
        ]
        
        merged = morning_df.merge(
            evening_df[['start_station_id', 'trip_count']],
            on='start_station_id',
            suffixes=('_morning', '_evening')
        )
        
        merged['delta'] = merged['trip_count_evening'] - merged['trip_count_morning']
        merged['delta_pct'] = (merged['delta'] / merged['trip_count_morning'] * 100).fillna(0)
        
        results = []
        for _, row in merged.iterrows():
            results.append({
                'station_name': str(row['start_station_name']),
                'lat': float(row['start_station_latitude']),
                'lon': float(row['start_station_longitude']),
                'morning_trips': float(row['trip_count_morning']),
                'evening_trips': float(row['trip_count_evening']),
                'delta': float(row['delta']),
                'delta_pct': round(float(row['delta_pct']), 1)
            })
        
        return jsonify({
            'date': target_date,
            'stations': results
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)