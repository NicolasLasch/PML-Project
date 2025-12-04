import joblib
import pandas as pd
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from src import get_feature_columns
from src.station_reco import recommend_best_station

app = Flask(__name__)

best_model = joblib.load("models/XGB_Tuned.joblib")
feature_cols = get_feature_columns()

sample_df = pd.read_csv("data/processed/sample_features.csv")
features_df = pd.read_csv("data/processed/feature_engineered_bergen_merged.csv")
comparison_df = pd.read_csv("output/model_comparison_results.csv", index_col=0)

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
    
    stations = features_df.groupby('start_station_id').agg({
        'start_station_name': 'first',
        'start_station_latitude': 'first',
        'start_station_longitude': 'first',
        'trip_count': 'mean'
    }).reset_index()
    
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
                    'avg_trips': round(float(row['trip_count']), 1)
                }
            }
            for _, row in stations.iterrows()
        ]
    }
    
    return jsonify({
        'metrics': {
            'best_model': best_model_name,
            'total_stations': int(sample_df['start_station_id'].nunique()),
            'total_samples': len(sample_df),
            'avg_trips': round(float(sample_df['trip_count'].mean()), 1),
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
        
        timeseries_all = {}
        last_50 = sample_df.tail(50)
        
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
            
            predictions = model.predict(last_50[feature_cols])
            timeseries_all[name] = [
                {
                    'time': f"{row['date']} {int(row['hour']):02d}:00",
                    'actual': float(row['trip_count']),
                    'predicted': float(predictions[idx])
                }
                for idx, (_, row) in enumerate(last_50.iterrows())
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


@app.route("/api/recommend", methods=["POST"])
def recommend_station():
    try:
        data = request.get_json()
        user_lat = float(data.get("lat", 60.3913))
        user_lon = float(data.get("lon", 5.3221))
        target_hour = int(data.get("hour", 8))
        
        recommendations = recommend_best_station(
            user_lat=user_lat,
            user_lon=user_lon,
            target_hour=target_hour,
            stations_info=features_df,
            features_df=features_df,
            model=best_model,
            feature_cols=feature_cols,
            top_k=5
        )
        
        if recommendations.empty:
            return jsonify({'error': 'No stations found nearby'}), 404
        
        best_station = recommendations.iloc[0]
        
        rec_list = []
        for _, rec in recommendations.iterrows():
            rec_list.append({
                'station_id': int(rec['station_id']),
                'station_name': str(rec['station_name']),
                'lat': float(rec['latitude']),
                'lon': float(rec['longitude']),
                'distance_km': round(float(rec['distance_km']), 2),
                'predicted_trips': round(float(rec['predicted_trips']), 1),
                'score': round(float(rec['recommendation_score']), 3)
            })
        
        return jsonify({
            'user_location': {'lat': user_lat, 'lon': user_lon},
            'target_hour': target_hour,
            'best_station': {
                'name': str(best_station['station_name']),
                'lat': float(best_station['latitude']),
                'lon': float(best_station['longitude']),
                'distance_km': round(float(best_station['distance_km']), 2),
                'predicted_trips': round(float(best_station['predicted_trips']), 1),
                'score': round(float(best_station['recommendation_score']), 3)
            },
            'recommendations': rec_list
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)