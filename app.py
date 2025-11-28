import joblib
import pandas as pd
import traceback
from flask import Flask, request, jsonify, render_template

from src import get_feature_columns

app = Flask(__name__)

best_model = joblib.load("models/XGB_Tuned.joblib")
feature_cols = get_feature_columns()

# Load some real rows to pick from
sample_df = pd.read_csv("data/processed/sample_features.csv")


@app.route("/", methods=["GET", "POST"])
def dashboard():
    """
    Simple web dashboard:
    - GET: show form
    - POST: take raw inputs, run pipelines, show prediction
    """
    prediction = None
    error = None
    selected_idx = None
    selected_row = None

    # Build labels
    options = []
    for i, row in sample_df.iterrows():
            station = row.get('start_station_name', 'Unknown')
            date = row.get('date', '?')
            hour = row.get('hour', '?')
            trips = row.get('trip_count', '?')
            label = f"{i}: {station} | {date} {hour}h | Actual: {trips}"
            options.append((i, label))

    if request.method == "POST":
        try:
            selected_idx = int(request.form["row_index"])
            row_data = sample_df.iloc[selected_idx]
            
            X = row_data[feature_cols].values.reshape(1, -1)

            y_pred = best_model.predict(X)[0]
            prediction = round(float(y_pred), 4)

            selected_row = row_data.to_dict()

        except Exception as e:
            print("ERROR:", str(e))
            print(traceback.format_exc())
            error = str(e)

    # Optional: placeholder metrics or chart paths
    metrics = None
    fi_chart_path = None
    hourly_chart_path = None

    return render_template(
        "index.html",
        options = options,
        selected_idx = selected_idx,
        selected_row = selected_row,
        prediction = prediction,
        error = error,
        metrics = metrics,
        fi_chart_path = fi_chart_path,
        hourly_chart_path = hourly_chart_path,
    )


@app.route("/health", methods = ["GET"])
def health():
    """
    Simple health check
    """
    return jsonify({
        "sttaus" : "ok",
        "model_loaded" : best_model is not None
    })


@app.route("/predict", methods = ["POST"])
def predict_api():
    """
    JSON API endpoint, expects JSON:
    {
      "features": {
        "start_time": "2025-03-01 08:00:00",
        "end_time": "2025-03-01 08:10:00",
        "duration": 600,
        "start_station_id": 101,
        "start_station_name": "Bergen Central",
        "start_station_latitude": 60.3913,
        "start_station_longitude": 5.3221,
        "end_station_id": 102,
        "end_station_name": "Station B",
        "end_station_latitude": 60.3920,
        "end_station_longitude": 5.3240,
        "temperature": 12.3,
        "max_temperature": 15.0,
        "min_temperature": 8.0,
        "wind_speed": 3.5,
        "precipitation": 0.2,
        "humidity": 65.0,
        "weather": "Cloudy",
        "sunshine": 3.0,
        "season": 2,
        "is_holiday": 0,
        "is_weekend": 1
      }
    }    """
    try:
        data = request.get_json()
        if data is None or "features" not in data:
            return jsonify(
                {"error" : "Request JSON must contain 'features'"}
            ),400

        raw = data["features"]

        raw_df = pd.DataFrame([raw])

        # Pipelines
        clean_df = preprocess_pipeline(raw_df)
        feature_df = feature_engineering_pipeline(clean_df)

        # Ensure correct order/columns
        X = feature_df[feature_cols]

        # Predict
        y_pred = best_model.predict(X)[0]

        return jsonify({"predicted_trip_counts" : float(y_pred)})

    except Exception as e:
        return jsonify({"error" : str(e)}), 500


def main():
    app.run(host = "0.0.0.0", port = 8000)

if __name__ == "__main__":
    main()
