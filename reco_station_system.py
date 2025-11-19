from src.station_reco import recommend_best_station, format_recommendation_output, get_feature_columns
import pandas as pd
import joblib

# Run setup.py before executing to get best model and dataframes for recommendations if needed

try:
    # Load feature dataframe
    features_df = pd.read_csv("data/processed/feature_engineered_bergen_merged.csv")
    # Load trained model
    best_model = joblib.load("models/XGB_Tuned.joblib")
    # Get feature columns
    feature_cols = get_feature_columns()

    # Set user location info and use-case parameters (Example)
    user_latitude = 60.3913
    user_longitude = 5.3221
    target_hour = 8
    print(f"User Location: ({user_latitude}, {user_longitude})")
    print(f"Target Hour: {target_hour}:00")

    # Get recommendations for nearest stations
    recommendations = recommend_best_station(
        user_lat=user_latitude,
        user_lon=user_longitude,
        target_hour=target_hour,
        stations_info=features_df,
        features_df=features_df,
        model=best_model,
        feature_cols=feature_cols,
        top_k=5
    )

    print(format_recommendation_output(recommendations))

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}. Run setup.py before this script if needed.")

except Exception as e:  # Catch any other errors
    print(f"An unexpected error occurred: {e}. Make sure setup.py has run and all dependencies exist.")