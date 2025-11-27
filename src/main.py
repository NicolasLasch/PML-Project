# --- Libraries ---
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src import *
from src.model_train_eval import train_random_forest, train_xgboost, hyperparameter_tuning_rf, hyperparameter_tuning_xgb, compare_models, evaluate_model, get_feature_importance
from src.constants.file_path_constants import *


def run_full_pipeline():
    """
    ML pipeline: 
    Load → Process → Train → Save
    """
    # Raw Data File Path
    RAW_FILE_PATH = 'data/raw/bergen_merged.csv'
    # Train, Val and Test Ratios
    TRAIN_VAL_TEST_RATIO = 0.2
    TRAIN_VAL_RATIO = 0.25

    # --- Dataset loading ---
    bicycles_df = pd.read_csv(RAW_FILE_PATH)

    print(f"[ LOAD DATAFRAME ]")
    print(f"Dataset Shape: {bicycles_df.shape}")
    print(f"Total Records: {bicycles_df.shape[0]:,}")
    print(f"Features: {bicycles_df.shape[1]}\n\n")



    # --- Process Data ---
    clean_bicycles_df = preprocess_pipeline(bicycles_df)

    print("[ DATA PROCESSING ]")
    print(f"After Preprocessing: {len(clean_bicycles_df):,}")
    print(f"Records Removed: {len(bicycles_df) - len(clean_bicycles_df):,}\n\n")

    # --- Feature Engineering ---
    features_df = feature_engineering_pipeline(clean_bicycles_df)

    print("[ FEATURE ENGINEERING ]")
    print(f"Feature DataFrame Shape: {features_df.shape}")
    print(f"Total Samples (station-hour combinations): {len(features_df):,}\n")

    print("Feature Columns:")
    feature_cols = get_feature_columns()
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2}. {col}")



    # --- Train-Validation-Test-Split ---
    # TrainVal and Test Split
    train_val_df, test_df = prepare_train_test_split(features_df, test_ratio = TRAIN_VAL_TEST_RATIO)
    # Train and Test Split
    train_df, val_df = prepare_train_test_split(train_val_df, test_ratio = TRAIN_VAL_RATIO)

    print("\n[ TRAINING, VALIDATION, TEST SETS ]")
    print(f"Training Set: {len(train_df):,} samples")
    print(f"Validation Set: {len(val_df):,} samples")
    print(f"Test Set: {len(test_df):,} samples")
    print(f"\nTotal: {len(train_df) + len(val_df) + len(test_df):,} samples")

    # Feature matrix and labels (target) for training set
    X_train = train_df[feature_cols]    # Training features
    y_train = train_df['trip_count']    # Training target (trip counts)

    # Feature matrix and labels for validation set
    X_val = val_df[feature_cols]    # Validation Features
    y_val = val_df['trip_count']    # Validation target

    # Feature matrix and labels for test set
    X_test = test_df[feature_cols]  # Test Features
    y_test = test_df['trip_count']  # Test target

    # (n_samples, n_features)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}\n")



    # --- Model Training, Hyperparameter Tuning ---
    print("\n===== MODEL TRAINING, HYPERPARAMETER TUNING, EVALUATION =====")

    def train_and_tune_report(model_name, train_func, tuning_func, X_train, y_train, X_val, y_val):
        # --- Model Training
        print(f"\n[ {model_name} Baseline Model ]")
        print(f"Training {model_name} Model ... ... ...")
        baseline_model = train_func(X_train, y_train)
        # RMSE, MAE, R2
        baseline_metrics = evaluate_model(baseline_model, X_val, y_val)
        print(f"Baseline {model_name} Performance (Validation):")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # --- Hyperparameter tuning
        print(f"\n[- Tuned {model_name} Model -]")
        print(f"Tuning {model_name} hyperparameters ... ... ...")
        tuned_model, tuned_params = tuning_func(X_train, y_train, X_val, y_val)
        # Best Parameters
        print(f"\nBest {model_name} Parameters:")
        for param, value in tuned_params.items():
            print(f"  {param}: {value}")
        # Tuned Model Performance
        tuned_metrics = evaluate_model(tuned_model, X_val, y_val)
        print(f"\nTuned {model_name} Performance (Validation):")
        for metric, value in tuned_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Base Model + Performance, Tuned Model + Best Parameters + Performance
        return baseline_model, tuned_model, tuned_params, baseline_metrics, tuned_metrics

    # Random Forest
    rf_base, rf_tuned, rf_params, rf_base_metrics, rf_tuned_metrics = train_and_tune_report(
        "Random Forest",
        train_random_forest, 
        hyperparameter_tuning_rf, 
        X_train, y_train, X_val, y_val
    )
    # XGBoost
    xgb_base, xgb_tuned, xgb_params, xgb_base_metrics, xgb_tuned_metrics = train_and_tune_report(
        "XGBoost", 
        train_xgboost, 
        hyperparameter_tuning_xgb, 
        X_train, y_train, X_val, y_val
    )



    # --- Model Comparison, Final Evaluation, Best Model ---
    # Models Dictionary
    models = {
        'RF_Baseline': rf_base,
        'RF_Tuned': rf_tuned,
        'XGB_Baseline': xgb_base,
        'XGB_Tuned': xgb_tuned
    }
    # Model copmarison
    comparison_results = compare_models(models, X_test, y_test)
    print("\n[ MODEL COMPARISON ]")
    print(comparison_results)



    # --- Best Model ---
    best_model_name = comparison_results['RMSE'].idxmin()
    best_model = models[best_model_name]
    print("\n[ BEST MODEL ]")
    print(f"Best Model Name: {best_model_name}")
    print(f"Test RMSE: {comparison_results.loc[best_model_name, 'RMSE']:.4f}")
    print(f"Test MAE: {comparison_results.loc[best_model_name, 'MAE']:.4f}")
    print(f"Test R²: {comparison_results.loc[best_model_name, 'R2']:.4f}")



    # --- Feature Importance ---
    importance_df = get_feature_importance(best_model, feature_cols)
    print("\n[ FEATURE IMPORTANCE]")
    print("Top 15 Most Important Features:")
    print(importance_df.head(15))



    # --- Model Prediction ---
    print("\n[ MODEL PREDICTION ]")
    print("Predicting with test set (first 20 predictions) ... ... ...")
    y_pred = best_model.predict(X_test)
    # Actual and Predicted (trip_count)
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results_df.head(20))



    # --- Model Failure Analysis ---
    test_df_analysis = test_df.copy()
    test_df_analysis['prediction'] = y_pred
    # Absolute error between actual trip count and predicted value
    test_df_analysis['absolute_error'] = np.abs(test_df_analysis['trip_count'] - test_df_analysis['prediction'])

    # Threshold for high error cases (95th percentile of absolute errors)
    high_error_threshold = np.percentile(test_df_analysis['absolute_error'], 95)
    # Filter rows where absolute error > high error threshold
    high_error_cases = test_df_analysis[test_df_analysis['absolute_error'] > high_error_threshold]

    print("\n[ MODEL FAILURE ANALYSIS ]")
    print(f"High Error Cases (top 5% errors, threshold: {high_error_threshold:.2f}):")
    print(f"Number of cases: {len(high_error_cases)}")
    print("\nSample of high-error predictions:")
    high_error_cases[
        ['start_station_name', 'date', 'hour', 'trip_count', 'prediction',
        'absolute_error', 'temperature', 'precipitation', 'is_rush_hour']
    ].head(10)

    print("\nAnalysis of High Error Cases:")
    print(f"  Average actual trips: {high_error_cases['trip_count'].mean():.2f}")
    print(f"  Average predicted trips: {high_error_cases['prediction'].mean():.2f}")
    print(f"  Rush hour percentage: {high_error_cases['is_rush_hour'].mean()*100:.1f}%")
    print(f"  Rainy conditions percentage: {(high_error_cases['precipitation'] > 0).mean()*100:.1f}%")
    print(f"  Holiday percentage: {high_error_cases['is_holiday'].mean()*100:.1f}%")



    # --- Saving Outputs & Best Model ---

    def save_outputs(outputs : dict, with_index = None):

        for path, output in outputs.items():
            if isinstance(output, pd.DataFrame):
                # Save Dataframe
                Path(path).parent.mkdir(
                    parents=True, 
                    exist_ok=True
                )
                # Check if df should include index (by variable or id)
                index_flag = with_index is not None and (
                    output is comparison_results or id(output) in with_index
                )
                output.to_csv(
                    path,
                    index=index_flag    # True for comparison_results, False for others
                )
                print(f"Saved {path}")
            # Save Charts
            elif isinstance(output, plt.Figure):
                full_path = f"{OUTPUT_CHARTS_FOLDER}/{path}"
                Path(full_path).parent.mkdir(
                    parents = True,
                    exist_ok = True
                )
                output.savefig(full_path)
                print(f"Saved chart to {full_path}")
            
            else:
                print(f"Skipping unsupported type for {path}")


    # Save Best Model
    print("\nSaving Models / Results ... ... ...")
    joblib.dump(best_model, f'models/{best_model_name}.joblib')
    print(f"Best model saved as {best_model_name}.joblib")
    # Save output csv
    outputs = {
        MC_FILE_PATH : comparison_results,
        FI_FILE_PATH : importance_df,
        PRED_FILE_PATH : results_df,
        PROCESSED_FILE_PATH : clean_bicycles_df,
        FE_FILE_PATH : features_df
    }
    save_outputs(outputs, with_index = {id(comparison_results)})



    # --- Visualizations --- (optional)
    print("\nConfiguring Exploratory Data Analysis ... ... ...")
    # Feature Importance
    feature_importance_chart = plot_feature_importance(importance_df)
    # Predictions and Residuals
    predictions_chart = plot_actual_vs_predicted(y_test, y_pred)
    residuals_chart = plot_residuals(y_test, y_pred)
    # Patterns
    hourly_patterns_chart = plot_hourly_patterns(features_df)
    weather_correlation_chart = plot_weather_correlation(features_df)
    seasonal_patterns_chart = plot_seasonal_patterns(features_df)
    # Model Comparison Results
    model_comparison_chart = plot_model_comparison(comparison_results)

            
    # Initiate Charts dict (key = path, value = figure)
    charts = {
        MC_CHART : model_comparison_chart,
        FI_CHART : feature_importance_chart,
        PRED_CHART : predictions_chart,
        HOURLY_PATTERN_CHART : hourly_patterns_chart,
        WEATHER_CORR_HEATMAP : weather_correlation_chart,
        SEASONAL_PATTERN_CHART : seasonal_patterns_chart
    }
    # Save Charts
    save_outputs(charts)

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Best Model: {best_model_name}")
    print(f"Models saved to: models/")
    print(f"Outputs saved to: output/")