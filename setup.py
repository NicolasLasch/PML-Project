# --- Libraries ---
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src import *

RAW_FILE_PATH = 'data/raw/bergen_merged.csv'
PROCESSED_FILE_PATH = 'data/processed'
OUTPUT_FOLDER = 'output'
OUTPUT_MODEL_FOLDER = 'models'
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
print("===== MODEL TRAINING, HYPERPARAMETER TUNING, EVALUATION =====")

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
print("Predicting with test set ... ... ...")
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
# Save Best Model
joblib.dump(best_model, f'models/{best_model_name}.joblib')
print(f"\nBest model saved as {best_model_name}.joblib")

def save_outputs(dfs, paths, with_index=None):
    for df, path in zip(dfs, paths):
        path.parent.mkdir(
            parents=True, 
            exist_ok=True
        )
        # Check if df should include index (by variable or id)
        index_flag = with_index is not None and (
            df is comparison_results or id(df) in with_index
        )
        df.to_csv(
            path,
            index=index_flag    # True for comparison_results, False for others
        )
        print(f"Saved {path}")

output_paths = [
    Path('output/model_comparison_results.csv'),
    Path('output/feature_importance.csv'),
    Path('output/predictions.csv'),
    Path('data/processed/processed_bergen_merged.csv'),
    Path('data/processed/feature_engineered_bergen_merged.csv')
]
outputs = [
    comparison_results, # Model Comparison Results
    importance_df,      # Feature Importance
    results_df,         # Predictions
    clean_bicycles_df,  # Processed Dataframe
    features_df         # Feature Engineered Dataframe
]
# Save outputs
save_outputs(outputs, output_paths, with_index = {id(comparison_results)})