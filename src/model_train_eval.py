from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np

def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest Regressor model on training data.
    - Optional Hyperparameter input
    """
    if params is None:
        # Default hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    # Create model with provided parameters
    model = RandomForestRegressor(**params)
    # Fit model to training data
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None):
    """
    Train a XGBoost regressor on provided training data
    - Optional Hyperparameter input
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    # Create model with provided parameters
    model = xgb.XGBRegressor(**params)
    # Fit model to training data
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using common regression metrics :
    (RMSE, MAE, R2)
    """
    # Model Prediction
    predictions = model.predict(X_test)
    # RMSE, MAE, R2
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def get_feature_importance(model, feature_names):
    """
    Gets a sorted Dataframe of model feature importances for interpretability
    """
    # Fetch importances and get DataFrame (for visualization)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return importance_df


def hyperparameter_tuning_rf(X_train, y_train, X_val, y_val):
    """
    [ Random Forest Hyperparameter Tuning ]
    - Manual Grid Search over Random Forest parameters
    - Trains and evaluates several models,
    selecting based on lowest validation RMSE
    """
    # List of parameter grids (n_estimators, max_depth)
    param_grid = [
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 15},
        {'n_estimators': 150, 'max_depth': 20},
        {'n_estimators': 200, 'max_depth': 25}
    ]
    # Initialize best score
    best_score = float('inf')
    best_model = None
    best_params = None
    
    for params in param_grid:
        # Train a RandomForest model and evaluate for each param setting
        model = train_random_forest(
            X_train,
            y_train,
            {**params, 'random_state': 42, 'n_jobs': -1}
        )
        metrics = evaluate_model(model, X_val, y_val)

        if metrics['RMSE'] < best_score:
            best_score = metrics['RMSE']
            best_model = model
            best_params = params
    # Best RandomForestRegressor, Dict of best-performing hyperparameters
    return best_model, best_params


def hyperparameter_tuning_xgb(X_train, y_train, X_val, y_val):
    """
    [ XGBoost Hyperparameter Tuning ]
    - Manual Grid Search over XGBoost Hyperparameters
    - Trains and evaluates several models,
    selecting based on lowest validation RMSE
    """
    # List of parameter grids (n_estimators, max_depth)
    param_grid = [
        {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.05},
        {'n_estimators': 150, 'max_depth': 12, 'learning_rate': 0.05}
    ]
    # Initialize best score
    best_score = float('inf')
    best_model = None
    best_params = None
    
    for params in param_grid:
        # Train a XGBoost model and evaluate for each param setting
        model = train_xgboost(X_train, y_train, {**params, 'random_state': 42, 'n_jobs': -1})
        metrics = evaluate_model(model, X_val, y_val)
        if metrics['RMSE'] < best_score:
            best_score = metrics['RMSE']
            best_model = model
            best_params = params
    # Best XGBRegressor, Dict of best-performing hyperparameters
    return best_model, best_params


def compare_models(models, X_test, y_test):
    """
    Model comparison of trained models on test set (metrics compilation)
    """
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test) # RMSE, MAE, R2
        # Label Results with model name
        metrics['model'] = name
        results.append(metrics)
    # Dataframe of model metrics indexed by model name
    return pd.DataFrame(results).set_index('model')