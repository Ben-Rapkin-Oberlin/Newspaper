import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    median_absolute_error, max_error, explained_variance_score
)
from sklearn.model_selection import GridSearchCV
from collections import defaultdict

# A constant offset for targets (if needed)
LOG_OFFSET = 1

def create_lag_features(df, target_col, lag_periods=[1, 2]):
    """Create lag features for time series prediction"""
    df = df.copy()
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df.dropna()

def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def find_best_xgboost_params(X_train, y_train):
    """Perform grid search for XGBoost with comprehensive parameter grid"""
    print('XGBoost parameter search in progress...')
    param_grid = {
        'n_estimators': [20,40],
        'max_depth': [1],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1],
        'min_child_weight': [1, 3, 4]
    }
    
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        objective='reg:squarederror'
    )
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest XGBoost parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate comprehensive set of regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        f'{prefix}mae': mean_absolute_error(y_true, y_pred),
        f'{prefix}mse': mse,
        f'{prefix}rmse': np.sqrt(mse),
        f'{prefix}r2': r2_score(y_true, y_pred),
        f'{prefix}median_ae': median_absolute_error(y_true, y_pred),
        f'{prefix}max_error': max_error(y_true, y_pred),
        f'{prefix}explained_variance': explained_variance_score(y_true, y_pred),
        f'{prefix}mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else np.nan
    }
    return metrics

def select_top_features(X, model, n_features=None, threshold=None):
    """
    Select top features based on feature importance from XGBoost model.
    Either selects the top n_features or all features with importance > threshold.
    """
    feature_names = X.columns
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")
    
    if n_features is not None:
        selected_features = feature_importance.head(n_features)['Feature'].tolist()
        print(f"\nSelected top {n_features} features:")
    elif threshold is not None:
        selected_features = feature_importance[feature_importance['Importance'] > threshold]['Feature'].tolist()
        print(f"\nSelected features with importance > {threshold}:")
    else:
        selected_features = feature_importance[feature_importance['Importance'] > 0]['Feature'].tolist()
        print("\nSelected features with non-zero importance:")
    
    print(", ".join(selected_features))
    print(f"Number of selected features: {len(selected_features)}")
    
    return selected_features

def evaluate_xgboost_rotation(X, y, years, val_size, xgb_params, selected_features=None):
    """
    Evaluate XGBoost model on a specific rotation of the data with log-transform on the target.
    The target is transformed as: y_log = log(y + LOG_OFFSET)
    Predictions are then converted back: y_pred = exp(predicted_log) - LOG_OFFSET
    """
    if selected_features is not None:
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features for evaluation")
    
    # Scale features
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply log transform to the target with offset
    y_log = np.log(y + LOG_OFFSET)
    
    # Split into training and validation sets
    X_train = X_scaled[:-val_size]
    y_train = y_log[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y_log[-val_size:]
    val_years = years[-val_size:].values
    
    # Train XGBoost in log-space
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    train_pred_log = xgb_model.predict(X_train)
    val_pred_log = xgb_model.predict(X_val)
    
    y_train_pred = np.exp(train_pred_log) - LOG_OFFSET
    y_val_pred = np.exp(val_pred_log) - LOG_OFFSET
    
    y_train_actual = np.exp(y_train) - LOG_OFFSET
    y_val_actual = np.exp(y_val) - LOG_OFFSET
    
    train_metrics = calculate_metrics(y_train_actual, y_train_pred, 'train_')
    val_metrics = calculate_metrics(y_val_actual, y_val_pred, 'val_')
    metrics = {**train_metrics, **val_metrics}
    
    feature_importance = {name: imp for name, imp in zip(X.columns, xgb_model.feature_importances_)}
    
    return {
        'years': {'val': val_years},
        'predictions': {'train': y_train_pred, 'val': y_val_pred},
        'actual': {'train': y_train_actual, 'val': y_val_actual},
        'metrics': metrics,
        'model': xgb_model,
        'scalers': {'x': x_scaler},
        'feature_importance': feature_importance,
        'feature_names': X.columns.tolist()
    }

def save_model(model, scalers, file_path="xgboost_model.pkl", feature_names=None):
    """Save the trained model and scaler for later use."""
    model_data = {
        'model': model,
        'x_scaler': scalers['x'],
        'feature_names': feature_names
    }
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {file_path}")

def load_model(file_path="xgboost_model.pkl"):
    """Load a previously trained model and its scaler.
       If y_scaler is missing, raise a KeyError so we can retrain."""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    
    feature_names = model_data.get('feature_names', None)
    # Try to load y_scaler. If missing, raise KeyError.
    try:
        y_scaler = model_data['y_scaler']
    except KeyError:
        raise KeyError("y_scaler not found in model data")
    
    if feature_names:
        print(f"Model was trained with {len(feature_names)} features: {', '.join(feature_names)}")
    
    return model_data['model'], model_data['x_scaler'], y_scaler, feature_names

def plot_feature_importance(feature_importance_dict, title="XGBoost Feature Importance", save_path=None):
    """Plot feature importance from the XGBoost model."""
    items = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    if not items:
        print("No feature importance values to plot.")
        return
    features, importance = zip(*items)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importance, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt

def main():
    base_dir = r"C:\Users\benra\Documents\Newspaper"
    output_dir = os.path.join(base_dir, "regression_finals")
    data_dir = os.path.join(base_dir, "yearly_occurrence_data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    train_path = os.path.join(data_dir, "training_data_1900_1936.csv")
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)
    
    # Add lag features
    train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
    
    rotation_size = 1
    val_size = 5
    n_rotations = len(train_df_with_lags) // rotation_size
    
    X = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    y = train_df_with_lags['estimated_deaths']
    years = train_df_with_lags['year']
    
    print(f"Dataset shape: {X.shape}")
    
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    retrain = False
    # Try loading the model. If it fails because 'y_scaler' is missing, automatically retrain.
    try:
        model, x_scaler, y_scaler, feature_names = load_model(model_path)
        user_choice = input("Do you want to retrain the model? (y/n): ").lower()
        retrain = (user_choice == 'y')
    except KeyError as e:
        print(f"Error loading model: {e}. Automatically retraining the model.")
        retrain = True
    
    if retrain:
        print("Finding best XGBoost parameters...")
        best_xgb_params = find_best_xgboost_params(X, y)
        
        feature_selection_enabled = True
        if feature_selection_enabled:
            initial_model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
            initial_model.fit(X, y)
            n_features = int(input("Enter number of top features to select (0 to use threshold instead): "))
            if n_features > 0:
                selected_features = select_top_features(X, initial_model, n_features=n_features)
            else:
                threshold = float(input("Enter importance threshold for feature selection: "))
                selected_features = select_top_features(X, initial_model, threshold=threshold)
        else:
            selected_features = None
        
        results = {}
        validation_predictions = defaultdict(list)
        validation_actuals = defaultdict(list)
        all_feature_importance = {}
        
        last_model = None
        last_scalers = None
        last_feature_names = None
        
        for i in range(n_rotations):
            if i > 0:
                X = rotate_series(X, rotation_size)
                y = rotate_series(pd.DataFrame(y), rotation_size).iloc[:, 0]
                years = rotate_series(pd.DataFrame(years), rotation_size).iloc[:, 0]
            
            key = f"Rotation {i+1}"
            print(f"\nEvaluating {key}...")
            results[key] = evaluate_xgboost_rotation(
                X, y, years,
                val_size=val_size,
                xgb_params=best_xgb_params,
                selected_features=selected_features
            )
            
            for feature, importance in results[key]['feature_importance'].items():
                if feature not in all_feature_importance:
                    all_feature_importance[feature] = []
                all_feature_importance[feature].append(importance)
            
            val_years = results[key]['years']['val']
            val_actuals = results[key]['actual']['val']
            val_preds = results[key]['predictions']['val']
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append(pred)
                validation_actuals[year].append(actual)
            
            if i == n_rotations - 1:
                last_model = results[key]['model']
                last_scalers = results[key]['scalers']
                last_feature_names = results[key]['feature_names']
        
        if last_model is not None:
            # Note: Since this log-transform model doesn't save y_scaler, we only save x_scaler and feature_names.
            save_model(last_model, last_scalers, model_path, last_feature_names)
        
        avg_feature_importance = {feature: np.mean(vals) for feature, vals in all_feature_importance.items()}
        feature_importance_path = os.path.join(output_dir, 'xgboost_feature_importance.png')
        plot_feature_importance(avg_feature_importance, title="XGBoost Feature Importance", save_path=feature_importance_path)
        
        aggregated_metrics = {}
        for year in validation_actuals.keys():
            predictions = validation_predictions[year]
            actuals = validation_actuals[year]
            aggregated_metrics[year] = {
                'actual': np.mean(actuals),
                'xgb_pred': np.mean(predictions)
            }
        agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
        agg_df.index.name = 'year'
        agg_df = agg_df.sort_index()
        
        print("\nXGBOOST Validation Metrics:")
        metrics = calculate_metrics(agg_df['actual'], agg_df['xgb_pred'])
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.3f}")
        print(f"Median AE: {metrics['median_ae']:.2f}")
        print(f"Max Error: {metrics['max_error']:.2f}")
        print(f"Explained Variance: {metrics['explained_variance']:.3f}")
        if not np.isnan(metrics['mape']):
            print(f"MAPE: {metrics['mape']:.2f}%")
        
        plt.figure(figsize=(15, 8))
        plt.plot(agg_df.index, agg_df['actual'], 'ko-', label='Actual', linewidth=2)
        plt.plot(agg_df.index, agg_df['xgb_pred'], 's--', label='XGBoost', color='red', alpha=0.7)
        plt.title('XGBoost: Aggregated Validation Predictions vs Actual Values')
        plt.xlabel('Year')
        plt.ylabel('Estimated Deaths')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_validation_predictions.png'))
        plt.show()
        
        plt.figure(figsize=(15, 6))
        errors = np.abs(agg_df['xgb_pred'] - agg_df['actual'])
        plt.plot(agg_df.index, errors, 'o-', label='XGBoost Error', color='red', alpha=0.7)
        plt.title('XGBoost: Absolute Prediction Errors Over Time (Validation)')
        plt.xlabel('Year')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_validation_errors.png'))
        plt.show()
        
        print("\nEvaluating on full training dataset...")
        train_df = pd.read_csv(train_path)
        train_df = train_df.sort_values("year").reset_index(drop=True)
        train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
        X_all = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
        y_all = train_df_with_lags['estimated_deaths']
        
        if selected_features:
            X_all = X_all[selected_features]
            print(f"Using {len(selected_features)} selected features for full dataset evaluation: {', '.join(selected_features)}")
        else:
            print("Using all features for full dataset evaluation")
        
        X_all_scaled = last_scalers['x'].transform(X_all)
        pred_log = last_model.predict(X_all_scaled)
        y_pred_full = np.exp(pred_log) - LOG_OFFSET
        
        full_metrics = calculate_metrics(y_all, y_pred_full)
        print("\nXGBOOST Full Dataset Metrics:")
        print(f"MAE: {full_metrics['mae']:.2f}")
        print(f"RMSE: {full_metrics['rmse']:.2f}")
        print(f"R²: {full_metrics['r2']:.3f}")
        print(f"Median AE: {full_metrics['median_ae']:.2f}")
        print(f"Max Error: {full_metrics['max_error']:.2f}")
        print(f"Explained Variance: {full_metrics['explained_variance']:.3f}")
        if not np.isnan(full_metrics['mape']):
            print(f"MAPE: {full_metrics['mape']:.2f}%")
        
        plt.figure(figsize=(15, 8))
        plt.plot(train_df_with_lags['year'], y_all, 'ko-', label='Actual', linewidth=2)
        plt.plot(train_df_with_lags['year'], y_pred_full, '^--', label='XGBoost', color='red', alpha=0.7)
        plt.title('XGBoost: Full Dataset Predictions vs Actual Values')
        plt.xlabel('Year')
        plt.ylabel('Estimated Deaths')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_full_predictions.png'))
        plt.show()
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        residuals = y_all - y_pred_full
        plt.scatter(y_pred_full, residuals, alpha=0.5, color='red')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title('XGBoost Residuals vs Predicted (Full Dataset)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.hist(residuals, bins=20, alpha=0.7, color='red')
        plt.title('XGBoost Residuals Histogram (Full Dataset)')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_full_residuals.png'))
        plt.show()
    else:
        print("Using existing model without retraining.")
        # You can add similar evaluation code here if needed.
        
if __name__ == "__main__":
    main()
