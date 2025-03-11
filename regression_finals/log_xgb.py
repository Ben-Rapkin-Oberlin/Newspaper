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

# A constant offset to allow log-transform even when y=0.
LOG_OFFSET = 1

def create_lag_features(df, target_col, lag_periods=[1, 2], keep_early_years=False):
    """
    Create lag features for time series prediction
    
    Parameters:
    df - DataFrame with time series data
    target_col - Name of target column to create lags for
    lag_periods - List of lag periods to create
    keep_early_years - If True, keep early years by filling NaN values with 0
    """
    df = df.copy()
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    if keep_early_years:
        # Fill NaN values with 0 instead of dropping rows
        return df.fillna(0)
    else:
        # Original behavior: drop rows with NaN
        return df.dropna()
def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning."""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def find_best_xgboost_params(X_train, y_train):
    """Perform grid search for XGBoost with various loss functions and regularization parameters."""
    print('XGBoost parameter search in progress...')
    """
    Overall best parameters:
 {'colsample_bytree': 0.5, 'learning_rate': 0.2, 'max_depth': 1, 'min_child_weight': 1, 'n_estimators': 30, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.75}
    """
    # Define base parameter grid
    param_grid = {
        'n_estimators': [30],
        'max_depth': [1],
        'learning_rate': [0.2],
        'subsample': [0.75],
        'colsample_bytree': [0.5],
        'min_child_weight': [1],
        # Adding regularization parameters
        'reg_alpha': [0],  # L1 regularization
        'reg_lambda': [0.1],  # L2 regularization
    }
    
    # Define different objective functions to try
    objectives = [
        'reg:squarederror',  # MSE
        #'reg:absoluteerror',  # MAE
        #'reg:squaredlogerror',  # Huber loss, more robust to outliers
        #'reg:pseudohubererror'  # Logistic loss, might help with scaling issues
    ]
    
    best_score = float('inf')
    best_params = None
    best_objective = None
    
    for objective in objectives:
        print(f"\nTrying objective: {objective}")
        
        xgb_model = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective=objective
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
        
        current_score = -grid_search.best_score_
        print(f"Best MAE with {objective}: {current_score}")
        print(f"Parameters: {grid_search.best_params_}")
        
        if current_score < best_score:
            best_score = current_score
            best_params = grid_search.best_params_
            best_objective = objective
    
    # Print overall best results
    print("\n" + "="*50)
    print(f"Overall best objective function: {best_objective}")
    print(f"Overall best parameters: {best_params}")
    print(f"Overall best MAE score: {best_score}")
    
    # Return complete best parameters including the objective
    complete_best_params = best_params.copy()
    complete_best_params['objective'] = best_objective
    
    return complete_best_params

def calculate_metrics(y_true, y_pred, prefix=''):
    """Calculate a comprehensive set of regression metrics."""
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
    Select top features based on feature importances from the XGBoost model.
    Either select the top n_features or those with importance greater than threshold.
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
    Evaluate the XGBoost model on a specific rotation of the data using a log transform on the target.
    """
    if selected_features is not None:
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features for evaluation")
    
    # Scale features
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply log transform to the target with an offset
    y_log = np.log(y + LOG_OFFSET)
    
    # Split into training and validation sets
    X_train = X_scaled[:-val_size]
    y_train = y_log[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y_log[-val_size:]
    val_years = years[-val_size:].values
    
    # Create and train the XGBoost model
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Predict in log-space and invert transform
    train_pred_log = xgb_model.predict(X_train)
    val_pred_log = xgb_model.predict(X_val)
    
    y_train_pred = np.exp(train_pred_log) - LOG_OFFSET
    y_val_pred = np.exp(val_pred_log) - LOG_OFFSET
    
    # Inverse transform the actual values for metric calculation
    y_train_actual = np.exp(y_train) - LOG_OFFSET
    y_val_actual = np.exp(y_val) - LOG_OFFSET
    
    # Calculate metrics on the original scale
    train_metrics = calculate_metrics(y_train_actual, y_train_pred, 'train_')
    val_metrics = calculate_metrics(y_val_actual, y_val_pred, 'val_')
    metrics = {**train_metrics, **val_metrics}
    
    # Gather feature importances
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



def add_early_year_predictions(train_df, model, x_scaler, selected_features, output_plot_path=None):
    """Add predictions for early years (1900-1901) that might not be in train_predictions"""
    
    # Find years 1900-1901 in the original data
    early_years = train_df[train_df['year'].isin([1900, 1901])]
    
    if early_years.empty:
        print("Years 1900-1901 not found in train_df")
        return {}
    
    # Create lagged features with keep_early_years=True
    early_df = create_lag_features(early_years, 'estimated_deaths', lag_periods=[1, 2], keep_early_years=True)
    
    # Get features and prepare them for prediction
    X_early = early_df.drop(['year', 'estimated_deaths'], axis=1)
    
    if selected_features is not None:
        X_early = X_early[selected_features]
    
    # Scale features
    X_early_scaled = x_scaler.transform(X_early)
    
    # Predict in log-space and convert back
    early_pred_log = model.predict(X_early_scaled)
    early_predictions = np.exp(early_pred_log) - LOG_OFFSET
    
    # Create a dictionary mapping years to predictions
    early_years_dict = {year: pred for year, pred in zip(early_df['year'], early_predictions)}
    
    # Optional - plot to verify
    if output_plot_path:
        plt.figure(figsize=(10, 6))
        plt.plot(early_df['year'], early_df['estimated_deaths'], 'ko-', label='Actual')
        plt.plot(early_df['year'], early_predictions, 'b^--', label='Model Predictions')
        plt.title('Early Years (1900-1901) Predictions')
        plt.xlabel('Year')
        plt.ylabel('Estimated Deaths')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_plot_path)
        plt.close()
    
    return early_years_dict



def evaluate_xgboost_rotation(X, y, years, val_size, xgb_params, selected_features=None):
    """
    Evaluate the XGBoost model on a specific rotation of the data using a log transform on the target.
    """
    if selected_features is not None:
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features for evaluation")
    
    # Scale features
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply log transform to the target with an offset
    y_log = np.log(y + LOG_OFFSET)
    
    # Split into training and validation sets
    X_train = X_scaled[:-val_size]
    y_train = y_log[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y_log[-val_size:]
    val_years = years[-val_size:].values
    
    # Create and train the XGBoost model
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Predict in log-space and invert transform
    train_pred_log = xgb_model.predict(X_train)
    val_pred_log = xgb_model.predict(X_val)
    
    y_train_pred = np.exp(train_pred_log) - LOG_OFFSET
    y_val_pred = np.exp(val_pred_log) - LOG_OFFSET
    
    # Inverse transform the actual values for metric calculation
    y_train_actual = np.exp(y_train) - LOG_OFFSET
    y_val_actual = np.exp(y_val) - LOG_OFFSET
    
    # Calculate metrics on the original scale
    train_metrics = calculate_metrics(y_train_actual, y_train_pred, 'train_')
    val_metrics = calculate_metrics(y_val_actual, y_val_pred, 'val_')
    metrics = {**train_metrics, **val_metrics}
    
    # Gather feature importances
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
    """Load a previously trained model and its scaler."""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    
    feature_names = model_data.get('feature_names', None)
    if feature_names:
        print(f"Model was trained with {len(feature_names)} features: {', '.join(feature_names)}")
    
    return model_data['model'], model_data['x_scaler'], feature_names

def backcast_model(df_desc, n_train, model, feature_names, x_scaler):
    """
    Given a DataFrame sorted in descending order (later years first) and a number of seed rows (n_train)
    with known estimated_deaths, iteratively backcast for rows with missing values.
    
    For XGBoost, the features are scaled using x_scaler, and because the XGB model was trained 
    on log(y+offset), the prediction is converted back via:
        y_pred = exp(predicted_log) - LOG_OFFSET
    """
    df = df_desc.copy()
    for i in range(n_train, len(df)):
        if pd.isna(df.loc[i, 'estimated_deaths']):
            features = {}
            # Build a feature dictionary using the expected feature names
            for col in feature_names:
                if col.startswith("estimated_deaths_lag_"):
                    try:
                        lag = int(col.split('_')[-1])
                    except ValueError:
                        lag = 1
                    if i - lag >= 0:
                        features[col] = df.loc[i - lag, 'estimated_deaths']
                    else:
                        features[col] = np.nan
                else:
                    features[col] = df.loc[i, col] if col in df.columns else 0
            
            # Create a DataFrame row with the proper column order
            X_input = pd.DataFrame([features], columns=feature_names)
            # Fill any missing feature values with 0
            if X_input.isnull().any().any():
                X_input = X_input.fillna(0)
            
            # Transform features with x_scaler
            X_scaled = x_scaler.transform(X_input)
            # Predict in log-space and invert transform
            pred_log = model.predict(X_scaled)[0]
            pred_value = np.exp(pred_log) - LOG_OFFSET
            
            # Update the DataFrame with the predicted value
            df.loc[i, 'estimated_deaths'] = pred_value
    
    return df

def main():
    # Base directory and data path configuration
    base_dir = r"C:\Users\benra\Documents\Newspaper"
    output_dir = os.path.join(base_dir, "regression_finals")
    data_dir = os.path.join(base_dir, "yearly_occurrence_data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    train_path = os.path.join(data_dir, "training_data_1900_1936.csv")
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)
    
    # Load prediction data for backcasting
    pred_path = os.path.join(data_dir, "pred_data_1870_1899.csv")
    pred_df = pd.read_csv(pred_path)
    
    # Add lag features
    train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2], keep_early_years=True)
    
    # Parameters for rotation
    rotation_size = 1
    val_size = 5
    n_rotations = len(train_df_with_lags) // rotation_size
    
    # Get features and target
    X = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    y = train_df_with_lags['estimated_deaths']
    years = train_df_with_lags['year']
    
    print(f"Dataset shape: {X.shape}")
    
    # Check if a model already exists
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"Found existing model at {model_path}")
        model, x_scaler, feature_names = load_model(model_path)
        retrain = input("Do you want to retrain the model? (y/n): ").lower() == 'y'
    else:
        feature_names = None
        retrain = True
    
    if retrain:
        print("Finding best XGBoost parameters...")
        best_xgb_params = find_best_xgboost_params(X, y)
        
        # Perform feature selection
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
        
        # For accumulating feature importances across rotations
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
            
            # Accumulate feature importances for averaging
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
        
        # Save the final model
        if last_model is not None:
            save_model(last_model, last_scalers, model_path, last_feature_names)
        
        # Compute aggregated validation metrics
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
        
        # ---- Evaluate on Full Dataset ----
        print("\nEvaluating on full training dataset...")
        # Reload the training data to ensure a clean state
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
        
        # Scale features using the final x_scaler
        X_all_scaled = last_scalers['x'].transform(X_all)
        # Predict in log-space and convert back
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
        
        # Create a dictionary to map years to predictions for the training period
        train_predictions = {year: pred for year, pred in zip(train_df_with_lags['year'], y_pred_full)}
        
        model = last_model
        x_scaler = last_scalers['x']
        feature_names = last_feature_names
    else:
        print("Using existing model without retraining.")
        model, x_scaler, feature_names = load_model(model_path)
        
        # Evaluate on full dataset to get training period predictions
        train_df = pd.read_csv(train_path)
        train_df = train_df.sort_values("year").reset_index(drop=True)
        train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
        
        # Get features and target
        X_all = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
        y_all = train_df_with_lags['estimated_deaths']
        
        if feature_names:
            X_all = X_all[feature_names]
            print(f"Using {len(feature_names)} selected features")
        
        # Scale features
        X_all_scaled = x_scaler.transform(X_all)
        # Predict in log-space and convert back
        pred_log = model.predict(X_all_scaled)
        y_pred_full = np.exp(pred_log) - LOG_OFFSET
        
        # Create a dictionary to map years to predictions for the training period
        train_predictions = {year: pred for year, pred in zip(train_df_with_lags['year'], y_pred_full)}
    
    # Now, perform backcasting for the earlier years (1870-1899)
    print("\nGenerating backcast predictions for 1870-1899...")
    
    # Ensure the prediction data has an 'estimated_deaths' column (set to NaN if missing)
    if 'estimated_deaths' not in pred_df.columns:
        pred_df['estimated_deaths'] = np.nan
    
    # Combine the two datasets. Training data (1900-1936) acts as seed.
    combined_df = pd.concat([pred_df, train_df], ignore_index=True)
    # In case of duplicate years, keep training data (last occurrence)
    combined_df = combined_df.sort_values('year').drop_duplicates(subset='year', keep='last').reset_index(drop=True)
    
    # For backcasting, sort descending so that later years (with known values) come first.
    combined_desc = combined_df.sort_values('year', ascending=False).reset_index(drop=True)
    
    # Count number of seed rows (years >= 1900)
    n_train = combined_desc[combined_desc['year'] >= 1900].shape[0]
    
    # Perform backcasting
    print("Performing backcasting with XGBoost model...")
    df_backcast = backcast_model(combined_desc, n_train, model, feature_names, x_scaler)
    
    # Sort back to ascending order by year
    final_backcast = df_backcast.sort_values('year').reset_index(drop=True)
    
    # Extract only the backcast period (years up to and including 1900)
    backcast_data = final_backcast[final_backcast['year'] < 1900]
    
    # Create a dictionary to map years to predictions for the backcast period
    backcast_predictions = {year: pred for year, pred in zip(backcast_data['year'], backcast_data['estimated_deaths'])}
    
    # Save backcast predictions to CSV
    output_csv = os.path.join(data_dir, "backcast_predictions_xgb_1870_1899.csv")
    backcast_data.to_csv(output_csv, index=False)
    print(f"Backcast predictions saved to {output_csv}")
    # Updated plotting code for log_xgb.py



    early_year_preds = add_early_year_predictions(
    train_df, 
    model, 
    x_scaler, 
    feature_names, 
    os.path.join(output_dir, 'early_years_check.png')
)

    # Update the train_predictions dictionary with early years
    train_predictions.update(early_year_preds)

    # 6. Updated plotting code
    # Create a combined plot showing both training and backcast predictions
    plt.figure(figsize=(15, 8))

    # Get actual values for all years from the original training data
    actual_values = train_df.set_index('year')['estimated_deaths']

    # Create years list for actual data and predictions
    all_actual_years = sorted([year for year in actual_values.index])
    train_pred_years = sorted([year for year in train_predictions.keys()])

    # Plot actual data (1900-1936)
    plt.plot(all_actual_years, [actual_values[year] for year in all_actual_years], 
             'ko-', linewidth=2, label='Actual (1900-1936)')

    # Plot training predictions (now 1900-1936, including early years)
    plt.plot(train_pred_years, [train_predictions[year] for year in train_pred_years], 
             'b^--', alpha=0.7, label='XGBoost Predictions (1900-1936)')

    # Plot backcast predictions (1870-1899)
    years_backcast = sorted(backcast_predictions.keys())
    predictions_backcast = [backcast_predictions[year] for year in years_backcast]
    plt.plot(years_backcast, predictions_backcast, 
             'r*--', alpha=0.7, label='XGBoost Backcast (1870-1899)')

    # Add vertical line to separate backcast from actual data
    plt.axvline(x=1899.5, color='gray', linestyle=':', alpha=0.7)

    plt.title('XGBoost: Training Predictions and Backcast (1870-1936)')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    output_plot = os.path.join(output_dir, 'xgboost_combined_predictions.png')
    plt.savefig(output_plot)
    plt.show()
    print(f"Combined predictions plot saved to {output_plot}")
if __name__ == "__main__":
    main()