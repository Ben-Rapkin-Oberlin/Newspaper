import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    median_absolute_error, max_error, explained_variance_score
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from collections import defaultdict

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
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def find_best_lasso_params(X_train, y_train):
    """Perform grid search for Lasso with improved convergence using 5-fold cross-validation"""
    param_grid = {
        'alpha': [2000],
        'max_iter': [20000],
        'tol': [1e-2],
        'selection': ['random'],
        'warm_start': [True]
    }
    
    lasso = Lasso(random_state=42)
    grid_search = GridSearchCV(
        lasso, 
        param_grid, 
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Lasso parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_, grid_search.best_estimator_

def select_top_features(X, model, n_features=None, threshold=None):
    """
    Select top features based on absolute coefficient values from Lasso model.
    Either selects the top n_features or all features with abs coefficient > threshold.
    """
    # Get feature names
    feature_names = X.columns
    
    # Get coefficients from model
    coef = model.coef_
    
    # Create dataframe of features and their coefficients
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    })
    
    # Sort by absolute coefficient value
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Coefficient']:.6f}")
    
    # Select features
    if n_features is not None:
        selected_features = feature_importance.head(n_features)['Feature'].tolist()
        print(f"\nSelected top {n_features} features:")
    elif threshold is not None:
        selected_features = feature_importance[
            feature_importance['Abs_Coefficient'] > threshold
        ]['Feature'].tolist()
        print(f"\nSelected features with abs coefficient > {threshold}:")
    else:
        # By default, select features with non-zero coefficients
        selected_features = feature_importance[
            feature_importance['Abs_Coefficient'] > 0
        ]['Feature'].tolist()
        print("\nSelected features with non-zero coefficients:")
    
    print(", ".join(selected_features))
    print(f"Number of selected features: {len(selected_features)}")
    
    return selected_features

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

def plot_lasso_feature_importance(feature_coefficients, feature_names=None, title="Lasso Feature Coefficients"):
    """Plot feature coefficients from Lasso model"""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(feature_coefficients))]
    
    # Create DataFrame with feature names and coefficients
    data = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': feature_coefficients,
        'AbsCoefficient': np.abs(feature_coefficients)
    })
    
    # Sort by absolute coefficient value
    data = data.sort_values('AbsCoefficient', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = ['red' if c < 0 else 'blue' for c in data['Coefficient']]
    plt.barh(range(len(data)), data['Coefficient'], align='center', color=colors)
    plt.yticks(range(len(data)), data['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title(title)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    
    return plt

def evaluate_lasso_rotation(X, y, years, val_size, lasso_params, selected_features=None):
    """Evaluate Lasso model on a specific rotation of the data without normalization"""
    
    # Use all features or only selected features
    if selected_features is not None:
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features for evaluation")
    
    # Split into training and validation without scaling
    X_train = X[:-val_size]
    y_train = y[:-val_size]
    X_val = X[-val_size:]
    y_val = y[-val_size:]
    val_years = years[-val_size:].values
    
    # Create and train Lasso model
    lasso_model = Lasso(**lasso_params, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Get predictions without normalization
    predictions = {
        'train': lasso_model.predict(X_train),
        'val': lasso_model.predict(X_val)
    }
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, predictions['train'], 'train_')
    val_metrics = calculate_metrics(y_val, predictions['val'], 'val_')
    metrics = {**train_metrics, **val_metrics}
    
    return {
        'years': {'val': val_years},
        'predictions': predictions,
        'actual': {'train': y_train, 'val': y_val},
        'metrics': metrics,
        'model': lasso_model,
        'feature_names': X.columns.tolist()
    }

def save_model(model, file_path="lasso_model.pkl", feature_names=None):
    """Save the trained model for later use"""
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {file_path}")

def load_model(file_path="lasso_model.pkl"):
    """Load a previously trained model"""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    
    # Get feature names if they exist
    feature_names = model_data.get('feature_names', None)
    if feature_names:
        print(f"Model was trained with {len(feature_names)} features: {', '.join(feature_names)}")
    
    return model_data['model'], feature_names

def add_early_year_predictions(train_df, model, selected_features=None):
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
    
    # LASSO doesn't need scaling - predict directly
    early_predictions = model.predict(X_early)
    
    # Create a dictionary mapping years to predictions
    early_years_dict = {year: pred for year, pred in zip(early_df['year'], early_predictions)}
    
    print(f"Added predictions for early years: {early_years_dict}")
    return early_years_dict

def backcast_model(df_desc, n_train, model, feature_names, selected_features=None):
    """
    Given a DataFrame sorted in descending order (later years first) and a number of seed rows (n_train)
    with known estimated_deaths, iteratively backcast for rows with missing values.
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
                        features[col] = 0  # Use 0 for missing lag values
                else:
                    features[col] = df.loc[i, col] if col in df.columns else 0
            
            # Create a DataFrame row with the proper column order
            X_input = pd.DataFrame([features], columns=feature_names)
            # Fill any missing feature values with 0
            if X_input.isnull().any().any():
                X_input = X_input.fillna(0)
            
            # For LASSO, predict directly without scaling
            if selected_features is not None:
                X_input = X_input[selected_features]
            
            pred_value = model.predict(X_input)[0]
            
            # Update the DataFrame with the predicted value
            df.loc[i, 'estimated_deaths'] = pred_value
    
    return df

def main():
    # Base directory and data path configuration
    base_dir = "C:\\Users\\benra\\Documents\\Newspaper"
    output_dir = os.path.join(base_dir, "regression_finals")
    data_dir = os.path.join(base_dir, "yearly_occurrence_data")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(data_dir, "training_data_1900_1936.csv")
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)

    # Add lag features - now using keep_early_years=True to include 1900-1901
    train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2], keep_early_years=True)
    
    # Parameters
    rotation_size = 1  # Number of years to rotate by
    val_size = 5       # Validation window size
    n_rotations = len(train_df_with_lags) // rotation_size  # Number of rotations to perform

    # Get features
    X = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    print(f"Dataset shape: {X.shape}")
    y = train_df_with_lags['estimated_deaths']
    years = train_df_with_lags['year']
    
    # Check if model already exists
    model_path = os.path.join(output_dir, "lasso_model.pkl")
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"Found existing model at {model_path}")
        model, feature_names = load_model(model_path)
        retrain = input("Do you want to retrain the model? (y/n): ").lower() == 'y'
    else:
        feature_names = None
        retrain = True
    
    if retrain:
        # Get best parameters for Lasso model and the best model
        print("Finding best Lasso parameters...")
        best_lasso_params, best_lasso_model = find_best_lasso_params(X, y)
        
        # Perform feature selection
        feature_selection_enabled = True  # Set to False to disable feature selection
        if feature_selection_enabled:
            # Get top N features or features with coefficients above threshold
            n_features = int(input("Enter number of top features to select (0 to use threshold instead): "))
            if n_features > 0:
                selected_features = select_top_features(X, best_lasso_model, n_features=n_features)
            else:
                threshold = float(input("Enter coefficient threshold for feature selection: "))
                selected_features = select_top_features(X, best_lasso_model, threshold=threshold)
        else:
            selected_features = None  # Use all features
    
        # Initialize storage
        results = {}
        validation_predictions = defaultdict(list)
        validation_actuals = defaultdict(list)
        
        # Variable to store the last trained model
        last_model = None
        last_feature_names = None
        
        # Evaluate all rotations
        for i in range(n_rotations):
            if i > 0:
                X = rotate_series(X, rotation_size)
                y = rotate_series(pd.DataFrame(y), rotation_size).iloc[:, 0]
                years = rotate_series(pd.DataFrame(years), rotation_size).iloc[:, 0]
            
            key = f"Rotation {i+1}"
            print(f"\nEvaluating {key}...")
            results[key] = evaluate_lasso_rotation(
                X, y, years,
                val_size=val_size,
                lasso_params=best_lasso_params,
                selected_features=selected_features
            )
            
            # Store the model from the last rotation
            if i == n_rotations - 1:
                last_model = results[key]['model']
                last_feature_names = results[key].get('feature_names', None)
            
            # Store validation predictions and actuals
            val_years = results[key]['years']['val']
            val_actuals = results[key]['actual']['val']
            val_preds = results[key]['predictions']['val']
            
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append(pred)
                validation_actuals[year].append(actual)

        # Save the final model
        if last_model is not None:
            save_model(last_model, model_path, last_feature_names)
            
        # Plot feature importance from the final model
        if last_model is not None and last_feature_names is not None:
            coef_plot = plot_lasso_feature_importance(last_model.coef_, last_feature_names)
            coef_plot.savefig(os.path.join(output_dir, 'lasso_feature_coefficients.png'))
            plt.close()

        # Compute aggregated metrics
        aggregated_metrics = {}
        for year in validation_predictions.keys():
            predictions = validation_predictions[year]
            actuals = validation_actuals[year]
            
            aggregated_metrics[year] = {
                'actual': np.mean(actuals),
                'lasso_pred': np.mean(predictions)
            }

        # Convert to DataFrame
        agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
        agg_df.index.name = 'year'
        agg_df = agg_df.sort_index()

        # Print comprehensive metrics
        print("\nLASSO Metrics:")
        metrics = calculate_metrics(agg_df['actual'], agg_df['lasso_pred'])
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.3f}")
        print(f"Median AE: {metrics['median_ae']:.2f}")
        print(f"Max Error: {metrics['max_error']:.2f}")
        print(f"Explained Variance: {metrics['explained_variance']:.3f}")
        if not np.isnan(metrics['mape']):
            print(f"MAPE: {metrics['mape']:.2f}%")

        # Create a dictionary of train_predictions for all years in training data
        X_all = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
        if selected_features:
            X_all = X_all[selected_features]
        
        # Get predictions for all years
        all_preds = last_model.predict(X_all)
        train_predictions = {year: pred for year, pred in zip(train_df_with_lags['year'], all_preds)}
        
        # Add predictions for early years (1900-1901) if they're missing
        if 1900 not in train_predictions or 1901 not in train_predictions:
            early_year_preds = add_early_year_predictions(train_df, last_model, selected_features)
            train_predictions.update(early_year_preds)
        
        # Optional: Generate backcast predictions (1870-1899)
        # Load prediction data for backcasting
        pred_path = os.path.join(data_dir, "pred_data_1870_1899.csv")
        if os.path.exists(pred_path):
            try:
                pred_df = pd.read_csv(pred_path)
                
                # Ensure the prediction data has an 'estimated_deaths' column (set to NaN if missing)
                if 'estimated_deaths' not in pred_df.columns:
                    pred_df['estimated_deaths'] = np.nan
                
                # Combine the two datasets. Training data (1900-1936) acts as seed.
                combined_df = pd.concat([pred_df, train_df], ignore_index=True)
                # In case of duplicate years, keep training data (last occurrence)
                combined_df = combined_df.sort_values('year').drop_duplicates(subset='year', keep='last').reset_index(drop=True)
                
                # For backcasting, sort descending so later years (with known values) come first.
                combined_desc = combined_df.sort_values('year', ascending=False).reset_index(drop=True)
                
                # Count number of seed rows (years >= 1900)
                n_train = combined_desc[combined_desc['year'] >= 1900].shape[0]
                
                # Perform backcasting
                print("\nPerforming backcasting with LASSO model...")
                df_backcast = backcast_model(combined_desc, n_train, last_model, 
                                             feature_names=last_feature_names, 
                                             selected_features=selected_features)
                
                # Sort back to ascending order by year
                final_backcast = df_backcast.sort_values('year').reset_index(drop=True)
                
                # Extract only the backcast period (years before 1900)
                backcast_data = final_backcast[final_backcast['year'] < 1900]
                
                # Create a dictionary to map years to predictions for the backcast period
                backcast_predictions = {year: pred for year, pred in zip(backcast_data['year'], backcast_data['estimated_deaths'])}
                
                # Save backcast predictions to CSV
                output_csv = os.path.join(data_dir, "backcast_predictions_lasso_1870_1899.csv")
                backcast_data.to_csv(output_csv, index=False)
                print(f"Backcast predictions saved to {output_csv}")
            except Exception as e:
                print(f"Error during backcasting: {e}")
                backcast_predictions = {}
        else:
            print(f"Backcast data file not found at {pred_path}")
            backcast_predictions = {}
        
        # Enhanced visualization with all years
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
                'g^--', alpha=0.7, label='LASSO Predictions (1900-1936)')
        
        # Plot backcast predictions (1870-1899)
        if backcast_predictions:
            years_backcast = sorted(backcast_predictions.keys())
            predictions_backcast = [backcast_predictions[year] for year in years_backcast]
            plt.plot(years_backcast, predictions_backcast, 
                    'r*--', alpha=0.7, label='LASSO Backcast (1870-1899)')
        
        # Add vertical line to separate backcast from actual data
        plt.axvline(x=1899.5, color='gray', linestyle=':', alpha=0.7)
        
        plt.title('LASSO: Training Predictions and Backcast (1870-1936)')
        plt.xlabel('Year')
        plt.ylabel('Estimated Deaths')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the combined plot
        output_plot = os.path.join(output_dir, 'lasso_combined_predictions.png')
        plt.savefig(output_plot)
        plt.show()
        print(f"Combined predictions plot saved to {output_plot}")

        # Create error analysis plot
        plt.figure(figsize=(15, 6))
        errors = np.abs(agg_df['lasso_pred'] - agg_df['actual'])
        plt.plot(agg_df.index, errors, 'o-', label='LASSO Error', color='green', alpha=0.7)
        
        plt.title('Lasso Regression: Absolute Prediction Errors Over Time')
        plt.xlabel('Year')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_errors.png'))
        plt.show()
    else:
        print("Using existing model without retraining.")
        
        # Add a section to run evaluation on existing model
        run_evaluation = input("Do you want to run evaluation on the existing model? (y/n): ").lower() == 'y'
        if run_evaluation:
            # Load the data
            train_df = pd.read_csv(os.path.join(data_dir, "training_data_1900_1936.csv"))
            train_df = train_df.sort_values("year").reset_index(drop=True)
            train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2], keep_early_years=True)
            
            # Get features and target
            X_all = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
            y = train_df_with_lags['estimated_deaths']
            
            # Use only the features the model was trained on
            if feature_names:
                X = X_all[feature_names]
                print(f"Using {len(feature_names)} selected features for evaluation: {', '.join(feature_names)}")
            else:
                X = X_all
                print("Using all features for evaluation")
            
            # Make predictions directly without scaling
            y_pred = model.predict(X)
            
            # Calculate metrics
            metrics = calculate_metrics(y, y_pred)
            
            print("\nLASSO Metrics on Full Dataset:")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"R²: {metrics['r2']:.3f}")
            print(f"Median AE: {metrics['median_ae']:.2f}")
            print(f"Max Error: {metrics['max_error']:.2f}")
            print(f"Explained Variance: {metrics['explained_variance']:.3f}")
            if not np.isnan(metrics['mape']):
                print(f"MAPE: {metrics['mape']:.2f}%")
                
            # Plot feature importance
            if feature_names:
                coef_plot = plot_lasso_feature_importance(model.coef_, feature_names)
                coef_plot.savefig(os.path.join(output_dir, 'lasso_feature_coefficients.png'))
                plt.show()
            
            # Create train_predictions dictionary
            train_predictions = {year: pred for year, pred in zip(train_df_with_lags['year'], y_pred)}
            
            # Add predictions for early years (1900-1901) if they're missing
            if 1900 not in train_predictions or 1901 not in train_predictions:
                early_year_preds = add_early_year_predictions(train_df, model, feature_names)
                train_predictions.update(early_year_preds)
            
            # Plot results
            plt.figure(figsize=(15, 8))
            
            # Get actual values for all years
            actual_values = train_df.set_index('year')['estimated_deaths']
            
            # Plot actual and predicted values
            all_actual_years = sorted([year for year in actual_values.index])
            train_pred_years = sorted([year for year in train_predictions.keys()])
            
            plt.plot(all_actual_years, [actual_values[year] for year in all_actual_years], 
                    'ko-', linewidth=2, label='Actual (1900-1936)')
            plt.plot(train_pred_years, [train_predictions[year] for year in train_pred_years], 
                    'g^--', alpha=0.7, label='LASSO Predictions (1900-1936)')
            
            plt.title('LASSO: Predictions vs Actual (1900-1936)')
            plt.xlabel('Year')
            plt.ylabel('Estimated Deaths')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            output_plot = os.path.join(output_dir, 'lasso_predictions_evaluation.png')
            plt.savefig(output_plot)
            plt.show()
            print(f"Evaluation plot saved to {output_plot}")

if __name__ == "__main__":
    main()