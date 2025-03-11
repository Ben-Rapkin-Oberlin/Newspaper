import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    median_absolute_error, max_error, explained_variance_score
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from collections import defaultdict

def create_lag_features(df, target_col, lag_periods=[1, 2]):
    """Create lag features for time series prediction"""
    df = df.copy()
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df.dropna()

def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def find_best_lasso_params(X_train, y_train):
    """Perform grid search for Lasso with improved convergence using 5-fold cross-validation"""
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'max_iter': [20000],  # Significantly increased for better convergence
        'tol': [1e-2],  # Increased tolerance for faster convergence
        'selection': ['random'],
        'warm_start': [True]
    }
    
    lasso = Lasso(random_state=42)
    grid_search = GridSearchCV(
        lasso, 
        param_grid, 
        cv=5,  # 5-fold cross-validation for parameter selection
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0  # Reducing verbosity
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Lasso parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_, grid_search.best_estimator_

def select_top_features(X, model, n_features=None, threshold=None):
    """
    Select top features based on absolute coefficient values from Lasso model.
    Either selects the top n_features or all features with abs coefficient > threshold.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataframe
    model : Lasso
        Trained Lasso model
    n_features : int, optional
        Number of top features to select
    threshold : float, optional
        Threshold for coefficient values
        
    Returns:
    --------
    list of str
        Names of selected features
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
    """Evaluate Lasso model on a specific rotation of the data with enhanced metrics and normalization"""
    
    # Use all features or only selected features
    if selected_features is not None:
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features for evaluation")
    
    # Scale features
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = x_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Scale target variable
    y_values = y.values if hasattr(y, 'values') else y
    y_scaled = y_scaler.fit_transform(y_values.reshape(-1, 1)).ravel()
    
    # Split into training and validation
    X_train = X_scaled[:-val_size]
    y_train = y_scaled[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y_scaled[-val_size:]
    val_years = years[-val_size:].values
    
    # Create and train Lasso model
    lasso_model = Lasso(**lasso_params, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Get predictions and inverse transform
    predictions = {
        'train': y_scaler.inverse_transform(lasso_model.predict(X_train).reshape(-1, 1)).ravel(),
        'val': y_scaler.inverse_transform(lasso_model.predict(X_val).reshape(-1, 1)).ravel()
    }
    
    # Inverse transform the actual values for metric calculation
    y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_val_original = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    
    # Calculate metrics using original scale values
    train_metrics = calculate_metrics(y_train_original, predictions['train'], 'train_')
    val_metrics = calculate_metrics(y_val_original, predictions['val'], 'val_')
    metrics = {**train_metrics, **val_metrics}
    
    return {
        'years': {'val': val_years},
        'predictions': predictions,
        'actual': {'train': y_train_original, 'val': y_val_original},
        'metrics': metrics,
        'model': lasso_model,
        'scalers': {'x': x_scaler, 'y': y_scaler},  # Save scalers for future use
        'feature_names': X.columns.tolist()  # Store feature names
    }
 
def save_model(model, scalers, file_path="lasso_model.pkl", feature_names=None):
    """Save the trained model and scalers for later use"""
    model_data = {
        'model': model,
        'x_scaler': scalers['x'],
        'y_scaler': scalers['y'],
        'feature_names': feature_names  # Store selected feature names
    }
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {file_path}")

def load_model(file_path="lasso_model.pkl"):
    """Load a previously trained model and its scalers"""
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model loaded from {file_path}")
    
    # Get feature names if they exist
    feature_names = model_data.get('feature_names', None)
    if feature_names:
        print(f"Model was trained with {len(feature_names)} features: {', '.join(feature_names)}")
    
    return model_data['model'], model_data['x_scaler'], model_data['y_scaler'], feature_names

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

    # Add lag features
    train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
    
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
        model, x_scaler, y_scaler, feature_names = load_model(model_path)
        # You could use the loaded model here if needed
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
        last_scalers = None
        
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
                last_scalers = results[key]['scalers']
                last_feature_names = results[key].get('feature_names', None)
            
            # Store validation predictions and actuals
            val_years = results[key]['years']['val']
            val_actuals = results[key]['actual']['val']
            val_preds = results[key]['predictions']['val']
            
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append(pred)
                validation_actuals[year].append(actual)

        # Save the final model
        if last_model is not None and last_scalers is not None:
            save_model(last_model, last_scalers, model_path, last_feature_names)
            
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

        # Create visualization
        plt.figure(figsize=(15, 8))
        plt.plot(agg_df.index, agg_df['actual'], 'ko-', label='Actual', linewidth=2)
        plt.plot(agg_df.index, agg_df['lasso_pred'], 's--', label='LASSO', color='green', alpha=0.7)
        
        plt.title('Lasso Regression: Aggregated Validation Predictions vs Actual Values')
        plt.xlabel('Year')
        plt.ylabel('Estimated Deaths')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_predictions.png'))
        plt.show()

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
        
        # Add a section to run evaluation on existing model if desired
        run_evaluation = input("Do you want to run evaluation on the existing model? (y/n): ").lower() == 'y'
        if run_evaluation:
            # Load the data
            train_df = pd.read_csv(os.path.join(data_dir, "training_data_1900_1936.csv"))
            train_df = train_df.sort_values("year").reset_index(drop=True)
            train_df_with_lags = create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
            
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
            
            # Scale features and target
            X_scaled = x_scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            
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
        # Here you could add code to use the loaded model for predictions
        # without retraining

if __name__ == "__main__":
    main()