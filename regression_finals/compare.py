import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import importlib.util
import time

def load_module(file_path, module_name):
    """Load a Python file as a module"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def evaluate_and_compare_models(base_dir, output_dir, data_dir, lasso_file, xgb_file, model_names=['LASSO', 'XGBoost']):
    """Evaluate and compare the LASSO and XGBoost models"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the modules from file paths
    lasso_module = load_module(os.path.join(base_dir, lasso_file), "lasso_model")
    xgb_module = load_module(os.path.join(base_dir, xgb_file), "xgb_model")
    
    # Load data
    train_path = os.path.join(data_dir, "training_data_1900_1936.csv")
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)

    # Add lag features
    train_df_with_lags = lasso_module.create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
    
    # Parameters
    rotation_size = 1  # Number of years to rotate by
    val_size = 5       # Validation window size
    n_rotations = len(train_df_with_lags) // rotation_size  # Number of rotations to perform

    # Get features
    X = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    print(f"Dataset shape: {X.shape}")
    y = train_df_with_lags['estimated_deaths']
    years = train_df_with_lags['year']
    
    # Get best parameters for both models
    print("Finding best LASSO parameters...")
    best_lasso_params, best_lasso_model = lasso_module.find_best_lasso_params(X, y)
    
    print("\nFinding best XGBoost parameters...")
    best_xgb_params = xgb_module.find_best_xgboost_params(X, y)
    
    # Feature selection
    n_features = int(input("Enter number of top features to select (recommend 10-20): "))
    
    # LASSO feature selection
    print("\nPerforming LASSO feature selection...")
    lasso_selected_features = lasso_module.select_top_features(X, best_lasso_model, n_features=n_features)
    
    # Train an initial XGBoost model to get feature importances
    print("\nPerforming XGBoost feature selection...")
    import xgboost as xgb
    initial_xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    initial_xgb_model.fit(X, y)
    xgb_selected_features = xgb_module.select_top_features(X, initial_xgb_model, n_features=n_features)
    
    # Compare selected features
    common_features = set(lasso_selected_features).intersection(set(xgb_selected_features))
    
    print(f"\nFeature selection comparison:")
    print(f"Number of common features: {len(common_features)}")
    print(f"Common features: {', '.join(common_features)}")
    
    # Initialize storage for both models
    lasso_results = {}
    xgb_results = {}
    validation_predictions = {
        'lasso': defaultdict(list),
        'xgb': defaultdict(list)
    }
    validation_actuals = defaultdict(list)
    
    # For storing models
    last_models = {'lasso': None, 'xgb': None}
    last_scalers = {'xgb': None}
    last_feature_names = {'lasso': None, 'xgb': None}
    
    # Create copies for rotation
    X_lasso = X.copy()
    X_xgb = X.copy()
    y_lasso = y.copy()
    y_xgb = y.copy()
    years_lasso = years.copy()
    years_xgb = years.copy()
    
    # Track performance metrics for each rotation
    metrics_by_rotation = []
    
    # Evaluate all rotations
    for i in range(n_rotations):
        if i > 0:
            X_lasso = lasso_module.rotate_series(X_lasso, rotation_size)
            X_xgb = xgb_module.rotate_series(X_xgb, rotation_size)
            y_lasso = lasso_module.rotate_series(pd.DataFrame(y_lasso), rotation_size).iloc[:, 0]
            y_xgb = xgb_module.rotate_series(pd.DataFrame(y_xgb), rotation_size).iloc[:, 0]
            years_lasso = lasso_module.rotate_series(pd.DataFrame(years_lasso), rotation_size).iloc[:, 0]
            years_xgb = xgb_module.rotate_series(pd.DataFrame(years_xgb), rotation_size).iloc[:, 0]
        
        rotation_key = f"Rotation {i+1}"
        print(f"\nEvaluating {rotation_key}...")
        
        # Evaluate LASSO
        start_time = time.time()
        lasso_results[rotation_key] = lasso_module.evaluate_lasso_rotation(
            X_lasso, y_lasso, years_lasso,
            val_size=val_size,
            lasso_params=best_lasso_params,
            selected_features=lasso_selected_features
        )
        lasso_time = time.time() - start_time
        
        # Evaluate XGBoost
        start_time = time.time()
        xgb_results[rotation_key] = xgb_module.evaluate_xgboost_rotation(
            X_xgb, y_xgb, years_xgb,
            val_size=val_size,
            xgb_params=best_xgb_params,
            selected_features=xgb_selected_features
        )
        xgb_time = time.time() - start_time
        
        # Store the models from the last rotation
        if i == n_rotations - 1:
            last_models['lasso'] = lasso_results[rotation_key]['model']
            last_models['xgb'] = xgb_results[rotation_key]['model']
            last_scalers['xgb'] = xgb_results[rotation_key]['scalers']
            last_feature_names['lasso'] = lasso_results[rotation_key]['feature_names']
            last_feature_names['xgb'] = xgb_results[rotation_key]['feature_names']
        
        # Store validation predictions and actuals
        lasso_val_years = lasso_results[rotation_key]['years']['val']
        xgb_val_years = xgb_results[rotation_key]['years']['val']
        
        lasso_val_actuals = lasso_results[rotation_key]['actual']['val']
        xgb_val_actuals = xgb_results[rotation_key]['actual']['val']
        
        lasso_val_preds = lasso_results[rotation_key]['predictions']['val']
        xgb_val_preds = xgb_results[rotation_key]['predictions']['val']
        
        for year, pred, actual in zip(lasso_val_years, lasso_val_preds, lasso_val_actuals):
            validation_predictions['lasso'][year].append(pred)
            validation_actuals[year].append(actual)
            
        for year, pred, actual in zip(xgb_val_years, xgb_val_preds, xgb_val_actuals):
            validation_predictions['xgb'][year].append(pred)
            # We only need to store actuals once (they're the same for both models)
            
        # Compare metrics for this rotation
        lasso_metrics = lasso_results[rotation_key]['metrics']
        xgb_metrics = xgb_results[rotation_key]['metrics']
        
        metrics_by_rotation.append({
            'rotation': i+1,
            'lasso_train_rmse': np.sqrt(lasso_metrics['train_mse']),
            'lasso_val_rmse': np.sqrt(lasso_metrics['val_mse']),
            'lasso_train_r2': lasso_metrics['train_r2'],
            'lasso_val_r2': lasso_metrics['val_r2'],
            'lasso_time': lasso_time,
            'xgb_train_rmse': np.sqrt(xgb_metrics['train_mse']),
            'xgb_val_rmse': np.sqrt(xgb_metrics['val_mse']),
            'xgb_train_r2': xgb_metrics['train_r2'],
            'xgb_val_r2': xgb_metrics['val_r2'],
            'xgb_time': xgb_time
        })
        
        print(f"LASSO - Train RMSE: {np.sqrt(lasso_metrics['train_mse']):.2f}, Val RMSE: {np.sqrt(lasso_metrics['val_mse']):.2f}, Time: {lasso_time:.2f}s")
        print(f"XGBoost - Train RMSE: {np.sqrt(xgb_metrics['train_mse']):.2f}, Val RMSE: {np.sqrt(xgb_metrics['val_mse']):.2f}, Time: {xgb_time:.2f}s")

    # Save both models
    lasso_module.save_model(last_models['lasso'], os.path.join(output_dir, "lasso_model.pkl"), last_feature_names['lasso'])
    xgb_module.save_model(last_models['xgb'], last_scalers['xgb'], os.path.join(output_dir, "xgboost_model.pkl"), last_feature_names['xgb'])

    # Compute aggregated metrics
    aggregated_metrics = {}
    for year in validation_actuals.keys():
        actuals = validation_actuals[year]
        lasso_preds = validation_predictions['lasso'][year]
        xgb_preds = validation_predictions['xgb'][year]
        
        aggregated_metrics[year] = {
            'actual': np.mean(actuals),
            'lasso_pred': np.mean(lasso_preds),
            'xgb_pred': np.mean(xgb_preds)
        }

    # Convert to DataFrame for analysis
    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    agg_df.index.name = 'year'
    agg_df = agg_df.sort_index()
    
    # Calculate overall metrics
    lasso_metrics = lasso_module.calculate_metrics(agg_df['actual'], agg_df['lasso_pred'])
    xgb_metrics = xgb_module.calculate_metrics(agg_df['actual'], agg_df['xgb_pred'])

    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON - VALIDATION METRICS")
    print("="*50)
    metrics_to_compare = ['mae', 'rmse', 'r2', 'median_ae', 'max_error', 'explained_variance']
    model_names = ['LASSO', 'XGBoost']
    
    # Create a comparison table
    comparison_table = pd.DataFrame({
        'Metric': metrics_to_compare,
        model_names[0]: [lasso_metrics[m] for m in metrics_to_compare],
        model_names[1]: [xgb_metrics[m] for m in metrics_to_compare],
        'Difference': [(xgb_metrics[m] - lasso_metrics[m]) for m in metrics_to_compare],
        'Better Model': [model_names[1] if ((m == 'r2' or m == 'explained_variance') and xgb_metrics[m] > lasso_metrics[m]) or 
                         ((m != 'r2' and m != 'explained_variance') and xgb_metrics[m] < lasso_metrics[m]) 
                         else model_names[0] for m in metrics_to_compare]
    })
    
    print(comparison_table.to_string(index=False))
    
    # Save the comparison table
    comparison_table.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create a rotation metrics DataFrame
    rotation_metrics_df = pd.DataFrame(metrics_by_rotation)
    rotation_metrics_df.to_csv(os.path.join(output_dir, 'rotation_metrics.csv'), index=False)
    
    # Calculate average metrics across all rotations
    avg_rotation_metrics = {
        'lasso_train_rmse': rotation_metrics_df['lasso_train_rmse'].mean(),
        'lasso_val_rmse': rotation_metrics_df['lasso_val_rmse'].mean(),
        'lasso_train_r2': rotation_metrics_df['lasso_train_r2'].mean(),
        'lasso_val_r2': rotation_metrics_df['lasso_val_r2'].mean(),
        'lasso_time': rotation_metrics_df['lasso_time'].mean(),
        'xgb_train_rmse': rotation_metrics_df['xgb_train_rmse'].mean(),
        'xgb_val_rmse': rotation_metrics_df['xgb_val_rmse'].mean(),
        'xgb_train_r2': rotation_metrics_df['xgb_train_r2'].mean(),
        'xgb_val_r2': rotation_metrics_df['xgb_val_r2'].mean(),
        'xgb_time': rotation_metrics_df['xgb_time'].mean()
    }
    
    print("\nAVERAGE ROTATION METRICS:")
    print(f"LASSO - Train RMSE: {avg_rotation_metrics['lasso_train_rmse']:.2f}, Val RMSE: {avg_rotation_metrics['lasso_val_rmse']:.2f}, R²: {avg_rotation_metrics['lasso_val_r2']:.3f}, Time: {avg_rotation_metrics['lasso_time']:.2f}s")
    print(f"XGBoost - Train RMSE: {avg_rotation_metrics['xgb_train_rmse']:.2f}, Val RMSE: {avg_rotation_metrics['xgb_val_rmse']:.2f}, R²: {avg_rotation_metrics['xgb_val_r2']:.3f}, Time: {avg_rotation_metrics['xgb_time']:.2f}s")

    # Create prediction visualization
    plt.figure(figsize=(15, 8))
    plt.plot(agg_df.index, agg_df['actual'], 'ko-', label='Actual', linewidth=2)
    plt.plot(agg_df.index, agg_df['lasso_pred'], 's--', label=model_names[0], color='green', alpha=0.7)
    plt.plot(agg_df.index, agg_df['xgb_pred'], '^--', label=model_names[1], color='red', alpha=0.7)
    
    plt.title('Model Comparison: Aggregated Validation Predictions vs Actual Values')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_predictions.png'))
    
    # Create error analysis plot
    plt.figure(figsize=(15, 6))
    lasso_errors = np.abs(agg_df['lasso_pred'] - agg_df['actual'])
    xgb_errors = np.abs(agg_df['xgb_pred'] - agg_df['actual'])
    
    plt.plot(agg_df.index, lasso_errors, 'o-', label=f'{model_names[0]} Error', color='green', alpha=0.7)
    plt.plot(agg_df.index, xgb_errors, 's-', label=f'{model_names[1]} Error', color='red', alpha=0.7)
    
    plt.title('Model Comparison: Absolute Prediction Errors Over Time')
    plt.xlabel('Year')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_errors.png'))
    
    # Plot rotation metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rotation_metrics_df['rotation'], rotation_metrics_df['lasso_val_rmse'], 'o-', label=f'{model_names[0]} Val RMSE', color='green')
    plt.plot(rotation_metrics_df['rotation'], rotation_metrics_df['xgb_val_rmse'], 's-', label=f'{model_names[1]} Val RMSE', color='red')
    plt.title('Validation RMSE by Rotation')
    plt.xlabel('Rotation')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rotation_metrics_df['rotation'], rotation_metrics_df['lasso_val_r2'], 'o-', label=f'{model_names[0]} Val R²', color='green')
    plt.plot(rotation_metrics_df['rotation'], rotation_metrics_df['xgb_val_r2'], 's-', label=f'{model_names[1]} Val R²', color='red')
    plt.title('Validation R² by Rotation')
    plt.xlabel('Rotation')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rotation_metrics_comparison.png'))
    
    # Evaluate on full dataset
    print("\n" + "="*50)
    print("FULL DATASET EVALUATION")
    print("="*50)
    
    # Load the data again to ensure clean state
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)
    train_df_with_lags = lasso_module.create_lag_features(train_df, 'estimated_deaths', lag_periods=[1, 2])
    
    # Get features and target
    X_all = train_df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    y_all = train_df_with_lags['estimated_deaths']
    
    # Make LASSO predictions with selected features
    X_lasso_selected = X_all[lasso_selected_features]
    lasso_pred = last_models['lasso'].predict(X_lasso_selected)
    
    # Make XGBoost predictions (with scaling) with selected features
    X_xgb_selected = X_all[xgb_selected_features]
    X_scaled = last_scalers['xgb']['x'].transform(X_xgb_selected)
    xgb_pred_scaled = last_models['xgb'].predict(X_scaled)
    xgb_pred = last_scalers['xgb']['y'].inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    full_lasso_metrics = lasso_module.calculate_metrics(y_all, lasso_pred)
    full_xgb_metrics = xgb_module.calculate_metrics(y_all, xgb_pred)
    
    # Create comparison table for full dataset
    full_comparison_table = pd.DataFrame({
        'Metric': metrics_to_compare,
        model_names[0]: [full_lasso_metrics[m] for m in metrics_to_compare],
        model_names[1]: [full_xgb_metrics[m] for m in metrics_to_compare],
        'Difference': [(full_xgb_metrics[m] - full_lasso_metrics[m]) for m in metrics_to_compare],
        'Better Model': [model_names[1] if ((m == 'r2' or m == 'explained_variance') and full_xgb_metrics[m] > full_lasso_metrics[m]) or 
                         ((m != 'r2' and m != 'explained_variance') and full_xgb_metrics[m] < full_lasso_metrics[m]) 
                         else model_names[0] for m in metrics_to_compare]
    })
    
    print(full_comparison_table.to_string(index=False))
    
    # Save the full comparison table
    full_comparison_table.to_csv(os.path.join(output_dir, 'full_model_comparison.csv'), index=False)
    
    # Create full dataset prediction visualization
    plt.figure(figsize=(15, 8))
    plt.plot(train_df_with_lags['year'], y_all, 'ko-', label='Actual', linewidth=2)
    plt.plot(train_df_with_lags['year'], lasso_pred, 's--', label=model_names[0], color='green', alpha=0.7)
    plt.plot(train_df_with_lags['year'], xgb_pred, '^--', label=model_names[1], color='red', alpha=0.7)
    
    plt.title('Model Comparison: Full Dataset Predictions vs Actual Values')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_model_comparison_predictions.png'))
    
    # Create residual plots
    plt.figure(figsize=(15, 10))
    
    # LASSO residuals
    plt.subplot(2, 2, 1)
    lasso_residuals = y_all - lasso_pred
    plt.scatter(lasso_pred, lasso_residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'{model_names[0]} Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # XGBoost residuals
    plt.subplot(2, 2, 2)
    xgb_residuals = y_all - xgb_pred
    plt.scatter(xgb_pred, xgb_residuals, alpha=0.5, color='red')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'{model_names[1]} Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # LASSO residuals histogram
    plt.subplot(2, 2, 3)
    plt.hist(lasso_residuals, bins=20, alpha=0.7, color='green')
    plt.title(f'{model_names[0]} Residuals Histogram')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # XGBoost residuals histogram
    plt.subplot(2, 2, 4)
    plt.hist(xgb_residuals, bins=20, alpha=0.7, color='red')
    plt.title(f'{model_names[1]} Residuals Histogram')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'))
    
    # Determine overall winner
    lasso_wins = sum(1 for m in comparison_table['Better Model'] if m == model_names[0])
    xgb_wins = sum(1 for m in comparison_table['Better Model'] if m == model_names[1])
    
    full_lasso_wins = sum(1 for m in full_comparison_table['Better Model'] if m == model_names[0])
    full_xgb_wins = sum(1 for m in full_comparison_table['Better Model'] if m == model_names[1])
    
    print("\n" + "="*50)
    print("OVERALL WINNER")
    print("="*50)
    print(f"Validation metrics: {model_names[0]} wins {lasso_wins} metrics, {model_names[1]} wins {xgb_wins} metrics")
    print(f"Full dataset metrics: {model_names[0]} wins {full_lasso_wins} metrics, {model_names[1]} wins {full_xgb_wins} metrics")
    
    total_lasso_wins = lasso_wins + full_lasso_wins
    total_xgb_wins = xgb_wins + full_xgb_wins
    
    winner = model_names[0] if total_lasso_wins > total_xgb_wins else model_names[1]
    print(f"\nOverall winner: {winner} with {max(total_lasso_wins, total_xgb_wins)} out of {total_lasso_wins + total_xgb_wins} metrics")
    
    # Return summary dictionary
    return {
        'validation_metrics': comparison_table,
        'full_metrics': full_comparison_table,
        'rotation_metrics': rotation_metrics_df,
        'aggregated_predictions': agg_df,
        'overall_winner': winner
    }

if __name__ == "__main__":
    # Base directory and data path configuration
    base_dir = "C:\\Users\\benra\\Documents\\Newspaper"  # Parent directory
    output_dir = os.path.join(base_dir, "regression_finals", "model_comparison_results")
    data_dir = os.path.join(base_dir, "yearly_occurrence_data")  # Data directory is at parent level
    
    # Current directory where the scripts are located
    current_dir = os.path.join(base_dir, "regression_finals")
    
    # File paths - adjust these to match your file locations
    lasso_file = os.path.join(current_dir, "lasso_basic.py")
    xgb_file = os.path.join(current_dir, "xgb.py")
    
    # Check if data path exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        print("Please update the data_dir path in the script.")
        sys.exit(1)
        
    data_file = os.path.join(data_dir, "training_data_1900_1936.csv")
    if not os.path.exists(data_file):
        print(f"Error: Training data file not found at {data_file}")
        print("Please update the data file path in the script.")
        sys.exit(1)
        
    print(f"Data file found at: {data_file}")
    
    # Run comparison
    results = evaluate_and_compare_models(base_dir, output_dir, data_dir, lasso_file, xgb_file)
    
    print("\nComparison complete! Results are saved in:", output_dir)