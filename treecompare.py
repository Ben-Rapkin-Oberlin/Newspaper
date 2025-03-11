import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict

def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def find_best_rf_params(X_train, y_train):
    """Perform grid search for Random Forest parameters"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 4, 6]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Random Forest parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def find_best_xgb_params(X_train, y_train):
    """Perform grid search for XGBoost parameters"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest XGBoost parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def find_best_lgb_params(X_train, y_train):
    """Perform grid search for LightGBM parameters"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [4, 8, 16],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_data_in_leaf': [10, 20, 30]  # Added to prevent overfitting
    }
    
    lgb_model = lgb.LGBMRegressor(
        random_state=42,
        verbose=-1,  # Suppress warnings
        min_data_in_bin=5  # Ensure enough data points in each bin
    )
    
    grid_search = GridSearchCV(
        lgb_model, param_grid, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest LightGBM parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def find_best_gbm_params(X_train, y_train):
    """Perform grid search for Gradient Boosting parameters"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'loss': ['squared_error', 'absolute_error', 'huber']
    }
    
    gbm = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        gbm, param_grid, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Gradient Boosting parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def create_models_with_best_params(rf_params, xgb_params, lgb_params, gbm_params):
    """Create models with optimized parameters"""
    models = {
        'rf': RandomForestRegressor(**rf_params, random_state=42),
        'xgb': xgb.XGBRegressor(**xgb_params, random_state=42),
        'lgb': lgb.LGBMRegressor(
            **lgb_params,
            random_state=42,
            verbose=-1,
            min_data_in_bin=5
        ),
        'gbm': GradientBoostingRegressor(**gbm_params, random_state=42)
    }
    return models

def evaluate_rotation_enhanced(X, y, years, val_size, best_params):
    """Evaluate all models on a specific rotation of the data"""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split into training and validation
    X_train = X_scaled[:-val_size]
    y_train = y[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y[-val_size:]
    val_years = years[-val_size:].values
    
    # Create and train AdaBoost model
    from a2 import create_optimized_adaboost
    adaboost_model = create_optimized_adaboost(best_params['ada'])
    adaboost_model.fit(X_train, y_train)
    
    # Get AdaBoost predictions
    ada_train_pred = adaboost_model.predict(X_train)
    ada_val_pred = adaboost_model.predict(X_val)
    
    # Initialize and train all tree models
    models = create_models_with_best_params(
        best_params['rf'],
        best_params['xgb'],
        best_params['lgb'],
        best_params['gbm']
    )
    
    predictions = {
        'ada': {
            'train': ada_train_pred,
            'val': ada_val_pred
        }
    }
    
    # Train and get predictions for each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = {
            'train': model.predict(X_train),
            'val': model.predict(X_val)
        }
    
    # Calculate metrics for all models
    metrics = {}
    for model_name, preds in predictions.items():
        metrics[f'{model_name}_train_mae'] = mean_absolute_error(y_train, preds['train'])
        metrics[f'{model_name}_val_mae'] = mean_absolute_error(y_val, preds['val'])
        metrics[f'{model_name}_train_r2'] = r2_score(y_train, preds['train'])
        metrics[f'{model_name}_val_r2'] = r2_score(y_val, preds['val'])
    
    return {
        'years': {'val': val_years},
        'predictions': predictions,
        'actual': {'train': y_train, 'val': y_val},
        'metrics': metrics
    }

def main():
    # Load data
    train_path = "yearly_occurrence_data/training_data_1900_1936.csv"
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)

    # Parameters
    rotation_size = 5  # Number of years to rotate by
    val_size = 5       # Validation window size
    n_rotations = len(train_df) // rotation_size  # Number of rotations to perform

    # Get features
    X = train_df.drop(['year', 'estimated_deaths'], axis=1)
    y = train_df['estimated_deaths']
    years = train_df['year']

    # Initialize storage for results
    results = {}
    validation_predictions = defaultdict(list)
    validation_actuals = defaultdict(list)
    
    # Get parameters for all models using the initial data
    print("Finding best parameters for all models...")
    from a2 import find_best_adaboost_params
    
    best_params = {
        'ada': find_best_adaboost_params(X, y),
        'rf': find_best_rf_params(X, y),
        'xgb': find_best_xgb_params(X, y),
        'lgb': find_best_lgb_params(X, y),
        'gbm': find_best_gbm_params(X, y)
    }
    
    # Evaluate all rotations
    for i in range(n_rotations):
        if i > 0:
            X = rotate_series(X, rotation_size)
            y = rotate_series(pd.DataFrame(y), rotation_size).iloc[:, 0]
            years = rotate_series(pd.DataFrame(years), rotation_size).iloc[:, 0]
        
        key = f"Rotation {i+1}"
        results[key] = evaluate_rotation_enhanced(
            X, y, years, 
            val_size=val_size,
            best_params=best_params
        )
        
        # Store validation predictions and actuals by year
        val_years = results[key]['years']['val']
        val_actuals = results[key]['actual']['val']
        
        for model_name in ['ada', 'rf', 'xgb', 'lgb', 'gbm']:
            val_preds = results[key]['predictions'][model_name]['val']
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append({
                    model_name: pred
                })
                validation_actuals[year].append(actual)

    # Compute aggregated validation metrics by year
    aggregated_metrics = {}
    for year in validation_predictions.keys():
        predictions = validation_predictions[year]
        actuals = validation_actuals[year]
        
        metrics = {
            'actual': np.mean(actuals)
        }
        
        for model_name in ['ada', 'rf', 'xgb', 'lgb', 'gbm']:
            metrics[f'{model_name}_pred'] = np.mean([
                p[model_name] for p in predictions if model_name in p
            ])
        
        aggregated_metrics[year] = metrics

    # Convert to DataFrame and compute metrics
    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    agg_df.index.name = 'year'
    agg_df = agg_df.sort_index()

    # Compute and print overall aggregated metrics
    print("\nAggregated Validation Metrics:")
    model_names = ['ada', 'rf', 'xgb', 'lgb', 'gbm']
    metrics_summary = []
    
    for model_name in model_names:
        mae = mean_absolute_error(agg_df['actual'], agg_df[f'{model_name}_pred'])
        r2 = r2_score(agg_df['actual'], agg_df[f'{model_name}_pred'])
        metrics_summary.append({
            'Model': model_name.upper(),
            'MAE': mae,
            'R²': r2
        })
        print(f"{model_name.upper()} - MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    # Plot actual values
    plt.plot(agg_df.index, agg_df['actual'], 'o-', label='Actual', color='black', linewidth=2)
    
    # Plot predictions for each model
    for model_name, color in zip(model_names, colors):
        plt.plot(agg_df.index, agg_df[f'{model_name}_pred'], 
                's--', label=model_name.upper(), color=color, alpha=0.7)
    
    plt.title('Model Comparison: Aggregated Validation Predictions vs Actual Values')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Create metrics comparison plot
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.set_index('Model', inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE comparison
    metrics_df['MAE'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_ylabel('MAE')
    ax1.grid(True, axis='y')
    
    # R² comparison
    metrics_df['R²'].plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('R² Score Comparison')
    ax2.set_ylabel('R²')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()