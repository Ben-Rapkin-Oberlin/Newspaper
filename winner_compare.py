import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    median_absolute_error, max_error, explained_variance_score
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

class NormalizedQuantileRandomForest:
    def __init__(self, n_estimators=100, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Ensure y is 2D for StandardScaler
        y = y.reshape(-1, 1)
        # Normalize y values
        y_normalized = self.y_scaler.fit_transform(y).ravel()
        self.rf.fit(X, y_normalized)
        return self
        
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        median_pred = np.percentile(predictions, 50, axis=0)
        # Inverse transform the predictions
        return self.y_scaler.inverse_transform(median_pred.reshape(-1, 1)).ravel()
        
    def predict_quantiles(self, X):
        if hasattr(X, 'values'):
            X = X.values
        predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        quantile_predictions = {}
        for q in self.quantiles:
            q_pred = np.percentile(predictions, q * 100, axis=0)
            # Inverse transform the quantile predictions
            quantile_predictions[q] = self.y_scaler.inverse_transform(
                q_pred.reshape(-1, 1)
            ).ravel()
        return quantile_predictions

        
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
    """Perform grid search for Lasso with improved convergence"""
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'max_iter': [7000],
        'tol': [1e-4],
        'selection': ['random'],
        'warm_start': [True]
    }
    
    lasso = Lasso(random_state=42)
    grid_search = GridSearchCV(
        lasso, param_grid, cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Lasso parameters:", grid_search.best_params_)
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


def evaluate_rotation_enhanced(X, y, years, val_size, ada_params, xgb_params, lasso_params):
    """Evaluate all models on a specific rotation of the data with enhanced metrics and normalization"""
    
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
    
    # Create and train models
    from a2 import create_optimized_adaboost
    
    # AdaBoost
    ada_model = create_optimized_adaboost(ada_params)
    ada_model.fit(X_train, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Lasso
    lasso_model = Lasso(**lasso_params, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Quantile Random Forest (already handles normalization internally)
    qrf_model = NormalizedQuantileRandomForest(n_estimators=100)
    qrf_model.fit(X_train, y_train)
    
    # Get predictions and inverse transform
    predictions = {
        'ada': {
            'train': y_scaler.inverse_transform(ada_model.predict(X_train).reshape(-1, 1)).ravel(),
            'val': y_scaler.inverse_transform(ada_model.predict(X_val).reshape(-1, 1)).ravel()
        },
        'xgb': {
            'train': y_scaler.inverse_transform(xgb_model.predict(X_train).reshape(-1, 1)).ravel(),
            'val': y_scaler.inverse_transform(xgb_model.predict(X_val).reshape(-1, 1)).ravel()
        },
        'lasso': {
            'train': y_scaler.inverse_transform(lasso_model.predict(X_train).reshape(-1, 1)).ravel(),
            'val': y_scaler.inverse_transform(lasso_model.predict(X_val).reshape(-1, 1)).ravel()
        },
        'qrf': {
            'train': qrf_model.predict(X_train),
            'val': qrf_model.predict(X_val)
        }
    }
    
    # Get quantile predictions for QRF (already inverse transformed)
    qrf_quantiles = {
        'train': qrf_model.predict_quantiles(X_train),
        'val': qrf_model.predict_quantiles(X_val)
    }
    
    # Inverse transform the actual values for metric calculation
    y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_val_original = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    
    # Calculate metrics using original scale values
    metrics = {}
    for model_name, preds in predictions.items():
        # Training metrics
        train_metrics = calculate_metrics(y_train_original, preds['train'], f'{model_name}_train_')
        # Validation metrics
        val_metrics = calculate_metrics(y_val_original, preds['val'], f'{model_name}_val_')
        metrics.update(train_metrics)
        metrics.update(val_metrics)
    
    return {
        'years': {'val': val_years},
        'predictions': predictions,
        'actual': {'train': y_train_original, 'val': y_val_original},
        'metrics': metrics,
        'qrf_quantiles': qrf_quantiles,
        'scalers': {'x': x_scaler, 'y': y_scaler}  # Save scalers for future use
    }
 
def find_best_xgboost_params(X_train, y_train):
    """Perform grid search for XGBoost with comprehensive parameter grid"""
    print('XGBoost search')
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7,.9,1],
        'min_child_weight': [1,3,4]
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
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest XGBoost parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_
 
def main():
    # Load data
    train_path = "yearly_occurrence_data/training_data_1900_1936.csv"
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
    
    print(X.shape)
    y = train_df_with_lags['estimated_deaths']
    years = train_df_with_lags['year']
    
    # Get best parameters for all models
    print("Finding best parameters...")
    from a2 import find_best_adaboost_params
    
    best_params = {
        'ada': find_best_adaboost_params(X, y),
        'xgb': find_best_xgboost_params(X,y),
        'lasso': find_best_lasso_params(X, y)
    }
    
    # Initialize storage
    results = {}
    validation_predictions = defaultdict(list)
    validation_actuals = defaultdict(list)
    validation_qrf_quantiles = defaultdict(list)
    
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
            ada_params=best_params['ada'],
            xgb_params=best_params['xgb'],
            lasso_params=best_params['lasso']
        )
        
        # Store validation predictions and actuals
        val_years = results[key]['years']['val']
        val_actuals = results[key]['actual']['val']
        
        for model_name in ['ada', 'xgb', 'lasso', 'qrf']:
            val_preds = results[key]['predictions'][model_name]['val']
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append({
                    model_name: pred
                })
                validation_actuals[year].append(actual)
                
                # Store QRF quantiles
                if model_name == 'qrf':
                    validation_qrf_quantiles[year].append(
                        results[key]['qrf_quantiles']['val']
                    )

    # Compute aggregated metrics
    aggregated_metrics = {}
    for year in validation_predictions.keys():
        predictions = validation_predictions[year]
        actuals = validation_actuals[year]
        
        metrics = {
            'actual': np.mean(actuals)
        }
        
        for model_name in ['ada', 'xgb', 'lasso', 'qrf']:
            metrics[f'{model_name}_pred'] = np.mean([
                p[model_name] for p in predictions if model_name in p
            ])
        
        aggregated_metrics[year] = metrics

    # Convert to DataFrame
    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    agg_df.index.name = 'year'
    agg_df = agg_df.sort_index()

    # Print comprehensive metrics
    print("\nAggregated Validation Metrics:")
    model_names = ['ada', 'xgb', 'lasso', 'qrf']
    metrics_summary = []
    
    for model_name in model_names:
        metrics = calculate_metrics(agg_df['actual'], agg_df[f'{model_name}_pred'])
        metrics_summary.append({
            'Model': model_name.upper(),
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'MedianAE': metrics['median_ae'],
            'MaxError': metrics['max_error'],
            'ExplainedVar': metrics['explained_variance'],
            'MAPE': metrics['mape']
        })
        
        print(f"\n{model_name.upper()} Metrics:")
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
    
    colors = ['blue', 'red', 'green', 'purple']
    for model_name, color in zip(model_names, colors):
        plt.plot(agg_df.index, agg_df[f'{model_name}_pred'], 
                's--', label=model_name.upper(), color=color, alpha=0.7)
        
        # Add confidence interval for QRF
        if model_name == 'qrf':
            qrf_lower = []
            qrf_upper = []
            for year in agg_df.index:
                quantiles = [q[0.1] for q in validation_qrf_quantiles[year]]
                qrf_lower.append(np.mean(quantiles))
                quantiles = [q[0.9] for q in validation_qrf_quantiles[year]]
                qrf_upper.append(np.mean(quantiles))
            
            plt.fill_between(agg_df.index, qrf_lower, qrf_upper,
                           color='purple', alpha=0.2, label='QRF 80% Interval')
    
    plt.title('Model Comparison: Aggregated Validation Predictions vs Actual Values')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Create error analysis plot
    plt.figure(figsize=(15, 6))
    for model_name, color in zip(model_names, colors):
        errors = np.abs(agg_df[f'{model_name}_pred'] - agg_df['actual'])
        plt.plot(agg_df.index, errors, 
                'o-', label=f'{model_name.upper()} Error', 
                color=color, alpha=0.7)
    
    plt.title('Absolute Prediction Errors Over Time')
    plt.xlabel('Year')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Additional visualization: Box plot of errors
    plt.figure(figsize=(10, 6))
    error_data = []
    labels = []
    for model_name in model_names:
        errors = np.abs(agg_df[f'{model_name}_pred'] - agg_df['actual'])
        error_data.append(errors)
        labels.extend([model_name.upper()] * len(errors))
    
    plt.boxplot(error_data, labels=[m.upper() for m in model_names])
    plt.title('Distribution of Absolute Errors by Model')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()