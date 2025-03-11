import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

class QuantileRandomForest:
    def __init__(self, n_estimators=100, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        self.rf.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        quantile_preds = {}
        for q in self.quantiles:
            quantile_preds[q] = np.percentile(predictions, q * 100, axis=0)
        return quantile_preds

class QuantileGradientBoosting:
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {}
        
        for q in quantiles:
            self.models[q] = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                max_depth=4,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
    
    def fit(self, X, y):
        for q in self.quantiles:
            self.models[q].fit(X, y)
        return self
    
    def predict(self, X):
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)
        return predictions

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculate and print comprehensive metrics"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²: {metrics['r2']:.3f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    return metrics

def main():
    # Load and prepare data
    train_path = "yearly_occurrence_data/training_data_1900_1936.csv"
    df = pd.read_csv(train_path)
    df = df.sort_values("year").reset_index(drop=True)
    
    # Create features (including lags)
    X = df.drop(['year', 'estimated_deaths'], axis=1)
    y = df['estimated_deaths']
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Train-validation split
    val_size = 5
    X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
    y_train, y_val = y_scaled[:-val_size], y_scaled[-val_size:]
    
    # 1. Quantile Random Forest
    qrf = QuantileRandomForest(quantiles=[0.1, 0.5, 0.9])
    qrf.fit(X_train, y_train)
    qrf_preds = qrf.predict(X_val)
    
    # 2. Quantile Gradient Boosting
    qgb = QuantileGradientBoosting(quantiles=[0.1, 0.5, 0.9])
    qgb.fit(X_train, y_train)
    qgb_preds = qgb.predict(X_val)
    
    # Evaluate models
    # Convert predictions back to original scale
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    qrf_median = scaler_y.inverse_transform(qrf_preds[0.5].reshape(-1, 1)).ravel()
    qgb_median = scaler_y.inverse_transform(qgb_preds[0.5].reshape(-1, 1)).ravel()
    
    # Scale back quantiles for QRF
    qrf_lower = scaler_y.inverse_transform(qrf_preds[0.1].reshape(-1, 1)).ravel()
    qrf_upper = scaler_y.inverse_transform(qrf_preds[0.9].reshape(-1, 1)).ravel()
    
    # Scale back quantiles for QGB
    qgb_lower = scaler_y.inverse_transform(qgb_preds[0.1].reshape(-1, 1)).ravel()
    qgb_upper = scaler_y.inverse_transform(qgb_preds[0.9].reshape(-1, 1)).ravel()
    
    # Evaluate each model
    qrf_metrics = evaluate_predictions(y_val_orig, qrf_median, "Quantile Random Forest")
    qgb_metrics = evaluate_predictions(y_val_orig, qgb_median, "Quantile Gradient Boosting")
    
    # Create visualization
    years = df['year'].values[-val_size:]
    
    # Plot 1: Predictions with confidence intervals
    plt.figure(figsize=(15, 8))
    
    # Actual values
    plt.plot(years, y_val_orig, 'ko-', label='Actual', linewidth=2)
    
    # QRF predictions and intervals
    plt.plot(years, qrf_median, 's--', label='QRF (median)', color='blue', alpha=0.7)
    plt.fill_between(years, qrf_lower, qrf_upper, 
                    alpha=0.2, color='blue', label='QRF 80% Interval')
    
    # QGB predictions and intervals
    plt.plot(years, qgb_median, '^--', label='QGB (median)', color='red', alpha=0.7)
    plt.fill_between(years, qgb_lower, qgb_upper, 
                    alpha=0.2, color='red', label='QGB 80% Interval')
    
    plt.title('Model Comparison: Quantile Regression Models')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 2: Error Analysis
    plt.figure(figsize=(15, 6))
    
    # Calculate absolute errors
    qrf_errors = np.abs(qrf_median - y_val_orig)
    qgb_errors = np.abs(qgb_median - y_val_orig)
    
    plt.plot(years, qrf_errors, 'bo-', label='QRF Error', alpha=0.7)
    plt.plot(years, qgb_errors, 'ro-', label='QGB Error', alpha=0.7)
    
    plt.title('Absolute Prediction Errors Over Time')
    plt.xlabel('Year')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot 3: Box plot of errors
    plt.figure(figsize=(10, 6))
    error_data = [qrf_errors, qgb_errors]
    plt.boxplot(error_data, labels=['QRF', 'QGB'])
    plt.title('Distribution of Absolute Errors by Model')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()