import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        self.rf.fit(X, y)
        return self
        
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        return np.percentile(predictions, 50, axis=0)
        
    def predict_quantiles(self, X):
        if hasattr(X, 'values'):
            X = X.values
        predictions = np.array([tree.predict(X) for tree in self.rf.estimators_])
        return {q: np.percentile(predictions, q * 100, axis=0) for q in self.quantiles}

class UncertaintyComparison:
    def __init__(self):
        self.qrf = QuantileRandomForest(n_estimators=100)
        kernel = C(1.0) * RBF([1.0])
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
            n_restarts_optimizer=10
        )
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit both models
        self.qrf.fit(X_scaled, y)
        self.gpr.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # QRF predictions and intervals
        qrf_pred = self.qrf.predict(X_scaled)
        qrf_quantiles = self.qrf.predict_quantiles(X_scaled)
        qrf_lower = qrf_quantiles[0.1]
        qrf_upper = qrf_quantiles[0.9]
        
        # GPR predictions and intervals
        gpr_pred, gpr_std = self.gpr.predict(X_scaled, return_std=True)
        gpr_lower = gpr_pred - 1.96 * gpr_std
        gpr_upper = gpr_pred + 1.96 * gpr_std
        
        return {
            'qrf_pred': qrf_pred,
            'qrf_lower': qrf_lower,
            'qrf_upper': qrf_upper,
            'gpr_pred': gpr_pred,
            'gpr_lower': gpr_lower,
            'gpr_upper': gpr_upper
        }

def create_lag_features(df, target_col, lag_periods=[1, 2]):
    """Create lag features for time series prediction"""
    df = df.copy()
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df.dropna()

def evaluate_predictions(actual, predictions):
    """Calculate comprehensive metrics for both models"""
    metrics = {}
    
    # Calculate coverage (percentage of actual values within bounds)
    def calculate_coverage(lower, upper, actual):
        within_bounds = np.logical_and(actual >= lower, actual <= upper)
        return np.mean(within_bounds) * 100
    
    # Calculate metrics for both models
    for model in ['qrf', 'gpr']:
        pred_key = f'{model}_pred'
        lower_key = f'{model}_lower'
        upper_key = f'{model}_upper'
        
        # Basic regression metrics
        metrics[f'{model}_rmse'] = np.sqrt(mean_squared_error(actual, predictions[pred_key]))
        metrics[f'{model}_r2'] = r2_score(actual, predictions[pred_key])
        
        # Uncertainty metrics
        metrics[f'{model}_coverage'] = calculate_coverage(
            predictions[lower_key],
            predictions[upper_key],
            actual
        )
        metrics[f'{model}_interval_width'] = np.mean(
            predictions[upper_key] - predictions[lower_key]
        )
    
    return metrics

def plot_comparison(years, actual, predictions, title='Model Comparison'):
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.plot(years, actual, 'ko-', label='Actual', linewidth=2)
    
    # Plot QRF predictions and uncertainty
    plt.plot(years, predictions['qrf_pred'], 'b--', label='QRF Prediction', alpha=0.7)
    plt.fill_between(years, 
                    predictions['qrf_lower'], 
                    predictions['qrf_upper'],
                    color='blue', alpha=0.2, label='QRF 80% Interval')
    
    # Plot GPR predictions and uncertainty
    plt.plot(years, predictions['gpr_pred'], 'r--', label='GPR Prediction', alpha=0.7)
    plt.fill_between(years, 
                    predictions['gpr_lower'], 
                    predictions['gpr_upper'],
                    color='red', alpha=0.2, label='GPR 95% Interval')
    
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def main():
    # Load data
    train_path = "yearly_occurrence_data/training_data_1900_1936.csv"
    df = pd.read_csv(train_path)
    df = df.sort_values("year").reset_index(drop=True)
    
    # Add lag features
    df_with_lags = create_lag_features(df, 'estimated_deaths', lag_periods=[1, 2])
    
    # Prepare features and target
    X = df_with_lags.drop(['year', 'estimated_deaths'], axis=1)
    y = df_with_lags['estimated_deaths']
    years = df_with_lags['year']
    
    # Split data
    X_train, X_test, y_train, y_test, years_train, years_test = train_test_split(
        X, y, years, test_size=0.2, shuffle=False
    )
    
    # Create and train models
    comparison = UncertaintyComparison()
    comparison.fit(X_train, y_train)
    
    # Get predictions
    train_predictions = comparison.predict(X_train)
    test_predictions = comparison.predict(X_test)
    
    # Calculate metrics
    train_metrics = evaluate_predictions(y_train, train_predictions)
    test_metrics = evaluate_predictions(y_test, test_predictions)
    
    # Print metrics
    print("\nTraining Metrics:")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.2f}")
    
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.2f}")
    
    # Plot results
    plot_comparison(years_train, y_train, train_predictions, "Training Set Predictions")
    plt.show()
    
    plot_comparison(years_test, y_test, test_predictions, "Test Set Predictions")
    plt.show()

if __name__ == "__main__":
    main()