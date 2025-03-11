import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso
from collections import defaultdict

def create_mixed_depth_adaboost(n_depth1, n_depth2, min_samples_leaf, learning_rate=0.1):
    """Create an AdaBoost ensemble with mixed depth trees"""
    model_depth1 = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=1),
        n_estimators=n_depth1,
        learning_rate=learning_rate,
        random_state=42
    )
    
    model_depth2 = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=min_samples_leaf),
        n_estimators=n_depth2,
        learning_rate=learning_rate,
        random_state=43
    )
    
    return model_depth1, model_depth2

def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def evaluate_rotation(X, y, years, n_features, n_depth1, n_depth2, min_samples_leaf, 
                     learning_rate, val_size):
    """Evaluate models on a specific rotation of the data"""
    
    # Feature selection
    if n_features is not None and n_features < X.shape[1]:
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, columns=selected_features)
    
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
    
    # Train AdaBoost models
    model_depth1, model_depth2 = create_mixed_depth_adaboost(
        n_depth1, n_depth2, min_samples_leaf, learning_rate
    )
    
    model_depth1.fit(X_train, y_train)
    model_depth2.fit(X_train, y_train)
    
    # Train Lasso Regression
    lasso_model = Lasso(alpha=1.0, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Make predictions - AdaBoost
    pred_depth1_train = model_depth1.predict(X_train)
    pred_depth2_train = model_depth2.predict(X_train)
    pred_depth1_val = model_depth1.predict(X_val)
    pred_depth2_val = model_depth2.predict(X_val)
    
    # Make predictions - Lasso
    lasso_pred_train = lasso_model.predict(X_train)
    lasso_pred_val = lasso_model.predict(X_val)
    
    # Combine AdaBoost predictions
    ada_pred_train = (pred_depth1_train + pred_depth2_train) / 2
    ada_pred_val = (pred_depth1_val + pred_depth2_val) / 2
    
    return {
        'years': {'val': val_years},
        'predictions': {
            'ada_train': ada_pred_train, 
            'ada_val': ada_pred_val,
            'lasso_train': lasso_pred_train,
            'lasso_val': lasso_pred_val
        },
        'actual': {'train': y_train, 'val': y_val},
        'metrics': {
            'ada_train_mae': mean_absolute_error(y_train, ada_pred_train),
            'ada_val_mae': mean_absolute_error(y_val, ada_pred_val),
            'ada_train_r2': r2_score(y_train, ada_pred_train),
            'ada_val_r2': r2_score(y_val, ada_pred_val),
            'lasso_train_mae': mean_absolute_error(y_train, lasso_pred_train),
            'lasso_val_mae': mean_absolute_error(y_val, lasso_pred_val),
            'lasso_train_r2': r2_score(y_train, lasso_pred_train),
            'lasso_val_r2': r2_score(y_val, lasso_pred_val)
        }
    }

def main():
    # Load data
    train_path = "yearly_occurrence_data/training_data_1900_1936.csv"
    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("year").reset_index(drop=True)

    # Parameters
    rotation_size = 1  # Number of years to rotate by
    val_size = 5       # Validation window size
    n_rotations = len(train_df) // rotation_size  # Number of rotations to perform

    # Get features
    X = train_df.drop(['year', 'estimated_deaths'], axis=1)
    y = train_df['estimated_deaths']
    years = train_df['year']

    # Model parameters
    params = {
        'n_features': 10,
        'n_depth1': 200,
        'n_depth2': 60,
        'min_samples_leaf': 4,
        'learning_rate': 0.1,
        'val_size': val_size
    }

    # Evaluate all rotations
    results = {}
    validation_predictions = defaultdict(list)
    validation_actuals = defaultdict(list)

    for i in range(n_rotations):
        if i > 0:
            X = rotate_series(X, rotation_size)
            y = rotate_series(pd.DataFrame(y), rotation_size).iloc[:, 0]
            years = rotate_series(pd.DataFrame(years), rotation_size).iloc[:, 0]
        
        key = f"Rotation {i+1}"
        results[key] = evaluate_rotation(X, y, years, **params)
        
        # Store validation predictions and actuals by year
        val_years = results[key]['years']['val']
        ada_val_preds = results[key]['predictions']['ada_val']
        lasso_val_preds = results[key]['predictions']['lasso_val']
        val_actuals = results[key]['actual']['val']
        
        for year, ada_pred, lasso_pred, actual in zip(val_years, ada_val_preds, lasso_val_preds, val_actuals):
            validation_predictions[year].append({
                'ada': ada_pred,
                'lasso': lasso_pred
            })
            validation_actuals[year].append(actual)

    # Compute aggregated validation metrics by year
    aggregated_metrics = {}
    for year in validation_predictions.keys():
        predictions = validation_predictions[year]
        actuals = validation_actuals[year]
        
        ada_avg_pred = np.mean([p['ada'] for p in predictions])
        lasso_avg_pred = np.mean([p['lasso'] for p in predictions])
        actual_avg = np.mean(actuals)
        
        aggregated_metrics[year] = {
            'ada_pred': ada_avg_pred,
            'lasso_pred': lasso_avg_pred,
            'actual': actual_avg
        }

    # Convert to DataFrame and compute metrics
    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    agg_df.index.name = 'year'
    agg_df = agg_df.sort_index()

    # Compute overall aggregated metrics
    ada_mae = mean_absolute_error(agg_df['actual'], agg_df['ada_pred'])
    lasso_mae = mean_absolute_error(agg_df['actual'], agg_df['lasso_pred'])
    ada_r2 = r2_score(agg_df['actual'], agg_df['ada_pred'])
    lasso_r2 = r2_score(agg_df['actual'], agg_df['lasso_pred'])

    print("\nAggregated Validation Metrics:")
    print(f"AdaBoost MAE: {ada_mae:.2f}")
    print(f"Lasso MAE: {lasso_mae:.2f}")
    print(f"AdaBoost R²: {ada_r2:.3f}")
    print(f"Lasso R²: {lasso_r2:.3f}")

    # Visualize aggregated results
    plt.figure(figsize=(15, 6))
    plt.plot(agg_df.index, agg_df['actual'], 'o-', label='Actual', color='blue')
    plt.plot(agg_df.index, agg_df['ada_pred'], 's--', label='AdaBoost', color='red')
    plt.plot(agg_df.index, agg_df['lasso_pred'], '^--', label='Lasso', color='green')
    plt.title('Aggregated Validation Predictions vs Actual Values')
    plt.xlabel('Year')
    plt.ylabel('Estimated Deaths')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print detailed metrics by year
    #print("\nDetailed Metrics by Year:")
    #print(agg_df)

if __name__ == "__main__":
    main()