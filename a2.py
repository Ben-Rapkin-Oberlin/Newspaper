import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from collections import defaultdict

def rotate_series(df, rotation_years):
    """Rotate the dataframe by n years, moving end points to beginning"""
    return pd.concat([df.iloc[-rotation_years:], df.iloc[:-rotation_years]]).reset_index(drop=True)

def select_features(X, y, n_features):
    """Select top n_features using mutual information"""
    if n_features is not None and n_features < X.shape[1]:
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        return pd.DataFrame(X_selected, columns=selected_features), selector
    return X, None

def find_best_gpr_params(X_train, y_train):
    """Perform grid search for GPR hyperparameters"""
    # Define different kernels to try
    kernels = [
        RBF([1.0] * X_train.shape[1], (1e-3, 1e3)),
        RBF([1.0] * X_train.shape[1], (1e-3, 1e3)) + WhiteKernel(1e-1),
        Matern(length_scale=[1.0] * X_train.shape[1], nu=1.5),
        Matern(length_scale=[1.0] * X_train.shape[1], nu=2.5),
        C(1.0, (1e-3, 1e3)) * RBF([1.0] * X_train.shape[1], (1e-3, 1e3)) + WhiteKernel(1e-1)
    ]
    
    best_score = float('inf')
    best_gpr = None
    
    for kernel in kernels:
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=True  # Important for this case
        )
        
        # Use cross-validation to evaluate
        scores = cross_val_score(
            gpr, X_train, y_train,
            scoring='neg_mean_absolute_error',
            cv=5
        )
        mean_score = -np.mean(scores)
        
        if mean_score < best_score:
            best_score = mean_score
            best_gpr = gpr
    
    # Fit the best model to get optimized kernel parameters
    best_gpr.fit(X_train, y_train)
    print("\nBest GPR kernel:", best_gpr.kernel_)
    print("Best GPR MAE score:", best_score)
    
    return best_gpr

def find_best_adaboost_params(X_train, y_train):
    """Perform enhanced grid search for AdaBoost hyperparameters"""
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1, 0.5],
        'estimator__max_depth': [1, 3],
        'estimator__min_samples_leaf': [1],
        'loss': ['linear', 'square', 'exponential']
    }
    
    base_estimator = DecisionTreeRegressor()
    adaboost = AdaBoostRegressor(estimator=base_estimator, random_state=42)
    
    grid_search = GridSearchCV(
        adaboost,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print("\nBest AdaBoost parameters:", grid_search.best_params_)
    print("Best MAE score:", -grid_search.best_score_)
    
    return grid_search.best_params_

def create_optimized_adaboost(best_params):
    """Create an AdaBoost model with optimized parameters"""
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(
            max_depth=best_params['estimator__max_depth'],
            min_samples_leaf=best_params['estimator__min_samples_leaf']
        ),
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        loss=best_params['loss'],
        random_state=42
    )

def evaluate_rotation(X, y, years, n_features, val_size, best_params=None):
    """Evaluate models on a specific rotation of the data"""
    
    # Feature selection
    X_selected, selector = select_features(X, y, n_features)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns)
    
    # Split into training and validation
    X_train = X_scaled[:-val_size]
    y_train = y[:-val_size]
    X_val = X_scaled[-val_size:]
    y_val = y[-val_size:]
    val_years = years[-val_size:].values
    
    # Initialize and train all models
    # AdaBoost
    if best_params is None:
        best_params = find_best_adaboost_params(X_train, y_train)
    adaboost_model = create_optimized_adaboost(best_params)
    adaboost_model.fit(X_train, y_train)
    
    # Lasso
    lasso_model = Lasso(alpha=1.0, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Theil-Sen
    theil_sen_model = TheilSenRegressor(random_state=42, max_iter=1000)
    theil_sen_model.fit(X_train, y_train)
    
    # Gaussian Process with optimized parameters
    gpr_model = find_best_gpr_params(X_train, y_train)
    
    # Make predictions for all models
    # Get predictions and uncertainty
    gpr_pred_train, gpr_std_train = gpr_model.predict(X_train, return_std=True)
    gpr_pred_val, gpr_std_val = gpr_model.predict(X_val, return_std=True)
    
    predictions = {
        'ada': {
            'train': adaboost_model.predict(X_train),
            'val': adaboost_model.predict(X_val)
        },
        'lasso': {
            'train': lasso_model.predict(X_train),
            'val': lasso_model.predict(X_val)
        },
        'theil_sen': {
            'train': theil_sen_model.predict(X_train),
            'val': theil_sen_model.predict(X_val)
        },
        'gpr': {
            'train': gpr_pred_train,
            'val': gpr_pred_val,
            'train_std': gpr_std_train,
            'val_std': gpr_std_val
        }
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

    # Model parameters
    params = {
        'n_features': 15,
        'val_size': val_size
    }

    # Find best parameters using the initial data
    print("Finding best parameters for AdaBoost...")
    initial_X, _ = select_features(X, y, params['n_features'])
    best_params = find_best_adaboost_params(initial_X, y)

    # Evaluate all rotations
    results = {}
    validation_predictions = defaultdict(list)
    validation_actuals = defaultdict(list)

    # Initialize dictionaries to store std values
    validation_uncertainties = defaultdict(list)
    
    for i in range(n_rotations):
        if i > 0:
            X = rotate_series(X, rotation_size)
            y = rotate_series(pd.DataFrame(y), rotation_size).iloc[:, 0]
            years = rotate_series(pd.DataFrame(years), rotation_size).iloc[:, 0]
        
        key = f"Rotation {i+1}"
        results[key] = evaluate_rotation(X, y, years, best_params=best_params, **params)
        
        # Store validation predictions, uncertainties, and actuals by year
        val_years = results[key]['years']['val']
        val_actuals = results[key]['actual']['val']
        
        for model_name in ['ada', 'lasso', 'theil_sen', 'gpr']:
            val_preds = results[key]['predictions'][model_name]['val']
            for year, pred, actual in zip(val_years, val_preds, val_actuals):
                validation_predictions[year].append({
                    model_name: pred
                })
                validation_actuals[year].append(actual)
                
            # Store GPR uncertainties
            if model_name == 'gpr':
                val_stds = results[key]['predictions']['gpr']['val_std']
                for year, std in zip(val_years, val_stds):
                    validation_uncertainties[year].append(std)

    # Compute aggregated validation metrics by year
    aggregated_metrics = {}
    for year in validation_predictions.keys():
        predictions = validation_predictions[year]
        actuals = validation_actuals[year]
        
        metrics = {
            'actual': np.mean(actuals)
        }
        
        for model_name in ['ada', 'lasso', 'theil_sen', 'gpr']:
            metrics[f'{model_name}_pred'] = np.mean([
                p[model_name] for p in predictions if model_name in p
            ])
        
        # Add uncertainty for GPR
        metrics[f'gpr_uncertainty'] = np.mean(validation_uncertainties[year])
        
        aggregated_metrics[year] = metrics

    # Convert to DataFrame and compute metrics
    agg_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')
    agg_df.index.name = 'year'
    agg_df = agg_df.sort_index()

    # Compute overall aggregated metrics
    print("\nAggregated Validation Metrics:")
    for model_name in ['ada', 'lasso', 'theil_sen', 'gpr']:
        mae = mean_absolute_error(agg_df['actual'], agg_df[f'{model_name}_pred'])
        r2 = r2_score(agg_df['actual'], agg_df[f'{model_name}_pred'])
        print(f"{model_name.upper()} MAE: {mae:.2f}")
        print(f"{model_name.upper()} R²: {r2:.3f}")

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
    
    # Plot predictions
    ax1.plot(agg_df.index, agg_df['actual'], 'o-', label='Actual', color='blue')
    ax1.plot(agg_df.index, agg_df['ada_pred'], 's--', label='AdaBoost', color='red')
    ax1.plot(agg_df.index, agg_df['lasso_pred'], '^--', label='Lasso', color='green')
    ax1.plot(agg_df.index, agg_df['theil_sen_pred'], 'D--', label='Theil-Sen', color='purple')
    ax1.plot(agg_df.index, agg_df['gpr_pred'], 'v--', label='GPR', color='orange')
    
    # Add GPR confidence intervals
    gpr_std = agg_df['gpr_uncertainty']
    ax1.fill_between(agg_df.index, 
                    agg_df['gpr_pred'] - 2*gpr_std,
                    agg_df['gpr_pred'] + 2*gpr_std,
                    color='orange', alpha=0.2, label='GPR 95% CI')
    
    ax1.set_title('Aggregated Validation Predictions vs Actual Values')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Estimated Deaths')
    ax1.legend()
    ax1.grid(True)
    
    # Plot GPR uncertainty
    ax2.plot(agg_df.index, agg_df['gpr_uncertainty'], 'o-', color='orange', label='GPR Uncertainty')
    ax2.fill_between(agg_df.index, 0, agg_df['gpr_uncertainty'], color='orange', alpha=0.2)
    ax2.set_title('GPR Prediction Uncertainty')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()