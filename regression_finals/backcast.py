import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Constant offset for the log transform (should match the one used during training)
LOG_OFFSET = 1

def load_lasso_model(model_path):
    """
    Load the pre-trained LASSO model.
    Expects a dictionary with keys 'model' and 'feature_names'.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    feature_names = model_data.get('feature_names', ['estimated_deaths_lag_1', 'estimated_deaths_lag_2'])
    return model, feature_names

def load_log_lasso_model(model_path):
    """
    Load the pre-trained Log LASSO model.
    Expects a dictionary with keys 'model', 'scalers', and 'feature_names'.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    x_scaler = model_data.get('scalers', {}).get('x', None)
    feature_names = model_data.get('feature_names', ['estimated_deaths_lag_1', 'estimated_deaths_lag_2'])
    return model, x_scaler, feature_names

def load_xgb_model(model_path):
    """
    Load the pre-trained XGBoost model (trained with log-transform).
    In the new script, the model data only includes:
      - 'model': the trained XGBoost model (predicting in log-space),
      - 'x_scaler': the scaler for the features,
      - 'feature_names': list of features.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    x_scaler = model_data['x_scaler']
    feature_names = model_data.get('feature_names', ['estimated_deaths_lag_1', 'estimated_deaths_lag_2'])
    return model, x_scaler, feature_names

def backcast_model(df_desc, n_train, model, feature_names, model_type="lasso", x_scaler=None):
    """
    Given a DataFrame sorted in descending order (later years first) and a number of seed rows (n_train)
    with known estimated_deaths, iteratively backcast for rows with missing values.
    
    For LASSO, the feature vector is built and prediction is made directly.
    For Log LASSO (model_type=="log_lasso"), the features are passed directly but the prediction
        is converted back via: y_pred = exp(predicted_log) - LOG_OFFSET
    For XGBoost (model_type=="xgb"), the features are scaled using x_scaler,
        and because the XGB model was trained on log(y+offset), the prediction is converted back via:
        y_pred = exp(predicted_log) - LOG_OFFSET
    """
    df = df_desc.copy()
    for i in range(n_train, len(df)):
        if pd.isna(df.loc[i, 'estimated_deaths']):
            features = {}
            # Build a feature dictionary using the expected feature names.
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
            # Create a DataFrame row with the proper column order.
            X_input = pd.DataFrame([features], columns=feature_names)
            # Fill any missing feature values with 0.
            if X_input.isnull().any().any():
                X_input = X_input.fillna(0)
            
            if model_type == "xgb":
                # For XGBoost, transform features with x_scaler
                X_scaled = x_scaler.transform(X_input)
                # Predict in log-space and invert transform
                pred_log = model.predict(X_scaled)[0]
                pred_value = np.exp(pred_log) - LOG_OFFSET
            elif model_type == "log_lasso":
                # For Log LASSO, scale features first, then predict in log-space and invert transform
                X_scaled = x_scaler.transform(X_input)
                pred_log = model.predict(X_scaled)[0]
                # Add safeguards against extremely large values
                if pred_log > 20:  # Cap at exp(20) to prevent overflow
                    print(f"Warning: Large log prediction at index {i}, year {df.loc[i, 'year']}: {pred_log}, capping at 20")
                    pred_log = 20
                pred_value = np.exp(pred_log) - LOG_OFFSET
                
                # Additional sanity check
                if pred_value > 10000:
                    print(f"Warning: Very large prediction at index {i}, year {df.loc[i, 'year']}: {pred_value}, capping at 10000")
                    pred_value = 10000
            else:  # regular lasso
                pred_value = model.predict(X_input)[0]
            
            # Update the DataFrame with the predicted value.
            df.loc[i, 'estimated_deaths'] = pred_value
    return df

def main():
    # File paths (adjust if needed)
    training_path = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\training_data_1900_1936.csv"
    pred_path = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\pred_data_1870_1899.csv"
    # Model paths (adjust if needed)
    lasso_model_path = r"C:\Users\benra\Documents\Newspaper\regression_finals\lasso_model.pkl"
    log_lasso_model_path = r"C:\Users\benra\Documents\Newspaper\regression_finals\lasso_log_model.pkl"
    xgb_model_path = r"C:\Users\benra\Documents\Newspaper\regression_finals\xgboost_model.pkl"
    
    # Load training and prediction data
    train_df = pd.read_csv(training_path)
    pred_df = pd.read_csv(pred_path)
    
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
    
    # Load models
    lasso_model, lasso_features = load_lasso_model(lasso_model_path)
    log_lasso_model, log_lasso_x_scaler, log_lasso_features = load_log_lasso_model(log_lasso_model_path)
    xgb_model, xgb_x_scaler, xgb_features = load_xgb_model(xgb_model_path)
    
    # Make copies of the descending DataFrame for each model's backcasting
    df_desc_lasso = combined_desc.copy()
    df_desc_log_lasso = combined_desc.copy()
    df_desc_xgb = combined_desc.copy()
    
    print("Backcasting with LASSO model...")
    df_lasso_pred = backcast_model(df_desc_lasso, n_train, lasso_model, lasso_features, model_type="lasso")
    
    print("Backcasting with Log LASSO model...")
    df_log_lasso_pred = backcast_model(df_desc_log_lasso, n_train, log_lasso_model, log_lasso_features, model_type="log_lasso", x_scaler=log_lasso_x_scaler)
    
    print("Backcasting with XGBoost model (log-transformed)...")
    df_xgb_pred = backcast_model(df_desc_xgb, n_train, xgb_model, xgb_features, model_type="xgb", x_scaler=xgb_x_scaler)
    
    # Sort back to ascending order by year
    final_lasso = df_lasso_pred.sort_values('year').reset_index(drop=True)
    final_log_lasso = df_log_lasso_pred.sort_values('year').reset_index(drop=True)
    final_xgb = df_xgb_pred.sort_values('year').reset_index(drop=True)
    
    # Extract only the backcast period (years before 1900)
    backcast_lasso = final_lasso[final_lasso['year'] < 1900]
    backcast_log_lasso = final_log_lasso[final_log_lasso['year'] < 1900]
    backcast_xgb = final_xgb[final_xgb['year'] < 1900]
    
    # Save predictions to CSV
    output_csv_lasso = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\backcast_predictions_lasso_1870_1899.csv"
    output_csv_log_lasso = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\backcast_predictions_log_lasso_1870_1899.csv"
    output_csv_xgb = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\backcast_predictions_xgb_1870_1899.csv"
    
    backcast_lasso.to_csv(output_csv_lasso, index=False)
    backcast_log_lasso.to_csv(output_csv_log_lasso, index=False)
    backcast_xgb.to_csv(output_csv_xgb, index=False)
    
    print(f"LASSO backcast predictions saved to {output_csv_lasso}")
    print(f"Log LASSO backcast predictions saved to {output_csv_log_lasso}")
    print(f"XGBoost backcast predictions saved to {output_csv_xgb}")
    
    # ---- Plot all three models' predictions on the same graph ----
    plt.figure(figsize=(12, 7))
    plt.plot(backcast_lasso['year'], backcast_lasso['estimated_deaths'], marker='o', linestyle='-', color='blue', label="LASSO Predictions")
    plt.plot(backcast_log_lasso['year'], backcast_log_lasso['estimated_deaths'], marker='x', linestyle='-.', color='green', label="Log LASSO Predictions")
    plt.plot(backcast_xgb['year'], backcast_xgb['estimated_deaths'], marker='s', linestyle='--', color='red', label="XGBoost Predictions")
    plt.xlabel("Year")
    plt.ylabel("Estimated Deaths")
    plt.title("Backcast Predictions (1870 - 1899)")
    plt.legend()
    plt.grid(True)
    
    output_plot = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\backcast_predictions_comparison.png"
    plt.savefig(output_plot)
    plt.show()
    print(f"Combined graph saved to {output_plot}")

    # ---- Plot comparison of LASSO and Log LASSO only ----
    plt.figure(figsize=(12, 7))
    plt.plot(backcast_lasso['year'], backcast_lasso['estimated_deaths'], marker='o', linestyle='-', color='blue', label="LASSO Predictions")
    plt.plot(backcast_log_lasso['year'], backcast_log_lasso['estimated_deaths'], marker='x', linestyle='-.', color='green', label="Log LASSO Predictions")
    plt.xlabel("Year")
    plt.ylabel("Estimated Deaths")
    plt.title("LASSO vs Log LASSO Backcast Predictions (1870 - 1899)")
    plt.legend()
    plt.grid(True)
    
    output_plot_lasso_comparison = r"C:\Users\benra\Documents\Newspaper\yearly_occurrence_data\backcast_predictions_lasso_comparison.png"
    plt.savefig(output_plot_lasso_comparison)
    print(f"LASSO comparison graph saved to {output_plot_lasso_comparison}")

if __name__ == "__main__":
    main()