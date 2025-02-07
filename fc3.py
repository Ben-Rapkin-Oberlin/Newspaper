import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from pygam import LinearGAM, s

###############################################################################
# Function: Run Backcasting Simulation (with lead features)
###############################################################################
def run_backcast_simulation(full_df, static_feature_list, n_leads=1, val_horizon=20):
    """
    Splits the full dataset (which must contain a 'year' column) into:
      - A validation set: the earliest `val_horizon` years.
      - A training set: all later years.
    
    The models are trained on the training set using both the static predictors and 
    lead features. Then, in a recursive backcasting simulation, the models produce 
    predictions on the validation set (in reverse chronological order) by combining 
    static predictors with an updated lead window.
    
    Returns a dictionary containing MAE, R², and prediction series.
    """
    # --- 0. Sort the Data & Split ---
    df_sorted = full_df.sort_values("year").reset_index(drop=True)
    
    if len(df_sorted) < (val_horizon + n_leads):
        raise ValueError("Not enough data to have the specified validation horizon and lead window.")
    
    df_val = df_sorted.iloc[:val_horizon].copy()   # validation set: earliest years
    df_train = df_sorted.iloc[val_horizon:].copy()   # training set: later years
    
    max_threshold = df_train['estimated_deaths'].max() * 2
    
    # --- 1. Prepare Training Data (Create Lead Features) ---
    for i in range(1, n_leads + 1):
        df_train[f'lead_{i}'] = df_train['estimated_deaths'].shift(-i)
    df_train = df_train.dropna().reset_index(drop=True)
    
    # Combine static features and lead features into one list.
    lead_cols = [f'lead_{i}' for i in range(1, n_leads+1)]
    ml_features = static_feature_list + lead_cols
    
    X_train = df_train[ml_features]
    y_train = df_train['estimated_deaths']
    
    # --- 2. Scale Data ---
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # --- 3. Train Models ---
    # Linear Regression (trained on scaled features/target)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    
    # Random Forest (trained on unscaled features)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # XGBoost (trained on unscaled features)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # SVR (trained on scaled features)
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train_scaled)
    
    # Gaussian Process Regression (trained on scaled features)
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
    gpr_model.fit(X_train_scaled, y_train_scaled)
    
    # GAM (trained on scaled features with log-transformed target)
    y_train_log = np.log(y_train + 1)
    terms = s(0)
    for i in range(1, X_train_scaled.shape[1]):
        terms += s(i)
    gam_model = LinearGAM(terms)
    gam_model.gridsearch(X_train_scaled, y_train_log)
    
    # --- 4. Backcasting Simulation Setup ---
    # Initialize the lead window using the first n_leads rows of the training set.
    df_train_sorted = df_train.sort_values("year").reset_index(drop=True)
    lead_window_val = df_train_sorted.iloc[:n_leads]['estimated_deaths'].tolist()
    
    # Set the validation set index to 'year'
    df_val = df_val.set_index("year")
    val_years = sorted(df_val.index)
    
    # Dictionaries to store validation predictions.
    preds_lr_val  = {}
    preds_rf_val  = {}
    preds_xgb_val = {}
    preds_svr_val = {}
    preds_gpr_val = {}
    preds_gam_val = {}
    
    # --- 5. Recursive Backcasting ---
    for year in sorted(val_years, reverse=True):
        # Get static predictors for this year.
        static_vals = df_val.loc[[year], static_feature_list].fillna(0)
        
        # Create DataFrame for lead features from the current lead window.
        current_leads_df = pd.DataFrame(
            np.array(lead_window_val).reshape(1, -1),
            columns=lead_cols,
            index=static_vals.index
        )
        
        # Combine static and lead features.
        X_input_df = pd.concat([static_vals, current_leads_df], axis=1)[ml_features]
        if X_input_df.isnull().any().any():
            X_input_df = X_input_df.fillna(0)
        
        X_input_scaled = scaler_X.transform(X_input_df)
        
        # --- Model Predictions ---
        # Linear Regression
        pred_lr_scaled = lr_model.predict(X_input_scaled)
        pred_lr_val = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
        pred_lr_val = np.clip(pred_lr_val, 0, max_threshold)
        
        # Random Forest
        pred_rf_val = rf_model.predict(X_input_df)[0]
        pred_rf_val = np.clip(pred_rf_val, 0, max_threshold)
        
        # XGBoost
        pred_xgb_val = xgb_model.predict(X_input_df)[0]
        pred_xgb_val = np.clip(pred_xgb_val, 0, max_threshold)
        
        # SVR
        pred_svr_scaled = svr_model.predict(X_input_scaled)
        pred_svr_val = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
        pred_svr_val = np.clip(pred_svr_val, 0, max_threshold)
        
        # Gaussian Process Regression
        pred_gpr_scaled = gpr_model.predict(X_input_scaled)
        pred_gpr_val = scaler_y.inverse_transform(pred_gpr_scaled.reshape(-1, 1)).flatten()[0]
        pred_gpr_val = np.clip(pred_gpr_val, 0, max_threshold)
        
        # GAM
        pred_gam_log = gam_model.predict(X_input_scaled)
        pred_gam_val = np.exp(pred_gam_log) - 1
        pred_gam_val = np.clip(pred_gam_val, 0, max_threshold)
        
        # Save predictions.
        preds_lr_val[year]  = pred_lr_val
        preds_rf_val[year]  = pred_rf_val
        preds_xgb_val[year] = pred_xgb_val
        preds_svr_val[year] = pred_svr_val
        preds_gpr_val[year] = pred_gpr_val
        preds_gam_val[year] = pred_gam_val
        
        # Update the lead window using the (clipped) LR prediction.
        lead_window_val = [pred_lr_val] + lead_window_val[:-1]
    
    # Convert prediction dictionaries to Series.
    preds_lr_val_series  = pd.Series({yr: preds_lr_val[yr] for yr in sorted(preds_lr_val.keys())})
    preds_rf_val_series  = pd.Series({yr: preds_rf_val[yr] for yr in sorted(preds_rf_val.keys())})
    preds_xgb_val_series = pd.Series({yr: preds_xgb_val[yr] for yr in sorted(preds_xgb_val.keys())})
    preds_svr_val_series = pd.Series({yr: preds_svr_val[yr] for yr in sorted(preds_svr_val.keys())})
    preds_gpr_val_series = pd.Series({yr: preds_gpr_val[yr] for yr in sorted(preds_gpr_val.keys())})
    preds_gam_val_series = pd.Series({yr: preds_gam_val[yr] for yr in sorted(preds_gam_val.keys())})
    
    actual_val = df_val.loc[preds_lr_val_series.index, "estimated_deaths"]
    
    # --- 6. Compute Metrics and Ensemble ---
    mae_lr   = mean_absolute_error(actual_val, preds_lr_val_series)
    mae_rf   = mean_absolute_error(actual_val, preds_rf_val_series)
    mae_xgb  = mean_absolute_error(actual_val, preds_xgb_val_series)
    mae_svr  = mean_absolute_error(actual_val, preds_svr_val_series)
    mae_gpr  = mean_absolute_error(actual_val, preds_gpr_val_series)
    mae_gam  = mean_absolute_error(actual_val, preds_gam_val_series)
    
    epsilon = 1e-6  # to avoid division by zero
    inv_lr   = 1 / (mae_lr   + epsilon)
    inv_rf   = 1 / (mae_rf   + epsilon)
    inv_xgb  = 1 / (mae_xgb  + epsilon)
    inv_svr  = 1 / (mae_svr  + epsilon)
    inv_gpr  = 1 / (mae_gpr  + epsilon)
    inv_gam  = 1 / (mae_gam  + epsilon)
    total_inv = inv_lr + inv_rf + inv_xgb + inv_svr + inv_gpr + inv_gam
    
    w_lr   = inv_lr   / total_inv
    w_rf   = inv_rf   / total_inv
    w_xgb  = inv_xgb  / total_inv
    w_svr  = inv_svr  / total_inv
    w_gpr  = inv_gpr  / total_inv
    w_gam  = inv_gam  / total_inv
    
    ensemble_val_series = (w_lr   * preds_lr_val_series +
                           w_rf   * preds_rf_val_series +
                           w_xgb  * preds_xgb_val_series +
                           w_svr  * preds_svr_val_series +
                           w_gpr  * preds_gpr_val_series +
                           w_gam  * preds_gam_val_series)
    
    mae_ensemble = mean_absolute_error(actual_val, ensemble_val_series)
    r2_lr   = r2_score(actual_val, preds_lr_val_series)
    r2_xgb  = r2_score(actual_val, preds_xgb_val_series)
    r2_gpr  = r2_score(actual_val, preds_gpr_val_series)
    r2_ensemble = r2_score(actual_val, ensemble_val_series)
    
    results = {
        "mae_lr": mae_lr,
        "mae_rf": mae_rf,
        "mae_xgb": mae_xgb,
        "mae_svr": mae_svr,
        "mae_gpr": mae_gpr,
        "mae_gam": mae_gam,
        "mae_ensemble": mae_ensemble,
        "r2_lr": r2_lr,
        "r2_xgb": r2_xgb,
        "r2_gpr": r2_gpr,
        "r2_ensemble": r2_ensemble,
        "ensemble_series": ensemble_val_series,
        "actual_series": actual_val,
        "preds_lr_val_series": preds_lr_val_series,
        "preds_xgb_val_series": preds_xgb_val_series,
        "preds_gpr_val_series": preds_gpr_val_series
    }
    return results

###############################################################################
# Function: Run In-Sample Fit Simulation (Train on Entire Dataset with lead features)
###############################################################################
def run_full_fit_simulation(full_df, static_feature_list, n_leads=1):
    """
    Trains models on the entire dataset (after creating lead features) and then
    produces in-sample predictions for all available years (i.e. the fitted values).
    Returns a dictionary with predictions, the actual target, and performance metrics.
    """
    # --- 0. Sort Data and Create Lead Features ---
    df_sorted = full_df.sort_values("year").reset_index(drop=True)
    for i in range(1, n_leads + 1):
        df_sorted[f'lead_{i}'] = df_sorted['estimated_deaths'].shift(-i)
    df_model = df_sorted.dropna().reset_index(drop=True)
    
    lead_cols = [f'lead_{i}' for i in range(1, n_leads+1)]
    ml_features = static_feature_list + lead_cols
    
    # --- 1. Prepare Data ---
    X = df_model[ml_features]
    y = df_model['estimated_deaths']
    
    max_threshold = y.max() * 2
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # --- 2. Train Models on the Entire Dataset ---
    lr_model = LinearRegression()
    lr_model.fit(X_scaled, y_scaled)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_scaled, y_scaled)
    
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
    gpr_model.fit(X_scaled, y_scaled)
    
    y_log = np.log(y + 1)
    terms = s(0)
    for i in range(1, X_scaled.shape[1]):
        terms += s(i)
    gam_model = LinearGAM(terms)
    gam_model.gridsearch(X_scaled, y_log)
    
    # --- 3. Compute In-Sample Predictions ---
    preds_lr_scaled = lr_model.predict(X_scaled)
    preds_lr = scaler_y.inverse_transform(preds_lr_scaled.reshape(-1, 1)).flatten()
    preds_lr = np.clip(preds_lr, 0, max_threshold)
    
    preds_rf = rf_model.predict(X)
    preds_rf = np.clip(preds_rf, 0, max_threshold)
    
    preds_xgb = xgb_model.predict(X)
    preds_xgb = np.clip(preds_xgb, 0, max_threshold)
    
    preds_svr_scaled = svr_model.predict(X_scaled)
    preds_svr = scaler_y.inverse_transform(preds_svr_scaled.reshape(-1, 1)).flatten()
    preds_svr = np.clip(preds_svr, 0, max_threshold)
    
    preds_gpr_scaled = gpr_model.predict(X_scaled)
    preds_gpr = scaler_y.inverse_transform(preds_gpr_scaled.reshape(-1, 1)).flatten()
    preds_gpr = np.clip(preds_gpr, 0, max_threshold)
    
    preds_gam_log = gam_model.predict(X_scaled)
    preds_gam = np.exp(preds_gam_log) - 1
    preds_gam = np.clip(preds_gam, 0, max_threshold)
    
    # --- 4. Create Ensemble Prediction (Weights computed on in-sample MAE) ---
    mae_lr  = mean_absolute_error(y, preds_lr)
    mae_rf  = mean_absolute_error(y, preds_rf)
    mae_xgb = mean_absolute_error(y, preds_xgb)
    mae_svr = mean_absolute_error(y, preds_svr)
    mae_gpr = mean_absolute_error(y, preds_gpr)
    mae_gam = mean_absolute_error(y, preds_gam)
    
    epsilon = 1e-6
    inv_lr  = 1 / (mae_lr  + epsilon)
    inv_rf  = 1 / (mae_rf  + epsilon)
    inv_xgb = 1 / (mae_xgb + epsilon)
    inv_svr = 1 / (mae_svr + epsilon)
    inv_gpr = 1 / (mae_gpr + epsilon)
    inv_gam = 1 / (mae_gam + epsilon)
    total_inv = inv_lr + inv_rf + inv_xgb + inv_svr + inv_gpr + inv_gam
    
    w_lr  = inv_lr  / total_inv
    w_rf  = inv_rf  / total_inv
    w_xgb = inv_xgb / total_inv
    w_svr = inv_svr / total_inv
    w_gpr = inv_gpr / total_inv
    w_gam = inv_gam / total_inv
    
    ensemble = (w_lr * preds_lr + w_rf * preds_rf + w_xgb * preds_xgb +
                w_svr * preds_svr + w_gpr * preds_gpr + w_gam * preds_gam)
    
    # --- 5. Compute Performance Metrics ---
    mae_ensemble = mean_absolute_error(y, ensemble)
    r2_ensemble = r2_score(y, ensemble)
    r2_lr = r2_score(y, preds_lr)
    
    results = {
        "years": df_model["year"] if "year" in df_model.columns else df_sorted["year"].iloc[:-n_leads],
        "actual": y,
        "preds_lr": preds_lr,
        "ensemble": ensemble,
        "mae_lr": mae_lr,
        "r2_lr": r2_lr,
        "mae_ensemble": mae_ensemble,
        "r2_ensemble": r2_ensemble
    }
    return results

###############################################################################
# MAIN SCRIPT
###############################################################################
# Load training data.
train_path = r"yearly_occurrence_data/training_data_1900_1936.csv"
train_df = pd.read_csv(train_path)
train_df = train_df.sort_values("year").reset_index(drop=True)

# Identify all static predictors (exclude 'year', 'estimated_deaths', and any lead columns).
static_cols = [col for col in train_df.columns if col not in ["year", "estimated_deaths"] and not col.startswith("lead_")]

# Limit the maximum number of features used in some groups.
max_features_to_use = 8

# --- Define Predictor Groups with Intuitive Names ---
# Group A: Smallpox Co‑occurrence Ratios.
groupA = [col for col in static_cols 
          if col.startswith("smallpox_") and "cooccurrences_ratio" in col]
if max_features_to_use is not None and len(groupA) > max_features_to_use:
    rf_importance_A = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_importance_A.fit(train_df[groupA], train_df['estimated_deaths'])
    importances_A = rf_importance_A.feature_importances_
    imp_df_A = pd.DataFrame({"feature": groupA, "importance": importances_A})
    imp_df_A = imp_df_A.sort_values("importance", ascending=False)
    groupA = imp_df_A["feature"].tolist()[:max_features_to_use]

# Let n be the number of features in Group A.
n = len(groupA)

# Group B: Top Overall Features (from all predictors).
rf_importance_all = RandomForestRegressor(n_estimators=100, random_state=42)
rf_importance_all.fit(train_df[static_cols], train_df['estimated_deaths'])
importances_all = rf_importance_all.feature_importances_
imp_df_all = pd.DataFrame({"feature": static_cols, "importance": importances_all})
imp_df_all = imp_df_all.sort_values("importance", ascending=False)
groupB = imp_df_all["feature"].tolist()[:n]

# Group C: Top Non‑Smallpox Features.
non_smallpox_candidates = [col for col in static_cols 
                           if not (col.startswith("smallpox_") and ("cooccurrence" in col or "cooccurence" in col))]
rf_importance_non_smallpox = RandomForestRegressor(n_estimators=100, random_state=42)
rf_importance_non_smallpox.fit(train_df[non_smallpox_candidates], train_df['estimated_deaths'])
importances_non_smallpox = rf_importance_non_smallpox.feature_importances_
imp_df_non_smallpox = pd.DataFrame({"feature": non_smallpox_candidates, "importance": importances_non_smallpox})
imp_df_non_smallpox = imp_df_non_smallpox.sort_values("importance", ascending=False)
groupC = imp_df_non_smallpox["feature"].tolist()[:n]

# Assign intuitive names.
groupA_name = "Smallpox Co‑occurrence Ratios"
groupB_name = "Top Overall Features"       # (will appear rightmost)
groupC_name = "Top Non‑Smallpox Features"

print(f"{groupA_name}: {len(groupA)} features")
print("Features:", groupA)
print(f"{groupC_name}: {len(groupC)} features")
print("Features:", groupC)
print(f"{groupB_name}: {len(groupB)} features")
print("Features:", groupB)

# --- Run Backcasting Simulation for All Three Groups (for out-of-sample performance) ---
results_A = run_backcast_simulation(train_df.copy(), groupA, n_leads=1, val_horizon=20)
results_B = run_backcast_simulation(train_df.copy(), groupB, n_leads=1, val_horizon=20)
results_C = run_backcast_simulation(train_df.copy(), groupC, n_leads=1, val_horizon=20)

# Print out-of-sample performance metrics.
print("\n--- Out-of-Sample Performance Metrics (Backcasting) ---")
print(f"{groupA_name} - Ensemble: MAE = {results_A['mae_ensemble']:.3f}, R² = {results_A['r2_ensemble']:.3f}")
print(f"{groupA_name} - Linear Regression: MAE = {results_A['mae_lr']:.3f}, R² = {results_A['r2_lr']:.3f}")

print(f"{groupC_name} - Ensemble: MAE = {results_C['mae_ensemble']:.3f}, R² = {results_C['r2_ensemble']:.3f}")
print(f"{groupC_name} - Linear Regression: MAE = {results_C['mae_lr']:.3f}, R² = {results_C['r2_lr']:.3f}")

print(f"{groupB_name} - Ensemble: MAE = {results_B['mae_ensemble']:.3f}, R² = {results_B['r2_ensemble']:.3f}")
print(f"{groupB_name} - Linear Regression: MAE = {results_B['mae_lr']:.3f}, R² = {results_B['r2_lr']:.3f}")

###############################################################################
# In-Sample Fit: Train on Entire Dataset and Evaluate Model Fit
###############################################################################
fit_A = run_full_fit_simulation(train_df.copy(), groupA, n_leads=1)
fit_B = run_full_fit_simulation(train_df.copy(), groupB, n_leads=1)
fit_C = run_full_fit_simulation(train_df.copy(), groupC, n_leads=1)

# Print in-sample (entire dataset) performance metrics for the linear models.
print("\n--- In-Sample Fit Metrics (Trained on Entire Dataset) ---")
print(f"{groupA_name} - Linear Regression: MAE = {fit_A['mae_lr']:.3f}, R² = {fit_A['r2_lr']:.3f}")
print(f"{groupA_name} - Ensemble: MAE = {fit_A['mae_ensemble']:.3f}, R² = {fit_A['r2_ensemble']:.3f}")

print(f"{groupC_name} - Linear Regression: MAE = {fit_C['mae_lr']:.3f}, R² = {fit_C['r2_lr']:.3f}")
print(f"{groupC_name} - Ensemble: MAE = {fit_C['mae_ensemble']:.3f}, R² = {fit_C['r2_ensemble']:.3f}")

print(f"{groupB_name} - Linear Regression: MAE = {fit_B['mae_lr']:.3f}, R² = {fit_B['r2_lr']:.3f}")
print(f"{groupB_name} - Ensemble: MAE = {fit_B['mae_ensemble']:.3f}, R² = {fit_B['r2_ensemble']:.3f}")

###############################################################################
# Plot In-Sample Model Fit (Actual vs. Predictions) for Each Predictor Group
###############################################################################
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# For clarity, we plot:
#   - The actual series (black solid line)
#   - The ensemble prediction (blue dashed line)
#   - The linear regression prediction (red dotted line)

# Group A: Smallpox Co‑occurrence Ratios.
axs[0].plot(fit_A["years"], fit_A["actual"], label="Actual", marker='o', color='black')
axs[0].plot(fit_A["years"], fit_A["ensemble"], label="Ensemble", linestyle='--', color='blue')
axs[0].plot(fit_A["years"], fit_A["preds_lr"], label="Linear Regression", linestyle=':', color='red')
axs[0].set_title(groupA_name)
axs[0].set_xlabel("Year")
axs[0].set_ylabel("Estimated Deaths")
axs[0].legend()
axs[0].grid(True)

# Group C: Top Non‑Smallpox Features.
axs[1].plot(fit_C["years"], fit_C["actual"], label="Actual", marker='o', color='black')
axs[1].plot(fit_C["years"], fit_C["ensemble"], label="Ensemble", linestyle='--', color='blue')
axs[1].plot(fit_C["years"], fit_C["preds_lr"], label="Linear Regression", linestyle=':', color='red')
axs[1].set_title(groupC_name)
axs[1].set_xlabel("Year")
axs[1].set_ylabel("Estimated Deaths")
axs[1].legend()
axs[1].grid(True)

# Group B: Top Overall Features.
axs[2].plot(fit_B["years"], fit_B["actual"], label="Actual", marker='o', color='black')
axs[2].plot(fit_B["years"], fit_B["ensemble"], label="Ensemble", linestyle='--', color='blue')
axs[2].plot(fit_B["years"], fit_B["preds_lr"], label="Linear Regression", linestyle=':', color='red')
axs[2].set_title(groupB_name)
axs[2].set_xlabel("Year")
axs[2].set_ylabel("Estimated Deaths")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
