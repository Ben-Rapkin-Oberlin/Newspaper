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

#############################
# Function: Run Validation Simulation
#############################
def run_validation_simulation(train_df, static_feature_list, n_leads=1, val_horizon=20):
    """
    For the given training DataFrame (which must include 'year' and 'estimated_deaths')
    and a list of static predictor column names, this function:
      1. Creates lead features.
      2. Prepares training data (and scales it where needed).
      3. Trains six models:
         - Linear Regression (with clipping to force nonnegative predictions)
         - Random Forest
         - XGBoost
         - Support Vector Regression (SVR)
         - Gaussian Process Regression (GPR)
         - A GAM (trained on a log-transformed target so that back-transformation yields nonnegative predictions)
      4. Runs a recursive backcasting simulation on the first `val_horizon` years.
      5. Computes MAE for each model and for an ensemble (weighted inversely by MAE).
      6. Computes R² for Linear Regression, XGBoost, and GPR (and for the ensemble).
      7. Returns a dictionary containing these metrics and the prediction series.
    """
    # --- 1. Create Lead Features ---
    for i in range(1, n_leads + 1):
        train_df[f'lead_{i}'] = train_df['estimated_deaths'].shift(-i)
    train_df = train_df.dropna().reset_index(drop=True)

    # --- 2. Prepare Training Data ---
    X_train = train_df[static_feature_list]
    y_train = train_df['estimated_deaths']
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # --- 3. Train Models ---
    # Linear Regression (scaled)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    
    # Random Forest (unscaled)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # XGBoost (unscaled)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # SVR (scaled)
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train_scaled)
    
    # GPR (scaled)
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
    gpr_model.fit(X_train_scaled, y_train_scaled)
    
    # GAM (scaled) with log-transformed target
    y_train_log = np.log(y_train + 1)
    terms = s(0)
    for i in range(1, X_train_scaled.shape[1]):
        terms += s(i)
    gam_model = LinearGAM(terms)
    gam_model.gridsearch(X_train_scaled, y_train_log)
    
    # --- 4. Validation Simulation (Recursive Backcasting) ---
    train_orig = train_df.copy()
    train_orig.set_index("year", inplace=True)
    val_years = sorted(train_orig.index)[:val_horizon]
    
    lead_years_val = list(train_orig.index[val_horizon:val_horizon+n_leads])
    lead_window_val = [train_orig.loc[yr, "estimated_deaths"] for yr in lead_years_val]
    
    preds_lr_val = {}
    preds_rf_val = {}
    preds_xgb_val = {}
    preds_svr_val = {}
    preds_gpr_val = {}
    preds_gam_val = {}
    
    for year in sorted(val_years, reverse=True):
        static_vals = train_orig.loc[[year], static_feature_list]
        X_input_df = static_vals.copy()
        if X_input_df.isnull().any().any():
            X_input_df = X_input_df.fillna(0)
        X_input_scaled = scaler_X.transform(X_input_df)
        
        # LR prediction (clip negatives)
        pred_lr_scaled = lr_model.predict(X_input_scaled)
        pred_lr_val = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
        pred_lr_val = max(0, pred_lr_val)
        
        pred_rf_val = rf_model.predict(X_input_df)[0]
        pred_xgb_val = xgb_model.predict(X_input_df)[0]
        
        pred_svr_scaled = svr_model.predict(X_input_scaled)
        pred_svr_val = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
        
        pred_gpr_scaled = gpr_model.predict(X_input_scaled)
        pred_gpr_val = scaler_y.inverse_transform(pred_gpr_scaled.reshape(-1, 1)).flatten()[0]
        
        pred_gam_log = gam_model.predict(X_input_scaled)
        pred_gam_val = np.exp(pred_gam_log) - 1
        
        preds_lr_val[year] = pred_lr_val
        preds_rf_val[year] = pred_rf_val
        preds_xgb_val[year] = pred_xgb_val
        preds_svr_val[year] = pred_svr_val
        preds_gpr_val[year] = pred_gpr_val
        preds_gam_val[year] = pred_gam_val
        
        lead_window_val = [pred_lr_val] + lead_window_val[:-1]
    
    preds_lr_val_series = pd.Series({yr: preds_lr_val[yr] for yr in sorted(preds_lr_val.keys())})
    preds_rf_val_series = pd.Series({yr: preds_rf_val[yr] for yr in sorted(preds_rf_val.keys())})
    preds_xgb_val_series = pd.Series({yr: preds_xgb_val[yr] for yr in sorted(preds_xgb_val.keys())})
    preds_svr_val_series = pd.Series({yr: preds_svr_val[yr] for yr in sorted(preds_svr_val.keys())})
    preds_gpr_val_series = pd.Series({yr: preds_gpr_val[yr] for yr in sorted(preds_gpr_val.keys())})
    preds_gam_val_series = pd.Series({yr: preds_gam_val[yr] for yr in sorted(preds_gam_val.keys())})
    
    actual_val = train_orig.loc[preds_lr_val_series.index, "estimated_deaths"]
    
    mae_lr  = mean_absolute_error(actual_val, preds_lr_val_series)
    mae_rf  = mean_absolute_error(actual_val, preds_rf_val_series)
    mae_xgb = mean_absolute_error(actual_val, preds_xgb_val_series)
    mae_svr = mean_absolute_error(actual_val, preds_svr_val_series)
    mae_gpr = mean_absolute_error(actual_val, preds_gpr_val_series)
    mae_gam = mean_absolute_error(actual_val, preds_gam_val_series)
    
    # Ensemble weights (using all six models)
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
    
    ensemble_val_series = (w_lr  * preds_lr_val_series +
                           w_rf  * preds_rf_val_series +
                           w_xgb * preds_xgb_val_series +
                           w_svr * preds_svr_val_series +
                           w_gpr * preds_gpr_val_series +
                           w_gam * preds_gam_val_series)
    
    mae_ensemble = mean_absolute_error(actual_val, ensemble_val_series)
    
    # Compute R² for LR, XGBoost, and GPR, and also for the ensemble.
    r2_lr  = r2_score(actual_val, preds_lr_val_series)
    r2_xgb = r2_score(actual_val, preds_xgb_val_series)
    r2_gpr = r2_score(actual_val, preds_gpr_val_series)
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

#############################
# MAIN SCRIPT
#############################
# Load training data.
#train_path = r"/usr/users/quota/students/2021/brapkin/Newspaper/yearly_occurrence_data/training_data_1900_1936.csv"
train_path = r"yearly_occurrence_data/training_data_1900_1936.csv"

train_df = pd.read_csv(train_path)
train_df = train_df.sort_values("year").reset_index(drop=True)

# Identify all static predictors (exclude 'year', 'estimated_deaths', and any lead columns).
static_cols = [col for col in train_df.columns if col not in ["year", "estimated_deaths"] and not col.startswith("lead_")]

# --- Define Feature Groups ---
# Group 1: All features whose names contain smallpox ratios.
group1_features = [col for col in static_cols if ("smallpox_sick_cooccurrences_ratio" in col) or 
                                                  ("smallpox_epidemic_cooccurrences_ratio" in col)]
# Group 2 (unused): All other features.
group2_features = [col for col in static_cols if col not in group1_features]

# Group 3: Top n most important features from Group 2, where n is the number of features in Group 1.
n = len(group1_features)
if n > 0 and len(group2_features) > 0:
    rf_importance = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_importance.fit(train_df[group2_features], train_df['estimated_deaths'])
    importances = rf_importance.feature_importances_
    imp_df = pd.DataFrame({"feature": group2_features, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    group3_features = imp_df["feature"].tolist()[:n]
else:
    group3_features = []

print("Group 1 features (smallpox ratios):", group1_features)
print("Group 3 features (top {} non-smallpox features):".format(n), group3_features)

# Run validation simulation for Group 1 and Group 3.
results_group1 = run_validation_simulation(train_df.copy(), group1_features, n_leads=1, val_horizon=20)
results_group3 = run_validation_simulation(train_df.copy(), group3_features, n_leads=1, val_horizon=20)

# Print MAE and R² results.
print("\n--- Group 1 (smallpox ratios) ---")
print("Ensemble: MAE = {:.3f}, R² = {:.3f}".format(results_group1["mae_ensemble"], results_group1["r2_ensemble"]))
print("Linear Regression: MAE = {:.3f}, R² = {:.3f}".format(results_group1["mae_lr"], results_group1["r2_lr"]))
print("XGBoost:           MAE = {:.3f}, R² = {:.3f}".format(results_group1["mae_xgb"], results_group1["r2_xgb"]))
print("GPR:               MAE = {:.3f}, R² = {:.3f}".format(results_group1["mae_gpr"], results_group1["r2_gpr"]))

print("\n--- Group 3 (top n non-smallpox features) ---")
print("Ensemble: MAE = {:.3f}, R² = {:.3f}".format(results_group3["mae_ensemble"], results_group3["r2_ensemble"]))
print("Linear Regression: MAE = {:.3f}, R² = {:.3f}".format(results_group3["mae_lr"], results_group3["r2_lr"]))
print("XGBoost:           MAE = {:.3f}, R² = {:.3f}".format(results_group3["mae_xgb"], results_group3["r2_xgb"]))
print("GPR:               MAE = {:.3f}, R² = {:.3f}".format(results_group3["mae_gpr"], results_group3["r2_gpr"]))

#############################
# TIME SERIES PREDICTION GRAPHS (Group 1 and Group 3)
#############################
plt.figure(figsize=(14, 6))

# Group 1: Smallpox ratios
plt.subplot(1, 2, 1)
plt.plot(results_group1["actual_series"].index, results_group1["actual_series"].values, 
         label="Actual", marker='o', color='black')
plt.plot(results_group1["ensemble_series"].index, results_group1["ensemble_series"].values, 
         label="Ensemble", marker='s', color='blue')
plt.plot(results_group1["preds_lr_val_series"].index, results_group1["preds_lr_val_series"].values, 
         label="Linear Regression", marker='x', color='red')
plt.plot(results_group1["preds_xgb_val_series"].index, results_group1["preds_xgb_val_series"].values, 
         label="XGBoost", marker='^', color='green')
plt.plot(results_group1["preds_gpr_val_series"].index, results_group1["preds_gpr_val_series"].values, 
         label="GPR", marker='D', color='purple')
plt.title("Group 1: Smallpox Ratios")
plt.xlabel("Year")
plt.ylabel("Estimated Deaths")
plt.legend()
plt.grid(True)

# Group 3: Top n non‑smallpox features
plt.subplot(1, 2, 2)
plt.plot(results_group3["actual_series"].index, results_group3["actual_series"].values, 
         label="Actual", marker='o', color='black')
plt.plot(results_group3["ensemble_series"].index, results_group3["ensemble_series"].values, 
         label="Ensemble", marker='s', color='blue')
plt.plot(results_group3["preds_lr_val_series"].index, results_group3["preds_lr_val_series"].values, 
         label="Linear Regression", marker='x', color='red')
plt.plot(results_group3["preds_xgb_val_series"].index, results_group3["preds_xgb_val_series"].values, 
         label="XGBoost", marker='^', color='green')
plt.plot(results_group3["preds_gpr_val_series"].index, results_group3["preds_gpr_val_series"].values, 
         label="GPR", marker='D', color='purple')
plt.title("Group 3: Top {} Non‑Smallpox Features".format(n))
plt.xlabel("Year")
plt.ylabel("Estimated Deaths")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#############################
# BAR CHARTS FOR MAE and R²
#############################
# Prepare data for bar charts.
models = ["LR", "XGBoost", "GPR", "Ensemble"]

# MAE values
mae_group1 = [results_group1["mae_lr"], results_group1["mae_xgb"], results_group1["mae_gpr"], results_group1["mae_ensemble"]]
mae_group3 = [results_group3["mae_lr"], results_group3["mae_xgb"], results_group3["mae_gpr"], results_group3["mae_ensemble"]]

# R² values
r2_group1 = [results_group1["r2_lr"], results_group1["r2_xgb"], results_group1["r2_gpr"], results_group1["r2_ensemble"]]
r2_group3 = [results_group3["r2_lr"], results_group3["r2_xgb"], results_group3["r2_gpr"], results_group3["r2_ensemble"]]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# MAE bar chart
axs[0].bar(x - width/2, mae_group1, width, label="Group 1", color='skyblue')
axs[0].bar(x + width/2, mae_group3, width, label="Group 3", color='salmon')
axs[0].set_ylabel("MAE")
axs[0].set_title("MAE by Model and Feature Group")
axs[0].set_xticks(x)
axs[0].set_xticklabels(models)
axs[0].legend()
axs[0].grid(axis='y')

# R² bar chart
axs[1].bar(x - width/2, r2_group1, width, label="Group 1", color='skyblue')
axs[1].bar(x + width/2, r2_group3, width, label="Group 3", color='salmon')
axs[1].set_ylabel("R²")
axs[1].set_title("R² by Model and Feature Group")
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].legend()
axs[1].grid(axis='y')

plt.tight_layout()
plt.show()
