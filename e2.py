import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ---------------------------
# 1. IMPORT MODELS & SETUP
# ---------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# --- New Model Imports ---
# Gaussian Process Regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Generalized Additive Model (GAM) using pyGAM
from pygam import LinearGAM, s

# ---------------------------
# User-Defined Settings
# ---------------------------
n_leads = 1  # Number of lead features to use

# ---------------------------
# 2. TRAINING: CREATE LEAD FEATURES & LOAD TRAINING DATA
# ---------------------------
train_path = r"yearly_occurrence_data/training_data_1900_1936.csv"
train_df = pd.read_csv(train_path)
train_df = train_df.sort_values("year").reset_index(drop=True)

# Define a maximum threshold based on training target.
max_threshold = train_df['estimated_deaths'].max() * 2

# Create lead features – these are the “future” values.
for i in range(1, n_leads + 1):
    train_df[f'lead_{i}'] = train_df['estimated_deaths'].shift(-i)
train_df = train_df.dropna().reset_index(drop=True)

# ---------------------------
# 2a. Determine Top Static Features
# ---------------------------
cols_for_importance = [
    col for col in train_df.columns 
    if col not in ["year", "estimated_deaths"] + [f'lead_{i}' for i in range(1, n_leads+1)]
]

rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(train_df[cols_for_importance], train_df['estimated_deaths'])
importances = rf_full.feature_importances_
feature_names = np.array(cols_for_importance)

# Select up to 8 top features (or all if fewer available)
top_n = min(8, len(feature_names))
indices = np.argsort(importances)[::-1][:top_n]
top_features = feature_names[indices]
print("Top {} features from original predictors: {}".format(top_n, top_features))

# The features for our models: top static predictors + lead features.
ml_features = list(top_features) + [f'lead_{i}' for i in range(1, n_leads+1)]

# ---------------------------
# 2b. Prepare Training Data for ML Models
# ---------------------------
X_train_ml = train_df[ml_features]
y_train_ml = train_df['estimated_deaths']  # target variable

# Scale features (for models that require scaling)
scaler_X = StandardScaler()
X_train_ml_scaled = scaler_X.fit_transform(X_train_ml)

# For models such as LR, SVR, and GPR we scale the target as well.
scaler_y = StandardScaler()
y_train_ml_scaled = scaler_y.fit_transform(y_train_ml.values.reshape(-1, 1)).flatten()

# ---------------------------
# 2c. Train the ML Models on the Entire Training Data
# ---------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_ml_scaled, y_train_ml_scaled)

# Random Forest (trained on unscaled features/target)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_ml, y_train_ml)

# XGBoost (trained on unscaled features/target)
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_ml, y_train_ml)

# Support Vector Regression
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_ml_scaled, y_train_ml_scaled)

# Gaussian Process Regression (GPR)
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5)
gpr_model.fit(X_train_ml_scaled, y_train_ml_scaled)

# --- GAM using a log-transformation ---
# Since estimated_deaths are counts, we transform the target using log(y+1)
y_train_log = np.log(y_train_ml + 1)
# Construct smoothing terms for all features.
terms = s(0)
for i in range(1, X_train_ml_scaled.shape[1]):
    terms += s(i)
gam_model = LinearGAM(terms)
# Optimize hyperparameters with grid search using the log-transformed target.
gam_model.gridsearch(X_train_ml_scaled, y_train_log)

# ---------------------------
# 2d. Compute and Print Training R² Values
# ---------------------------
pred_lr_scaled_train = lr_model.predict(X_train_ml_scaled)
pred_lr_train = scaler_y.inverse_transform(pred_lr_scaled_train.reshape(-1, 1)).flatten()
r2_lr = r2_score(y_train_ml, pred_lr_train)

pred_rf_train = rf_model.predict(X_train_ml)
r2_rf = r2_score(y_train_ml, pred_rf_train)

pred_xgb_train = xgb_model.predict(X_train_ml)
r2_xgb = r2_score(y_train_ml, pred_xgb_train)

pred_svr_scaled_train = svr_model.predict(X_train_ml_scaled)
pred_svr_train = scaler_y.inverse_transform(pred_svr_scaled_train.reshape(-1, 1)).flatten()
r2_svr = r2_score(y_train_ml, pred_svr_train)

print("\nTraining R² values:")
print("Linear Regression R²: {:.3f}".format(r2_lr))
print("Random Forest R²:     {:.3f}".format(r2_rf))
print("XGBoost R²:           {:.3f}".format(r2_xgb))
print("SVR R²:               {:.3f}".format(r2_svr))

# ---------------------------
# 3. LOAD UNLABELED TEST DATA FOR BACKCASTING (e.g., 1870-1899)
# ---------------------------
test_path = r"yearly_occurrence_data/pred_data_1870_1899.csv"
test_df = pd.read_csv(test_path)
test_df = test_df.sort_values("year").reset_index(drop=True)
# test_df contains only static predictors (no 'estimated_deaths')
test_df.set_index("year", inplace=True)

# ---------------------------
# 4. RECURSIVE BACKCASTING ON THE TEST SET
# ---------------------------
# Initialize the lead window using the first n_leads years from the training data.
train_orig = pd.read_csv(train_path)
train_orig = train_orig.sort_values("year").reset_index(drop=True)
train_orig.set_index("year", inplace=True)
lead_years = list(train_orig.index[:n_leads])
lead_window = [train_orig.loc[yr, "estimated_deaths"] for yr in lead_years]

# Dictionaries to store predictions for each model.
preds_lr  = {}
preds_rf  = {}
preds_xgb = {}
preds_svr = {}
preds_gpr = {}
preds_gam = {}

# To prevent runaway predictions from the recursive update, we also define a safe maximum
# for GAM's log prediction (to avoid overflow when applying exp).
SAFE_MAX_LOG = 4500

# Predict in descending order.
test_years_desc = sorted(test_df.index, reverse=True)

for year in test_years_desc:
    # Get static predictor values.
    static_vals = test_df.loc[[year], top_features]
    # Create DataFrame for lead features.
    current_leads_df = pd.DataFrame(
        np.array(lead_window).reshape(1, -1),
        columns=[f'lead_{i}' for i in range(1, n_leads+1)],
        index=static_vals.index
    )
    # Combine static and lead features.
    X_input_df = pd.concat([static_vals, current_leads_df], axis=1)
    X_input_df = X_input_df[ml_features]
    if X_input_df.isnull().any().any():
        X_input_df = X_input_df.fillna(0)
    
    # Scale inputs for models that require scaling.
    X_input_scaled = scaler_X.transform(X_input_df)
    
    # --- Linear Regression Prediction ---
    pred_lr_scaled = lr_model.predict(X_input_scaled)
    pred_lr = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
    pred_lr = np.clip(pred_lr, 0, max_threshold)
    
    # --- Random Forest Prediction ---
    pred_rf = rf_model.predict(X_input_df)[0]
    pred_rf = np.clip(pred_rf, 0, max_threshold)
    
    # --- XGBoost Prediction ---
    pred_xgb = xgb_model.predict(X_input_df)[0]
    pred_xgb = np.clip(pred_xgb, 0, max_threshold)
    
    # --- SVR Prediction ---
    pred_svr_scaled = svr_model.predict(X_input_scaled)
    pred_svr = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
    pred_svr = np.clip(pred_svr, 0, max_threshold)
    
    # --- GPR Prediction ---
    pred_gpr_scaled = gpr_model.predict(X_input_scaled)
    pred_gpr = scaler_y.inverse_transform(pred_gpr_scaled.reshape(-1, 1)).flatten()[0]
    pred_gpr = np.clip(pred_gpr, 0, max_threshold)
    
    # --- GAM Prediction ---
    pred_gam_log = gam_model.predict(X_input_scaled)
    pred_gam_log = np.clip(pred_gam_log, -np.inf, SAFE_MAX_LOG)
    pred_gam = np.exp(pred_gam_log) - 1
    pred_gam = np.clip(pred_gam, 0, max_threshold)
    
    # Save predictions.
    preds_lr[year]  = pred_lr
    preds_rf[year]  = pred_rf
    preds_xgb[year] = pred_xgb
    preds_svr[year] = pred_svr
    preds_gpr[year] = pred_gpr
    preds_gam[year] = pred_gam
    
    # Update the lead window using the (clipped) LR prediction.
    lead_window = [pred_lr] + lead_window[:-1]

# Convert prediction dictionaries to Series (sorted in ascending order).
years_sorted = sorted(preds_lr.keys())
preds_lr_series  = pd.Series({yr: preds_lr[yr] for yr in years_sorted})
preds_rf_series  = pd.Series({yr: preds_rf[yr] for yr in years_sorted})
preds_xgb_series = pd.Series({yr: preds_xgb[yr] for yr in years_sorted})
preds_svr_series = pd.Series({yr: preds_svr[yr] for yr in years_sorted})
preds_gpr_series = pd.Series({yr: preds_gpr[yr] for yr in years_sorted})
preds_gam_series = pd.Series({yr: preds_gam[yr] for yr in years_sorted})

# ---------------------------
# 5. VALIDATION BACKCASTING SIMULATION FROM TRAINING DATA
# ---------------------------
val_horizon = 20
train_orig2 = pd.read_csv(train_path)
train_orig2 = train_orig2.sort_values("year").reset_index(drop=True)
train_orig2.set_index("year", inplace=True)

val_years = sorted(train_orig2.index)[:val_horizon]
lead_years_val = list(train_orig2.index[val_horizon:val_horizon+n_leads])
lead_window_val = [train_orig2.loc[yr, "estimated_deaths"] for yr in lead_years_val]

# Dictionaries to store validation predictions.
preds_lr_val  = {}
preds_rf_val  = {}
preds_xgb_val = {}
preds_svr_val = {}
preds_gpr_val = {}
preds_gam_val = {}

for year in sorted(val_years, reverse=True):
    static_vals = train_orig2.loc[[year], top_features]
    current_leads_df = pd.DataFrame(
        np.array(lead_window_val).reshape(1, -1),
        columns=[f'lead_{i}' for i in range(1, n_leads+1)],
        index=static_vals.index
    )
    X_input_df = pd.concat([static_vals, current_leads_df], axis=1)
    X_input_df = X_input_df[ml_features]
    if X_input_df.isnull().any().any():
        X_input_df = X_input_df.fillna(0)
    
    X_input_scaled = scaler_X.transform(X_input_df)
    
    # Linear Regression validation prediction.
    pred_lr_scaled = lr_model.predict(X_input_scaled)
    pred_lr_val = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
    pred_lr_val = np.clip(pred_lr_val, 0, max_threshold)
    
    # Random Forest validation prediction.
    pred_rf_val = rf_model.predict(X_input_df)[0]
    pred_rf_val = np.clip(pred_rf_val, 0, max_threshold)
    
    # XGBoost validation prediction.
    pred_xgb_val = xgb_model.predict(X_input_df)[0]
    pred_xgb_val = np.clip(pred_xgb_val, 0, max_threshold)
    
    # SVR validation prediction.
    pred_svr_scaled = svr_model.predict(X_input_scaled)
    pred_svr_val = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
    pred_svr_val = np.clip(pred_svr_val, 0, max_threshold)
    
    # GPR validation prediction.
    pred_gpr_scaled_val = gpr_model.predict(X_input_scaled)
    pred_gpr_val = scaler_y.inverse_transform(pred_gpr_scaled_val.reshape(-1, 1)).flatten()[0]
    pred_gpr_val = np.clip(pred_gpr_val, 0, max_threshold)
    
    # GAM validation prediction.
    pred_gam_log_val = gam_model.predict(X_input_scaled)
    pred_gam_log_val = np.clip(pred_gam_log_val, -np.inf, SAFE_MAX_LOG)
    pred_gam_val = np.exp(pred_gam_log_val) - 1
    pred_gam_val = np.clip(pred_gam_val, 0, max_threshold)
    
    preds_lr_val[year]  = pred_lr_val
    preds_rf_val[year]  = pred_rf_val
    preds_xgb_val[year] = pred_xgb_val
    preds_svr_val[year] = pred_svr_val
    preds_gpr_val[year] = pred_gpr_val
    preds_gam_val[year] = pred_gam_val
    
    # Update validation lead window using the LR prediction.
    lead_window_val = [pred_lr_val] + lead_window_val[:-1]

# Convert validation predictions to Series.
preds_lr_val_series  = pd.Series({yr: preds_lr_val[yr] for yr in sorted(preds_lr_val.keys())})
preds_rf_val_series  = pd.Series({yr: preds_rf_val[yr] for yr in sorted(preds_rf_val.keys())})
preds_xgb_val_series = pd.Series({yr: preds_xgb_val[yr] for yr in sorted(preds_xgb_val.keys())})
preds_svr_val_series = pd.Series({yr: preds_svr_val[yr] for yr in sorted(preds_svr_val.keys())})
preds_gpr_val_series = pd.Series({yr: preds_gpr_val[yr] for yr in sorted(preds_gpr_val.keys())})
preds_gam_val_series = pd.Series({yr: preds_gam_val[yr] for yr in sorted(preds_gam_val.keys())})

actual_val = train_orig2.loc[preds_lr_val_series.index, "estimated_deaths"]

# Compute MAE for each model on the validation set.
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

print("\nValidation MAE for individual models:")
print("Linear Regression MAE: {:.3f}".format(mae_lr))
print("Random Forest MAE:     {:.3f}".format(mae_rf))
print("XGBoost MAE:           {:.3f}".format(mae_xgb))
print("SVR MAE:               {:.3f}".format(mae_svr))
print("GPR MAE:               {:.3f}".format(mae_gpr))
print("GAM MAE:               {:.3f}".format(mae_gam))

print("\nComputed ensemble weights based on validation MAE:")
print("Linear Regression: {:.3f}".format(w_lr))
print("Random Forest:     {:.3f}".format(w_rf))
print("XGBoost:           {:.3f}".format(w_xgb))
print("SVR:               {:.3f}".format(w_svr))
print("GPR:               {:.3f}".format(w_gpr))
print("GAM:               {:.3f}".format(w_gam))

# Compute ensemble prediction on the validation set (weighted).
ensemble_val_series = (w_lr   * preds_lr_val_series +
                       w_rf   * preds_rf_val_series +
                       w_xgb  * preds_xgb_val_series +
                       w_svr  * preds_svr_val_series +
                       w_gpr  * preds_gpr_val_series +
                       w_gam  * preds_gam_val_series)

mae_ensemble = mean_absolute_error(actual_val, ensemble_val_series)
print("Ensemble (Weighted Avg) MAE on validation set: {:.3f}".format(mae_ensemble))

# --- New: Compute Equal-Weighted Ensemble on Validation Set ---
ensemble_val_equal_series = (
    preds_lr_val_series +
    preds_rf_val_series +
    preds_xgb_val_series +
    preds_svr_val_series +
    preds_gpr_val_series +
    preds_gam_val_series
) / 6
mae_ensemble_equal = mean_absolute_error(actual_val, ensemble_val_equal_series)
print("Ensemble (Equal Avg) MAE on validation set: {:.3f}".format(mae_ensemble_equal))

# ---------------------------
# 6. COMPUTE THE ENSEMBLE PREDICTION ON THE TEST SET
# ---------------------------
ensemble_series = (w_lr   * preds_lr_series +
                   w_rf   * preds_rf_series +
                   w_xgb  * preds_xgb_series +
                   w_svr  * preds_svr_series +
                   w_gpr  * preds_gpr_series +
                   w_gam  * preds_gam_series)

# --- New: Compute Equal-Weighted Ensemble on Test Set ---
ensemble_equal_series = (
    preds_lr_series +
    preds_rf_series +
    preds_xgb_series +
    preds_svr_series +
    preds_gpr_series +
    preds_gam_series
) / 6

# ---------------------------
# 7. PLOTTING THE RESULTS
# ---------------------------
plt.figure(figsize=(14, 8))

# Plot individual model predictions.
plt.plot(
    preds_lr_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_lr_series.values, label="ML Linear Regression", marker='x'
)
plt.plot(
    preds_rf_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_rf_series.values, label="ML Random Forest", marker='^'
)
plt.plot(
    preds_xgb_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_xgb_series.values, label="ML XGBoost", marker='D'
)
plt.plot(
    preds_svr_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_svr_series.values, label="ML SVR", marker='v'
)
plt.plot(
    preds_gpr_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_gpr_series.values, label="ML GPR", marker='o'
)
plt.plot(
    preds_gam_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    preds_gam_series.values, label="ML GAM", marker='p'
)

# Plot the weighted ensemble prediction.
plt.plot(
    ensemble_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    ensemble_series.values, label="Ensemble (Weighted Avg)", marker='s', linewidth=3
)

# --- New: Plot the equal-weighted ensemble prediction.
plt.plot(
    ensemble_equal_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    ensemble_equal_series.values, label="Ensemble (Equal Avg)", marker='*', linewidth=3, linestyle='--'
)

plt.xlabel("Year")
plt.ylabel("Estimated Deaths")
plt.title("Backcasting from 1900 to 1870 (Predictions for 1870-1899)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
