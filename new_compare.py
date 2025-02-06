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
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# ---------------------------
# User-Defined Settings
# ---------------------------
# Set the number of lead features you want to use.
n_leads = 1  # You can change this to any positive integer.

# ---------------------------
# 2. TRAINING: CREATE LEAD FEATURES & TRAIN ML MODELS
# ---------------------------
train_path = r"yearly_occurrence_data\training_data_1900_1936.csv"

# Load labeled training data.
train_df = pd.read_csv(train_path)
train_df = train_df.sort_values("year").reset_index(drop=True)

# Create lead features – these are the “future” values.
for i in range(1, n_leads + 1):
    train_df[f'lead_{i}'] = train_df['estimated_deaths'].shift(-i)

# Drop rows that don’t have full lead information.
train_df = train_df.dropna().reset_index(drop=True)

# ---------------------------
# 2a. Determine Top Static Features
# ---------------------------
# Exclude columns that are not part of the original predictors.
cols_for_importance = [
    col for col in train_df.columns 
    if col not in ["year", "estimated_deaths"] + [f'lead_{i}' for i in range(1, n_leads+1)]
]

# Use a Random Forest to select the top n static features.
rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(train_df[cols_for_importance], train_df['estimated_deaths'])
importances = rf_full.feature_importances_
feature_names = np.array(cols_for_importance)

# Here we select the top features. (You can adjust the number as needed.)
top_n = min(8, len(feature_names))
indices = np.argsort(importances)[::-1][:top_n]
top_features = feature_names[indices]
print("Top {} features from original predictors: {}".format(top_n, top_features))

# Define the features that the ML models will use: top static predictors + lead features.
ml_features = list(top_features) + [f'lead_{i}' for i in range(1, n_leads+1)]

# ---------------------------
# 2b. Prepare Training Data for ML Models
# ---------------------------
X_train_ml = train_df[ml_features]
y_train_ml = train_df['estimated_deaths']  # target for current year

# Scale features for models that require it.
scaler_X = StandardScaler()
X_train_ml_scaled = scaler_X.fit_transform(X_train_ml)

scaler_y = StandardScaler()
y_train_ml_scaled = scaler_y.fit_transform(y_train_ml.values.reshape(-1, 1)).flatten()

# ---------------------------
# 2c. Train the ML Models on the Entire Training Data
# ---------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_ml_scaled, y_train_ml_scaled)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_ml, y_train_ml)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_ml, y_train_ml)

# Support Vector Regression
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_ml_scaled, y_train_ml_scaled)

# ---------------------------
# 3. LOAD UNLABELED TEST DATA FOR BACKCASTING (1880-1899)
# ---------------------------
test_path = r"yearly_occurrence_data\pred_data_1880_1899.csv"
test_df = pd.read_csv(test_path)
test_df = test_df.sort_values("year").reset_index(drop=True)
# Note: test_df contains only static predictors (no 'estimated_deaths')
test_df.set_index("year", inplace=True)

# ---------------------------
# 4. RECURSIVE BACKCASTING ON THE TEST SET
# ---------------------------
# For recursive backcasting, we need an initial lead window.
# Instead of hardcoding the years, we derive the first n_leads years from the training data.
train_orig = pd.read_csv(train_path)
train_orig = train_orig.sort_values("year").reset_index(drop=True)
train_orig.set_index("year", inplace=True)
lead_years = list(train_orig.index[:n_leads])
lead_window = [train_orig.loc[yr, "estimated_deaths"] for yr in lead_years]

# We predict in descending order (from the latest test year down to the earliest).
test_years_desc = sorted(test_df.index, reverse=True)

# Dictionaries to store predictions for each model.
preds_lr = {}
preds_rf = {}
preds_xgb = {}
preds_svr = {}

for year in test_years_desc:
    # Get static predictor values.
    static_vals = test_df.loc[[year], top_features]
    
    # Create a DataFrame for the lead values with the same index.
    current_leads_df = pd.DataFrame(
        np.array(lead_window).reshape(1, -1),
        columns=[f'lead_{i}' for i in range(1, n_leads+1)],
        index=static_vals.index
    )
    
    # Combine static predictors and lead features.
    X_input_df = pd.concat([static_vals, current_leads_df], axis=1)
    X_input_df = X_input_df[ml_features]  # Ensure correct column order.
    
    if X_input_df.isnull().any().any():
        X_input_df = X_input_df.fillna(0)
    
    # Scale input for models that require scaling.
    X_input_scaled = scaler_X.transform(X_input_df)
    
    # Generate predictions from each model.
    pred_lr_scaled = lr_model.predict(X_input_scaled)
    pred_lr = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
    
    pred_rf = rf_model.predict(X_input_df)[0]
    pred_xgb = xgb_model.predict(X_input_df)[0]
    
    pred_svr_scaled = svr_model.predict(X_input_scaled)
    pred_svr = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
    
    # Save predictions.
    preds_lr[year] = pred_lr
    preds_rf[year] = pred_rf
    preds_xgb[year] = pred_xgb
    preds_svr[year] = pred_svr
    
    # Update lead window using the LR prediction.
    lead_window = [pred_lr] + lead_window[:-1]

# Convert prediction dictionaries to Series (sorted in ascending order).
years_sorted = sorted(preds_lr.keys())
preds_lr_series = pd.Series({yr: preds_lr[yr] for yr in years_sorted})
preds_rf_series = pd.Series({yr: preds_rf[yr] for yr in years_sorted})
preds_xgb_series = pd.Series({yr: preds_xgb[yr] for yr in years_sorted})
preds_svr_series = pd.Series({yr: preds_svr[yr] for yr in years_sorted})

# ---------------------------
# 5. VALIDATION BACKCASTING SIMULATION FROM TRAINING DATA
# ---------------------------
val_horizon = 20
train_orig2 = pd.read_csv(train_path)
train_orig2 = train_orig2.sort_values("year").reset_index(drop=True)
train_orig2.set_index("year", inplace=True)

# Define validation years as the earliest val_horizon years.
val_years = sorted(train_orig2.index)[:val_horizon]

# Create a lead window from the next n_leads years.
lead_years_val = list(train_orig2.index[val_horizon:val_horizon+n_leads])
lead_window_val = [train_orig2.loc[yr, "estimated_deaths"] for yr in lead_years_val]

# Dictionaries to store validation predictions.
preds_lr_val = {}
preds_rf_val = {}
preds_xgb_val = {}
preds_svr_val = {}

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
    
    pred_lr_scaled = lr_model.predict(X_input_scaled)
    pred_lr_val = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()[0]
    pred_rf_val = rf_model.predict(X_input_df)[0]
    pred_xgb_val = xgb_model.predict(X_input_df)[0]
    pred_svr_scaled = svr_model.predict(X_input_scaled)
    pred_svr_val = scaler_y.inverse_transform(pred_svr_scaled.reshape(-1, 1)).flatten()[0]
    
    preds_lr_val[year] = pred_lr_val
    preds_rf_val[year] = pred_rf_val
    preds_xgb_val[year] = pred_xgb_val
    preds_svr_val[year] = pred_svr_val
    
    # Update validation lead window using the LR prediction.
    lead_window_val = [pred_lr_val] + lead_window_val[:-1]

# Convert validation predictions to Series.
preds_lr_val_series = pd.Series({yr: preds_lr_val[yr] for yr in sorted(preds_lr_val.keys())})
preds_rf_val_series = pd.Series({yr: preds_rf_val[yr] for yr in sorted(preds_rf_val.keys())})
preds_xgb_val_series = pd.Series({yr: preds_xgb_val[yr] for yr in sorted(preds_xgb_val.keys())})
preds_svr_val_series = pd.Series({yr: preds_svr_val[yr] for yr in sorted(preds_svr_val.keys())})

# Get the actual target values for the validation years.
actual_val = train_orig2.loc[preds_lr_val_series.index, "estimated_deaths"]

# Compute MAE for each model on the validation set.
mae_lr = mean_absolute_error(actual_val, preds_lr_val_series)
mae_rf = mean_absolute_error(actual_val, preds_rf_val_series)
mae_xgb = mean_absolute_error(actual_val, preds_xgb_val_series)
mae_svr = mean_absolute_error(actual_val, preds_svr_val_series)

epsilon = 1e-6  # small constant to avoid division by zero in weight calculation
inv_lr = 1 / (mae_lr + epsilon)
inv_rf = 1 / (mae_rf + epsilon)
inv_xgb = 1 / (mae_xgb + epsilon)
inv_svr = 1 / (mae_svr + epsilon)
total_inv = inv_lr + inv_rf + inv_xgb + inv_svr

w_lr = inv_lr / total_inv
w_rf = inv_rf / total_inv
w_xgb = inv_xgb / total_inv
w_svr = inv_svr / total_inv

print("\nValidation MAE for individual models:")
print("Linear Regression MAE: {:.3f}".format(mae_lr))
print("Random Forest MAE:     {:.3f}".format(mae_rf))
print("XGBoost MAE:           {:.3f}".format(mae_xgb))
print("SVR MAE:               {:.3f}".format(mae_svr))

print("\nComputed ensemble weights based on validation MAE:")
print("Linear Regression: {:.3f}, Random Forest: {:.3f}, XGBoost: {:.3f}, SVR: {:.3f}".format(
    w_lr, w_rf, w_xgb, w_svr
))

# Compute ensemble prediction on the validation set.
ensemble_val_series = (w_lr * preds_lr_val_series +
                       w_rf * preds_rf_val_series +
                       w_xgb * preds_xgb_val_series +
                       w_svr * preds_svr_val_series)

mae_ensemble = mean_absolute_error(actual_val, ensemble_val_series)
print("Ensemble MAE on validation set: {:.3f}".format(mae_ensemble))

# ---------------------------
# 6. COMPUTE THE ENSEMBLE PREDICTION ON THE TEST SET
# ---------------------------
ensemble_series = (w_lr * preds_lr_series +
                   w_rf * preds_rf_series +
                   w_xgb * preds_xgb_series +
                   w_svr * preds_svr_series)

# ---------------------------
# 7. PLOTTING THE RESULTS
# ---------------------------
plt.figure(figsize=(14, 8))

# Plot individual model predictions on the test set.
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

# Plot the ensemble prediction.
plt.plot(
    ensemble_series.index.to_series().apply(lambda yr: pd.to_datetime(f"{yr}-01-01")),
    ensemble_series.values, label="Ensemble (Weighted Avg)", marker='s', linewidth=3
)

plt.xlabel("Year")
plt.ylabel("Estimated Deaths")
plt.title("Backcasting from 1900 to 1880 (Predictions for 1880-1899)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
