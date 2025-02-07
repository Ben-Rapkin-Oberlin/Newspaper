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

from sklearn.model_selection import TimeSeriesSplit

from pygam import LinearGAM, s

###############################################################################
# Existing functions (run_backcast_simulation and run_full_fit_simulation)
# ... [your previous functions remain here if you wish to keep them]
###############################################################################

###############################################################################
# NEW: Function to Perform Time Series Cross Validation for Forecasting
###############################################################################
def run_forecast_cv(df, static_feature_list, n_splits=5):
    """
    Performs a one-step-ahead forecast evaluation using TimeSeriesSplit.
    
    For each fold:
      - The model is trained on the training fold (using the static features).
      - Then a one-step forecast is produced for each sample in the test fold.
      - The performance (MAE and R²) is computed on the test fold.
    
    Returns the average MAE and R² across folds.
    
    Here we illustrate the method using a linear regression model. You could
    extend this to the ensemble or any other model of interest.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_list = []
    r2_list = []
    
    for train_index, test_index in tscv.split(df):
        train_fold = df.iloc[train_index].copy()
        test_fold = df.iloc[test_index].copy()
        
        # Prepare the training data.
        X_train = train_fold[static_feature_list]
        y_train = train_fold['estimated_deaths']
        X_test  = test_fold[static_feature_list]
        y_test  = test_fold['estimated_deaths']
        
        # Scale features and target.
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # Train a linear regression model (for illustration).
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train_scaled)
        
        # Forecast on the test fold (one-step-ahead predictions).
        preds_scaled = lr_model.predict(X_test_scaled)
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mae_list.append(mae)
        r2_list.append(r2)
    
    avg_mae = np.mean(mae_list)
    avg_r2  = np.mean(r2_list)
    return avg_mae, avg_r2

###############################################################################
# MAIN SCRIPT
###############################################################################
# Load training data.
# (Adjust the path if needed.)
train_path = r"yearly_occurrence_data/training_data_1900_1936.csv"
train_df = pd.read_csv(train_path)
train_df = train_df.sort_values("year").reset_index(drop=True)

# Identify all static predictors (exclude 'year', 'estimated_deaths', and any lead columns).
static_cols = [col for col in train_df.columns if col not in ["year", "estimated_deaths"] and not col.startswith("lead_")]

# Limit the maximum number of features used in some groups.
max_features_to_use = 10

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

###############################################################################
# Evaluate Forecast Performance Using Time Series Cross Validation
###############################################################################
print("\n--- Time Series Cross Validation (One-Step-Ahead Forecast) ---")
mae_A_cv, r2_A_cv = run_forecast_cv(train_df.copy(), groupA, n_splits=5)
mae_C_cv, r2_C_cv = run_forecast_cv(train_df.copy(), groupC, n_splits=5)
mae_B_cv, r2_B_cv = run_forecast_cv(train_df.copy(), groupB, n_splits=5)

print(f"{groupA_name} (Linear Regression CV): MAE = {mae_A_cv:.3f}, R² = {r2_A_cv:.3f}")
print(f"{groupC_name} (Linear Regression CV): MAE = {mae_C_cv:.3f}, R² = {r2_C_cv:.3f}")
print(f"{groupB_name} (Linear Regression CV): MAE = {mae_B_cv:.3f}, R² = {r2_B_cv:.3f}")

###############################################################################
# (Optional) You can also compare these CV metrics to your backcasting and full-fit metrics.
# For example, you might re-run your backcasting simulation and compare its out-of-sample performance.
###############################################################################

# You can now use these cross validation results to better understand the generalizability
# of the predictive power of each predictor group. For instance, if Group A (the smallpox
# co‑occurrence features) shows consistently lower MAE and higher R² across folds, that
# suggests that those features yield more stable predictions over time.

# (You may later extend this approach to other models or to multi-step forecasts.)

plt.show()
