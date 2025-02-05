#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined script to forecast historical estimated deaths using several methods:
    1. Linear Regression (with bootstrapped uncertainty)
    2. GAM (using pyGAM with smooth terms only)
    3. Bayesian Regression (using sklearn’s BayesianRidge)
    4. Small GRU (with Monte Carlo dropout uncertainty)
    
An ensemble forecast is then computed by averaging the predictions from the four methods.

This version:
   - Reverses the order of the training and test data (so that the most recent observations come first).
   - Uses a TimeSeriesSplit with a gap.
   - Standardizes the features.
   - Plots model performances on the training data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from functools import reduce
import sys
import os

#####################################
# 1. Data Loading, Reversal & Preparation
#####################################
print(">>> Loading data...")

train_path = r"yearly_occurrence_data\training_data_1900_1936.csv"
test_path = r"yearly_occurrence_data\pred_data_1880_1899.csv"

# Load CSVs
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print(">>> Data loaded successfully.")
print("\nTraining data sample (original order):")
print(train_df.head())

# Reverse the data order (so that the most recent years come first)
train_df = train_df.iloc[::-1].reset_index(drop=True)
test_df = test_df.iloc[::-1].reset_index(drop=True)

print("\nTraining data sample (reversed order):")
print(train_df.head())
print("\nTest data sample (reversed order):")
print(test_df.head())

# Define features and target.
features = [
    "year", "n_articles", "total_word_count",
    "topic1_total", "topic2_total", "topic3_total", "topic4_total", "topic5_total",
    "topic1_avg_prob", "topic2_avg_prob", "topic3_avg_prob", "topic4_avg_prob", "topic5_avg_prob",
    "total_smallpox_death_cooccurrences"
]
target = "estimated_deaths"

# Create DataFrames for modeling.
X_train = train_df[features].copy()
y_train = train_df[target].copy()
X_test = test_df[features].copy()  # test data has no target

# Standardize features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for models that require it.
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)

# For GRU, we work with NumPy arrays.
X_train_gru = X_train_scaled  # shape (n_samples, n_features)
X_test_gru = X_test_scaled

print(">>> Data preparation complete.\n")

#####################################
# 2. Robust Time Series Cross-Validation Setup (with gap)
#####################################
print(">>> Setting up TimeSeriesSplit cross-validation with gap...")
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)
print(f">>> Using {n_splits} splits for cross-validation.\n")

#####################################
# 3. Model 1: Linear Regression with Bootstrapped Uncertainty
#####################################
print(">>> Starting Linear Regression cross-validation...")
lr_mse_scores = []
lr_cv_preds = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled_df)):
    print(f"   Processing fold {fold+1}...")
    X_tr = X_train_scaled_df.iloc[train_idx]
    X_val = X_train_scaled_df.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    lr = LinearRegression().fit(X_tr, y_tr)
    y_pred = lr.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred)
    lr_mse_scores.append(mse_val)
    lr_cv_preds.append(y_pred)
    print(f"      Fold {fold+1} MSE: {mse_val:.2f}")

lr_cv_mse = np.mean(lr_mse_scores)
print(f">>> Average CV MSE (Linear Regression): {lr_cv_mse:.2f}")

print(">>> Fitting final Linear Regression model on full training data...")
lr_final = LinearRegression().fit(X_train_scaled_df, y_train)
lr_train_preds = lr_final.predict(X_train_scaled_df)
residuals = y_train - lr_train_preds
resid_std = np.std(residuals)
z = 1.96
lr_pred_interval = (lr_train_preds - z * resid_std, lr_train_preds + z * resid_std)
print(">>> Linear Regression prediction intervals (first 5 points):")
for i in range(5):
    print(f"   Predicted: {lr_train_preds[i]:.2f}, 95% PI: ({lr_pred_interval[0][i]:.2f}, {lr_pred_interval[1][i]:.2f})")
print(">>> Linear Regression complete.\n")

#####################################
# 4. Model 2: GAM (Smooth Terms Only)
#####################################
print(">>> Starting GAM (smooth terms only) cross-validation...")
n_features = len(features)
terms = [s(i) for i in range(n_features)]
gam_terms = reduce(lambda a, b: a + b, terms)

gam_mse_scores = []
gam_cv_preds = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled_df)):
    print(f"   Processing fold {fold+1}...")
    X_tr = X_train_scaled_df.iloc[train_idx]
    X_val = X_train_scaled_df.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    gam = LinearGAM(gam_terms)
    gam.fit(X_tr.values, y_tr.values)
    y_pred = gam.predict(X_val.values)
    mse_val = mean_squared_error(y_val, y_pred)
    gam_mse_scores.append(mse_val)
    gam_cv_preds.append(y_pred)
    print(f"      Fold {fold+1} MSE: {mse_val:.2f}")

gam_cv_mse = np.mean(gam_mse_scores)
print(f">>> Average CV MSE (GAM): {gam_cv_mse:.2f}")

print(">>> Fitting final GAM model on full training data...")
gam_final = LinearGAM(gam_terms).fit(X_train_scaled_df.values, y_train.values)
gam_train_preds = gam_final.predict(X_train_scaled_df.values)
gam_conf_intervals = gam_final.confidence_intervals(X_train_scaled_df.values, width=0.95)
print(">>> GAM prediction intervals (first 5 points):")
for i in range(5):
    print(f"   Predicted: {gam_train_preds[i]:.2f}, 95% CI: ({gam_conf_intervals[i,0]:.2f}, {gam_conf_intervals[i,1]:.2f})")
print(">>> GAM complete.\n")

#####################################
# 5. Model 3: Bayesian Regression (BayesianRidge)
#####################################
print(">>> Starting Bayesian Regression cross-validation...")
bayes_mse_scores = []
bayes_cv_preds = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled_df)):
    print(f"   Processing fold {fold+1}...")
    X_tr = X_train_scaled_df.iloc[train_idx]
    X_val = X_train_scaled_df.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    bayes = BayesianRidge()
    bayes.fit(X_tr, y_tr)
    y_pred = bayes.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred)
    bayes_mse_scores.append(mse_val)
    bayes_cv_preds.append(y_pred)
    print(f"      Fold {fold+1} MSE: {mse_val:.2f}")

bayes_cv_mse = np.mean(bayes_mse_scores)
print(f">>> Average CV MSE (Bayesian Regression): {bayes_cv_mse:.2f}")

print(">>> Fitting final Bayesian Regression model on full training data...")
bayes_final = BayesianRidge().fit(X_train_scaled_df, y_train)
bayes_train_preds = bayes_final.predict(X_train_scaled_df)
print(">>> Bayesian Regression complete.\n")

#####################################
# 6. Model 4: Small GRU with Monte Carlo Dropout
#####################################
print(">>> Preparing GRU training sequences...")

def create_sequences(X, y, window_size=3):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 3
X_train_gru_np = X_train_gru.astype(np.float32)
y_train_gru_np = y_train.values.astype(np.float32)
X_seq, y_seq = create_sequences(X_train_gru_np, y_train_gru_np, window_size=window_size)
print(f">>> GRU sequence shape: {X_seq.shape}, {y_seq.shape}")

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=8, activation='tanh', return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

print(">>> Starting GRU cross-validation...")
gru_mse_scores = []
gru_cv_preds = []
gru_epochs = 200
gru_batch_size = 4
gru_patience = 10

# Use a separate TimeSeriesSplit for sequence data.
tscv_seq = TimeSeriesSplit(n_splits=n_splits, gap=1)
for fold, (train_idx, val_idx) in enumerate(tscv_seq.split(X_seq)):
    print(f"   Processing fold {fold+1}...")
    model = build_gru_model(input_shape=(window_size, X_seq.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=gru_patience, restore_best_weights=True)
    model.fit(X_seq[train_idx], y_seq[train_idx],
              epochs=gru_epochs, batch_size=gru_batch_size,
              verbose=0, callbacks=[early_stop])
    y_pred = model.predict(X_seq[val_idx]).flatten()
    mse_val = mean_squared_error(y_seq[val_idx], y_pred)
    gru_mse_scores.append(mse_val)
    gru_cv_preds.append(y_pred)
    print(f"      Fold {fold+1} MSE: {mse_val:.2f}")

gru_cv_mse = np.mean(gru_mse_scores)
print(f">>> Average CV MSE (GRU): {gru_cv_mse:.2f}")

def predict_mc_dropout(model, X, n_iter=100):
    preds = []
    for _ in range(n_iter):
        preds.append(model(X, training=True).numpy().flatten())
    preds = np.array(preds)
    mean_preds = preds.mean(axis=0)
    std_preds = preds.std(axis=0)
    return mean_preds, std_preds

print(">>> Fitting final GRU model on full training sequence data for MC dropout uncertainty...")
gru_model_full = build_gru_model(input_shape=(window_size, X_seq.shape[2]))
early_stop_full = EarlyStopping(monitor='loss', patience=gru_patience, restore_best_weights=True)
gru_model_full.fit(X_seq, y_seq, epochs=gru_epochs, batch_size=gru_batch_size,
                   verbose=0, callbacks=[early_stop_full])
gru_mean_train, gru_std_train = predict_mc_dropout(gru_model_full, X_seq, n_iter=100)
print(">>> GRU predictions with MC dropout (first 5 sequences):")
for i in range(5):
    print(f"   Predicted: {gru_mean_train[i]:.2f} ± {gru_std_train[i]:.2f}")
print(">>> GRU model complete.\n")

#####################################
# 7. Final Model Fitting & Backcasting on Test Data
#####################################
print(">>> Fitting final models on full training data and backcasting on test data...")

# Final predictions:
print("   Fitting final Linear Regression...")
lr_final = LinearRegression().fit(X_train_scaled_df, y_train)
lr_test_preds = lr_final.predict(X_test_scaled_df)

print("   Fitting final GAM model...")
gam_final = LinearGAM(gam_terms).fit(X_train_scaled_df.values, y_train.values)
gam_test_preds = gam_final.predict(X_test_scaled_df.values)

print("   Fitting final Bayesian Regression model...")
bayes_final = BayesianRidge().fit(X_train_scaled_df, y_train)
bayes_test_preds = bayes_final.predict(X_test_scaled_df)

print("   Fitting final GRU model backcasting on test data...")
def create_test_sequences(X, window_size=3):
    Xs = []
    for i in range(len(X) - window_size + 1):
        Xs.append(X[i:i+window_size])
    return np.array(Xs)

X_test_seq = create_test_sequences(X_test_gru, window_size=window_size)
gru_mean_test, gru_std_test = predict_mc_dropout(gru_model_full, X_test_seq, n_iter=100)

# Pad GRU predictions to match length.
gru_full_preds = np.full(len(lr_test_preds), np.nan)
gru_full_preds[:len(gru_mean_test)] = gru_mean_test

# Assemble predictions for ensemble.
ensemble_dict = {
    'LR': lr_test_preds,
    'GAM': gam_test_preds,
    'Bayesian': bayes_test_preds,
    'GRU': gru_full_preds
}
ensemble_array = np.vstack([ensemble_dict[m] for m in ensemble_dict])
ensemble_mean = np.nanmean(ensemble_array, axis=0)

print(">>> Final backcast predictions:")
print("Linear Regression (first 5):", lr_test_preds[:5])
print("GAM (first 5):", gam_test_preds[:5])
print("Bayesian (first 5):", bayes_test_preds[:5])
print("GRU (first 5):", gru_full_preds[:5])
print("Ensemble (first 5):", ensemble_mean[:5])
print(">>> Final model fitting and backcasting complete.\n")

#####################################
# 8. Plotting Model Performances
#####################################
print(">>> Plotting model performance comparisons...")

# Plot training fits for each final model.
plt.figure(figsize=(12, 8))
plt.plot(train_df['year'], y_train, 'ko-', label='Actual (Training)')
plt.plot(train_df['year'], lr_final.predict(X_train_scaled_df), 'b--', label='Linear Regression')
plt.plot(train_df['year'], gam_final.predict(X_train_scaled_df.values), 'r-.', label='GAM')
plt.plot(train_df['year'], bayes_final.predict(X_train_scaled_df), 'g:', label='Bayesian Regression')
# For GRU, plot the predictions for the training sequences (aligned with the corresponding time stamps)
# We plot the mean GRU predictions on the training set for indices [window_size:]
plt.plot(train_df['year'][window_size:], gru_mean_train, 'm--', label='GRU (MC dropout)')
plt.xlabel("Year (Reversed Order)")
plt.ylabel("Estimated Deaths")
plt.title("Training Fit Comparison for All Models")
plt.legend()
plt.gca().invert_xaxis()  # Invert x-axis so that most recent years are on the left
plt.show()

# Plot a bar chart of average CV MSE for each model.
cv_mse = {
    'Linear Regression': lr_cv_mse,
    'GAM': gam_cv_mse,
    'Bayesian Regression': bayes_cv_mse,
    'GRU': gru_cv_mse
}
plt.figure(figsize=(8, 6))
plt.bar(cv_mse.keys(), cv_mse.values(), color=['blue', 'red', 'green', 'magenta'])
plt.ylabel("Average CV MSE")
plt.title("Cross-Validation MSE Comparison")
plt.yscale("log")  # Use logarithmic scale if errors vary greatly
plt.show()

print(">>> Plotting complete.")
print(">>> Script completed successfully.")
