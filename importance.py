import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------
# Load the training data (original file, which includes the year column)
# -----------------------------------------------------------------------------
df = pd.read_csv(r"yearly_occurrence_data/training_data_1900_1936.csv")

# -----------------------------------------------------------------------------
# Preprocess: Drop the 'year' column (and any other columns you don't want as features)
# -----------------------------------------------------------------------------
df_features = df.drop(columns=["year"])

# Separate features (X) and target (y)
# Here, 'estimated_deaths' is the target, and all remaining columns are features.
X_features = df_features.drop(columns=["estimated_deaths"])
y_features = df_features["estimated_deaths"]

# -----------------------------------------------------------------------------
# Build a Random Forest Regressor to gauge feature importance
# -----------------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_features, y_features)

# Get the feature importances from the fitted model
importances = rf.feature_importances_
feature_names = np.array(X_features.columns)

# Sort the features by importance (highest first)
indices = np.argsort(importances)[::-1]

# -----------------------------------------------------------------------------
# Plot the Feature Importances
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.ylabel("Relative Importance")
plt.tight_layout()
plt.show()
