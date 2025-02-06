import pandas as pd

# Path to your test data.
test_path = r"yearly_occurrence_data\pred_data_1880_1899.csv"

# Load test data.
test_df = pd.read_csv(test_path)

# Print out the column names available in the test data.
print("Columns in test data:")
print(test_df.columns.tolist())

# Check for missing values in the test data.
print("\nMissing values per column in test data:")
print(test_df.isnull().sum())

# --- Checking for static predictor columns ---
# Suppose you have already determined top_features from your training code.
# For example, if top_features is:
top_features = ['total_smallpox_death_cooccurrences', 'topic5_total', 'topic2_fraction',
                'topic1_total', 'topic5_fraction']

# Check if these columns exist in the test data.
missing_static_cols = [col for col in top_features if col not in test_df.columns]
if missing_static_cols:
    print("\nWARNING: The following static predictor columns are missing in the test data:")
    print(missing_static_cols)
else:
    print("\nAll static predictor columns are present in the test data.")

# Optionally, for each static predictor, list the rows (years) that have missing values.
if not missing_static_cols:
    for col in top_features:
        if test_df[col].isnull().sum() > 0:
            print(f"\nColumn '{col}' has {test_df[col].isnull().sum()} missing values.")
            # Optionally, print the indices (or years) where values are missing:
            print("Rows with missing values in", col, ":\n", test_df[test_df[col].isnull()].index.tolist())
        else:
            print(f"\nColumn '{col}' has no missing values.")


import pandas as pd
import numpy as np

# Assume these variables are defined as in your workflow.
top_features = ['total_smallpox_death_cooccurrences', 'topic5_total', 'topic2_fraction',
                'topic1_total', 'topic5_fraction']
n_leads = 3

# Example: Load your test data.
test_path = r"yearly_occurrence_data\pred_data_1880_1899.csv"
test_df = pd.read_csv(test_path)
test_df = test_df.sort_values("year").reset_index(drop=True)
test_df.set_index("year", inplace=True)

# Simulate one iteration (e.g., for year 1899) and a sample lead_window.
year = 1899
# Extract static predictors for this year.
static_vals = test_df.loc[[year], top_features]

# Example lead_window (normally, you'd get these from training data)
lead_window = [1000, 1100, 1050]  # Replace with actual numbers

# Create the lead DataFrame with the same index as static_vals.
current_leads_df = pd.DataFrame(np.array(lead_window).reshape(1, -1),
                                columns=[f'lead_{i}' for i in range(1, n_leads+1)],
                                index=static_vals.index)

# Concatenate along axis=1.
X_input_df = pd.concat([static_vals, current_leads_df], axis=1)

print("Concatenated input DataFrame:")
print(X_input_df)
print("\nAre there any missing values?", X_input_df.isnull().any().any())
