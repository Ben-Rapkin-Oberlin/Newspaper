import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Data for validation metrics
validation_metrics = {
    'Model': ['XGBoost', 'LASSO', 'Log-LASSO'],
    'MAE': [311.59, 408.17, 407.09],
    'RMSE': [744.87, 807.13, 1017.92],
    'R²': [0.404, 0.301, -0.112],
    'Explained Variance': [0.461, 0.309, 0.0],
    'MAPE': [109.19, 311.61, 280.76]
}

# Data for full dataset metrics
full_dataset_metrics = {
    'Model': ['XGBoost', 'LASSO', 'Log-LASSO'],
    'MAE': [119.42, 276.72, 380.81],
    'RMSE': [324, 437.55, 916.67],
    'R²': [0.871, 0.794, 0.098],
    'Explained Variance': [0.875, 0.80, 0.156],
    'MAPE': [91.80, 369.66, 284.69]
}

# Create DataFrames
df_validation = pd.DataFrame(validation_metrics)
df_full = pd.DataFrame(full_dataset_metrics)

# List of metrics to plot
metrics_to_plot = ['MAE', 'RMSE', 'R²', 'Explained Variance', 'MAPE']

# Define which metrics are "error metrics" (lower is better) vs "goodness metrics" (higher is better)
error_metrics = ['MAE', 'RMSE', 'MAPE']
goodness_metrics = ['R²', 'Explained Variance']

# Create two separate figures - first for validation dataset
fig_validation, axs_validation = plt.subplots(1, 5, figsize=(20, 5))
fig_validation.suptitle('Validation Dataset Performance', fontsize=16, y=1.05)

# Loop through metrics for validation dataset
for i, metric in enumerate(metrics_to_plot):
    ax = axs_validation[i]
    bars = ax.bar(df_validation['Model'], df_validation[metric])
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Set appropriate y-axis limits
    if metric in ['R²']:
        # For R² and Explained Variance, use symmetric limits centered around 0
        max_abs_val = max(abs(df_validation[metric].max()), abs(df_validation[metric].min()))
        ax.set_ylim(-max_abs_val * 1.2, max_abs_val * 1.2)
    elif metric == 'MAPE':
        ax.set_ylim(0, max(df_validation[metric]) * 1.2)  # Give extra room at the top
    elif metric == 'MAE':
        ax.set_ylim(0, max(df_validation[metric]) * 1.2)  # Give extra room at the top
    elif metric == 'RMSE':
        ax.set_ylim(0, max(df_validation[metric]) * 1.15)  # Give extra room at the top
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if metric in goodness_metrics:
            label = f'{height:.2f}'
        elif metric == 'MAPE':
            label = f'{height:.1f}%'
        else:
            label = f'{height:.1f}'
        
        # Position the label above or below the bar depending on whether it's positive or negative
        va = 'bottom' if height >= 0 else 'top'
        
        # Adjust offset to prevent overlap with title
        if metric == 'MAPE' and height > 300:
            y_offset = -30  # Place labels inside the bar for very tall MAPE values
            va = 'top'
        elif height > 800:  # For very tall bars like RMSE
            y_offset = -30
            va = 'top'
        else:
            y_offset = 0.05 * max(df_validation[metric]) if height >= 0 else -0.05 * max(df_validation[metric])
        
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                label, ha='center', va=va, fontweight='bold')
    
    # Color bars
    if metric in error_metrics:
        best_idx = np.argmin(df_validation[metric])
    else:
        best_idx = np.argmax(df_validation[metric])
    
    for j, bar in enumerate(bars):
        bar.set_color('forestgreen' if j == best_idx else 'steelblue')

# Create second figure for full dataset
fig_full, axs_full = plt.subplots(1, 5, figsize=(20, 5))
fig_full.suptitle('Full Dataset Performance', fontsize=16, y=1.05)

# Loop through metrics for full dataset
for i, metric in enumerate(metrics_to_plot):
    ax = axs_full[i]
    bars = ax.bar(df_full['Model'], df_full[metric])
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Set appropriate y-axis limits
    if metric in ['a²']:
        # For R² and Explained Variance, use symmetric limits centered around 0
        max_abs_val = max(abs(df_full[metric].max()), abs(df_full[metric].min()))
        ax.set_ylim(-max_abs_val * 1.2, max_abs_val * 1.2)
    elif metric == 'MAPE':
        ax.set_ylim(0, max(df_full[metric]) * 1.2)  # Give extra room at the top
    elif metric == 'MAE':
        ax.set_ylim(0, max(df_full[metric]) * 1.2)  # Give extra room at the top
    elif metric == 'RMSE':
        ax.set_ylim(0, max(df_full[metric]) * 1.15)  # Give extra room at the top
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if metric in goodness_metrics:
            label = f'{height:.2f}'
        elif metric == 'MAPE':
            label = f'{height:.1f}%'
        else:
            label = f'{height:.1f}'
        
        # Position the label above or below the bar depending on whether it's positive or negative
        va = 'bottom' if height >= 0 else 'top'
        
        # Adjust offset to prevent overlap with title
        if metric == 'MAPE' and height > 300:
            y_offset = -30  # Place labels inside the bar for very tall MAPE values
            va = 'top'
        elif height > 800:  # For very tall bars like RMSE
            y_offset = -30
            va = 'top'
        else:
            y_offset = 0.05 * max(df_full[metric]) if height >= 0 else -0.05 * max(df_full[metric])
        
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                label, ha='center', va=va, fontweight='bold')
    
    # Color bars
    if metric in error_metrics:
        best_idx = np.argmin(df_full[metric])
    else:
        best_idx = np.argmax(df_full[metric])
    
    for j, bar in enumerate(bars):
        bar.set_color('forestgreen' if j == best_idx else 'steelblue')

# Adjust layout and save
plt.figure(fig_validation.number)
plt.tight_layout()
plt.savefig('validation_dataset_comparison.png', dpi=300, bbox_inches='tight')

plt.figure(fig_full.number)
plt.tight_layout()
plt.savefig('full_dataset_comparison.png', dpi=300, bbox_inches='tight')

print("Visualizations have been saved as 'validation_dataset_comparison.png' and 'full_dataset_comparison.png'")