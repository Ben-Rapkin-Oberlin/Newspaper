import os
import pandas as pd

# Base directory where the CSV files are stored.
base_path = r"window_5_10.0"

# List to hold the aggregated data for each year.
aggregated_data = []

start = 1900
end = 1936

# Loop over the years of interest.
for year in range(start, end + 1):
    print(f"Processing year: {year}")
    file_name = f"{year}_cooccurrence_analysis.csv"
    file_path = os.path.join(base_path, file_name)
    
    # Check that the file exists.
    if not os.path.exists(file_path):
        print(f"Warning: File not found for year {year} at {file_path}")
        continue

    # Read the CSV file.
    df = pd.read_csv(file_path)

    # Count of articles for the year.
    n_articles = len(df)

    # Total word count across articles.
    total_word_count = df['word_count'].sum() if 'word_count' in df.columns else None

    # -----------------------------
    # 1. Aggregate co-occurrence totals by topic.
    # -----------------------------
    topic_totals = {}
    for topic in range(1, 6):
        regex_pattern = f"^cooccur_topic{topic}_"
        topic_cols = df.filter(regex=regex_pattern).columns
        if not topic_cols.empty:
            topic_sum = df[topic_cols].sum().sum()
        else:
            topic_sum = 0
        topic_totals[f"topic{topic}_total"] = topic_sum
        topic_totals[f"topic{topic}_total_ratio"] = (
            topic_sum / total_word_count if total_word_count and total_word_count > 0 else None
        )

    # -----------------------------
    # 2. Compute average topic probabilities.
    # -----------------------------
    topic_avg_probs = {}
    for topic in range(1, 6):
        col_name = f"topic_{topic}_prob"
        if col_name in df.columns:
            topic_avg_probs[f"topic{topic}_avg_prob"] = df[col_name].mean()
        else:
            topic_avg_probs[f"topic{topic}_avg_prob"] = None

    # -----------------------------
    # 3. Aggregate all co-occurrence features.
    #    (This includes any column starting with 'cooccur_' or 'smallpox_',
    #     which makes it general to any new words you add later.)
    # -----------------------------
    # Exclude the columns we don’t want to aggregate.
    cooccurrence_cols = [
        col for col in df.columns
        if col not in ['id', 'word_count']
           and (col.startswith('cooccur_') or col.startswith('smallpox_'))
    ]

    aggregated_cooccurrence = {}
    for col in cooccurrence_cols:
        col_sum = df[col].sum()
        ratio = col_sum / total_word_count if total_word_count and total_word_count > 0 else None
        # Store both the aggregated sum and its ratio.
        aggregated_cooccurrence[col] = col_sum
        aggregated_cooccurrence[col + "_ratio"] = ratio

    # -----------------------------
    # 4. Combine all aggregated information into one dictionary.
    #    (Note: we do not include the "id" column.)
    # -----------------------------
    year_data = {
        "year": year,
        "n_articles": n_articles,
        "total_word_count": total_word_count,
    }
    year_data.update(topic_totals)
    year_data.update(topic_avg_probs)
    year_data.update(aggregated_cooccurrence)

    aggregated_data.append(year_data)

# Convert the list of dictionaries into a DataFrame.
agg_df = pd.DataFrame(aggregated_data)

# Optionally, sort the DataFrame by year.
agg_df = agg_df.sort_values(by="year")

# -----------------------------
# 5. Merge with Death Estimates (if applicable).
#    (Your original logic merges truth data only for years >= 1900.)
# -----------------------------
if start >= 1900:
    print("Including truth data")
    a = "training_data"
    # Read the ground truth death estimates.
    death_df = pd.read_csv(r"/usr/users/quota/students/2021/brapkin/Newspaper/data/death_estimates.csv")
    print("\nDeath estimates loaded. Sample:")
    print(death_df.head())

    # Keep only the 'year' and 'estimated_deaths' columns.
    death_subset = death_df[['year', 'estimated_deaths']]

    # Merge on the 'year' column.
    final_df = pd.merge(agg_df, death_subset, on='year', how='inner')
else:
    print("No truth data to merge")
    a = "pred_data"
    final_df = agg_df

print("\nFinal aggregated dataset sample:")
print(final_df.head())

# -----------------------------
# 6. Save the aggregated data to a new CSV file.
# -----------------------------
output_filename = f"{a}_{start}_{end}.csv"
final_df.to_csv(output_filename, index=False)
print(f"Aggregated yearly data saved to: {output_filename}")
