import os
import pandas as pd

# Base directory where the CSV files are stored.
base_path = r"window_5_10.0"

# List to hold the aggregated data for each year.
aggregated_data = []


start=1900
end=1936
# Loop over the years of interest.
for year in range(start, end+1):  # 1900 to 1936 inclusive
    print(year)
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

    # Total word count across articles (if available).
    total_word_count = df['word_count'].sum() if 'word_count' in df.columns else None

    # Aggregate co-occurrence counts per topic.
    # We'll sum up all columns that start with "cooccur_topicX_"
    topic_totals = {}
    for topic in range(1, 6):
        regex_pattern = f"^cooccur_topic{topic}_"
        # Select columns for the topic.
        topic_cols = df.filter(regex=regex_pattern).columns
        if not topic_cols.empty:
            # Sum all the selected columns (first sum returns a Series, then sum the Series).
            topic_total = df[topic_cols].sum().sum()
        else:
            topic_total = 0
        topic_totals[f"topic{topic}_total"] = topic_total

    # Sum the smallpox death co-occurrences.
    total_smallpox_death_cooccurrences = (
        df['smallpox_death_cooccurrences'].sum()
        if 'smallpox_death_cooccurrences' in df.columns
        else None
    )

    # Compute the average topic probabilities per topic.
    topic_avg_probs = {}
    for topic in range(1, 6):
        col_name = f"topic_{topic}_prob"
        if col_name in df.columns:
            topic_avg_probs[f"topic{topic}_avg_prob"] = df[col_name].mean()
        else:
            topic_avg_probs[f"topic{topic}_avg_prob"] = None

    # Combine all the aggregated information into one dictionary.
    aggregated_data.append({
        "year": year,
        "n_articles": n_articles,
        "total_word_count": total_word_count,
        **topic_totals,
        **topic_avg_probs,
        "total_smallpox_death_cooccurrences": total_smallpox_death_cooccurrences
    })

# Convert the list of dictionaries into a DataFrame.
agg_df = pd.DataFrame(aggregated_data)

# Optionally, sort the DataFrame by year.
agg_df = agg_df.sort_values(by="year")


if start>=1900:
    print("including truth")
    a="training_data"
    # Read the ground truth death estimates.
    death_df = pd.read_csv(r"/usr/users/quota/students/2021/brapkin/Newspaper/data/death_estimates.csv")
    print("\nDeath estimates loaded. Sample:")
    print(death_df.head())

    # ------------------------------
    # Merge Datasets
    # ------------------------------

    # We only need the 'year' and 'estimated_deaths' columns from death_df.
    death_subset = death_df[['year', 'estimated_deaths']]

    # Merge on the 'year' column. Using an inner join here ensures we only keep years present in both files.
    final_df = pd.merge(agg_df, death_subset, on='year', how='inner')
else:
    print("no truth")
    a="pred_data"
    final_df=agg_df
print("\nMerged training set sample:")
print(final_df.head())

# Save the aggregated data to a new CSV file.
#output_file = os.path.join(base_path, "yearly_aggregated_data_1900_1936.csv")
final_df.to_csv(f"yearly_occurrence_data\{a}_{start}_{end}.csv", index=False)
print(f"Aggregated yearly data saved to: yearly_occurrence_data\{a}_{start}_{end}.csv")
