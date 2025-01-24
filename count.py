import os
import re
import pandas as pd
from datasets import load_dataset

def count_occurrences(text: str, word: str) -> int:
    """
    Counts how many times `word` appears in `text` (case-insensitive),
    matching whole words only.
    """
    # E.g. using regex word boundaries, ignoring case
    pattern = rf"\b{word}\b"
    return len(re.findall(pattern, text.lower()))

def main():
    # ------------------------------------------------
    # Configure your sampling & year range
    # ------------------------------------------------
    start_year = 1880
    end_year = 1890
    sample_percentage = 0.10  # 10%
    
    # Folder to store the CSV outputs
    output_folder = "yearly_smallpox_death_counts"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # ------------------------------------------------
    # Loop through each year & load data
    # ------------------------------------------------
    for year in range(start_year, end_year):
        year_str = str(year)
        print(f"Loading data for year {year_str}...")

        # Load from huggingface "dell-research-harvard/AmericanStories"
        # specifying "subset_years" with year_list=[year_str]
        dataset = load_dataset(
            "dell-research-harvard/AmericanStories",
            "subset_years",
            year_list=[year_str],
            trust_remote_code=True
        )
        df_year = dataset[year_str].to_pandas()

        if len(df_year) == 0:
            print(f"  No articles found for year {year_str}, skipping.")
            continue

        # Sample 10%
        sample_size = int(len(df_year) * sample_percentage)
        if sample_size == 0:
            print(f"  Sample size is 0 for year {year_str}, skipping.")
            continue
        
        df_sampled = df_year.sample(n=sample_size, random_state=42)
        print(f"  Sampled {len(df_sampled)} articles from {len(df_year)} total")

        # ------------------------------------------------
        # Count "smallpox" and "death"/"deaths" occurrences
        # ------------------------------------------------
        counts_list = []
        for _, row in df_sampled.iterrows():
            article_id = row["article_id"]
            article_text = row["article"] if "article" in row else ""
            
            smallpox_count = count_occurrences(article_text, "smallpox")
            # Combine "death" and "deaths" into one total
            death_count = count_occurrences(article_text, "death") + count_occurrences(article_text, "deaths")
            
            counts_list.append({
                "article_id": article_id,
                "smallpox_count": smallpox_count,
                "death_count": death_count
            })
        
        # ------------------------------------------------
        # Save results to CSV
        # ------------------------------------------------
        df_counts = pd.DataFrame(counts_list)
        output_path = os.path.join(output_folder, f"{year_str}_smallpox_death_counts.csv")
        df_counts.to_csv(output_path, index=False)
        
        print(f"  Saved {len(df_counts)} rows to {output_path}\n")


if __name__ == "__main__":
    main()
