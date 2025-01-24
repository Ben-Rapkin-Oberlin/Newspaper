import os
import re
import pandas as pd

def unify_small_pox_variants(text: str) -> str:
    """
    Converts "small-pox", "small pox", etc. to "smallpox" (case-insensitive).
    Ensures all variants become a single token "smallpox".
    """
    return re.sub(r'\bsmall[\-\s]+pox\b', 'smallpox', text, flags=re.IGNORECASE)

def count_occurrences(text: str, word: str) -> int:
    """
    Counts how many times `word` appears in `text` (case-insensitive),
    matching whole words only.
    """
    pattern = rf"\b{word}\b"  # word boundary
    return len(re.findall(pattern, text.lower()))

def main():
    # ----------------------------------------------------------------------
    # Adjust these parameters to match your main script's settings
    # ----------------------------------------------------------------------
    start_year = 1810
    end_year = 1820
    window_size = 5  # or whatever you used in the main script
    
    # The folder where your main script saved sample_{year}.csv
    sample_input_dir = f"sampled_data/window_{window_size}"

    # Output folder for our mention counts
    output_folder = "yearly_smallpox_death_counts_same_sample"
    os.makedirs(output_folder, exist_ok=True)
    
    # ----------------------------------------------------------------------
    # Loop through each year, reading EXACT same sampled CSV from main script
    # ----------------------------------------------------------------------
    for year in range(start_year, end_year):
        csv_path = os.path.join(sample_input_dir, f"sample_{year}.csv")
        if not os.path.exists(csv_path):
            print(f"Sample file not found for year {year}: {csv_path}")
            continue
        
        df_sampled = pd.read_csv(csv_path)
        if df_sampled.empty:
            print(f"No data in {csv_path}, skipping.")
            continue

        print(f"\nLoaded {len(df_sampled)} sampled articles for year {year} from {csv_path}")

        # ------------------------------------------------------------------
        # For each article, unify small pox => smallpox, then count mentions
        # ------------------------------------------------------------------
        counts_list = []
        for _, row in df_sampled.iterrows():
            article_id = row.get("article_id", "")
            article_text = row.get("article", "")

            # 1) unify "small-pox" => "smallpox"
            article_text = unify_small_pox_variants(article_text)

            # 2) count literal "smallpox" mentions
            smallpox_count = count_occurrences(article_text, "smallpox")

            # 3) sum of "death" + "deaths"
            death_count = (
                count_occurrences(article_text, "death") +
                count_occurrences(article_text, "deaths")
            )

            counts_list.append({
                "article_id": article_id,
                "smallpox_count": smallpox_count,
                "death_count": death_count
            })

        # ------------------------------------------------------------------
        # Build DataFrame of results & save
        # ------------------------------------------------------------------
        df_counts = pd.DataFrame(counts_list)
        output_path = os.path.join(output_folder, f"{year}_smallpox_death_counts.csv")
        df_counts.to_csv(output_path, index=False)
        print(f"Saved {len(df_counts)} rows with mention counts to {output_path}")

if __name__ == "__main__":
    main()
