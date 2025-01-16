from datasets import load_dataset
import pandas as pd
#  Download data for the year 1900 at the associated article level (Default)
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years",
    year_list=["1900"]
)

"""# Download and process data for all years at the article level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years"
)

# Download and process data for 1900 at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years_content_regions",
    year_list=["1900"]
)

# Download ad process data for all years at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years_content_regions")
"""
print(type(dataset))
print(dataset.keys())

print(type(dataset['1900']))
print(dataset['1900'])
df = dataset['1900'].to_pandas()
print(df.columns)
print(df.head())

#print(df['newspaper_name'].unique())

#Date out of 365ish, let the model know if this impacts the current year
#