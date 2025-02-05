from datasets import load_dataset
import pandas as pd
#  Download data for the year 1805 at the associated article level (Default)
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years",
    year_list=["1805"]
)

"""# Download and process data for all years at the article level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years"
)

# Download and process data for 1805 at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "subset_years_content_regions",
    year_list=["1805"]
)

# Download ad process data for all years at the scan level
dataset = load_dataset("dell-research-harvard/AmericanStories",
    "all_years_content_regions")
"""
print(type(dataset))
print(dataset.keys())

print(type(dataset['1805']))
print(dataset['1805'])
df = dataset['1805'].to_pandas()
print(df.columns)
print(df.head())
print(df.loc[df['article_id'] == '15_1805-08-26_p2_sn83016082_00332895060_1805082601_0684']['newspaper_name'])
print(df.loc[df['article_id'] == '15_1805-08-26_p2_sn83016082_00332895060_1805082601_0684']['edition'])
print(df.loc[df['article_id'] == '15_1805-08-26_p2_sn83016082_00332895060_1805082601_0684']['date'])


print(df.loc[df['article_id'] == '15_1805-08-26_p2_sn83016082_00332895060_1805082601_0684']['article'].values)
#print(df['newspaper_name'].unique())

#Date out of 365ish, let the model know if this impacts the current year
#