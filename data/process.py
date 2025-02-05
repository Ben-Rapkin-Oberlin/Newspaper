import pandas as pd

# Read and filter the mortality data
df = pd.read_csv("data/master_mortality_data_1900_1936_ucb.csv")
df = df[["state_name", "year", "total_mortality", "small_pox"]]
states = ["Maine", "Connecticut", "Indiana", "Massachusetts", "Michigan", 
          "New Hampshire", "New Jersey", "Rhode Island", "New York"]
df = df[df["state_name"].isin(states)]

# Group by year and convert to DataFrame
df = df.groupby('year')["small_pox"].sum().to_frame()
df = df.reset_index()

# Read the state population data
dfstate = pd.read_csv("data/State Pops - Sheet1.csv")

# Reverse the ratio values and add them to df
reversed_ratios = dfstate["Total_Ratio"].values[:len(df)][::-1]  # [::-1] reverses the array
df['ratio'] = reversed_ratios


df['estimated_deaths'] = round(df['small_pox'] / df['ratio']).astype(int)
# Save to CSV
df.to_csv("data/death_estimates.csv", index=False)