import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
data = pd.read_csv('master_mortality_data_1900_1936_ucb.csv')

# Fill missing small_pox values with 0
data['small_pox'] = data['small_pox'].fillna(0)

# Total smallpox cases per year
total_per_year = data.groupby('year')['small_pox'].sum().reset_index()

# Filter out records with zero cases
reported_data = data[data['small_pox'] > 0]

# Find the first year each state reported cases
start_years = reported_data.groupby('state_name')['year'].min().reset_index()
start_years = start_years.rename(columns={'year': 'start_year'})
# Create the figure
fig = go.Figure()

# Add total cases line
fig.add_trace(go.Scatter(
    x=total_per_year['year'],
    y=total_per_year['small_pox'],
    mode='lines+markers',
    name='Total Smallpox Cases',
    line=dict(color='blue'),
    marker=dict(size=6)
))

# Add start year lines for each state
for idx, row in start_years.iterrows():
    fig.add_vline(
        x=row['start_year'],
        line=dict(color='orange', width=1, dash='dash'),
        opacity=0.5,
        annotation_text=row['state_name'],
        annotation_position="top left",
        annotation=dict(font_size=10),
    )

# Customize the layout
fig.update_layout(
    title='Total Smallpox Cases Over Time (1900-1936) with State Reporting Start Years',
    xaxis_title='Year',
    yaxis_title='Number of Smallpox Cases',
    template='plotly_white',
    hovermode='x unified',
    width=1200,
    height=600
)

# Show the plot
fig.show()
