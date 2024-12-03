import numpy as np
import pandas as pd
import plotly.express as px
import geopandas as gpd

def make_time_series_dataset(states, n_timestamps):
    """
    Generate random time-series sentiment data for each state.
    """
    state_codes = states['STUSPS'].tolist()
    time_series_data = []

    for state in state_codes:
        for t in range(n_timestamps):
            time_series_data.append({
                'state': state,
                'timestamp': t,
                'trump_sentiment': np.random.uniform(-1, 1),
                'biden_sentiment': np.random.uniform(-1, 1),
                'state_name': states[states['STUSPS'] == state]['NAME'].values[0]
            })

    return pd.DataFrame(time_series_data)


# Plot with Plotly
def plot_with_slider_plotly(data, candidate):
    """
    Plot the sentiment of each state for a given candidate over time using Plotly
    """
    fig = px.choropleth(
        data,
        locations="state",  # State codes
        locationmode="USA-states",
        color=f'{candidate}_sentiment',
        hover_name="state_name",
        animation_frame="timestamp",
        color_continuous_scale="Reds" if candidate == "trump" else "Blues",
        range_color=(-1, 1),
        scope="usa",
        title= "Biden sentiment over time" if candidate == "biden" else "Trump sentiment over time"
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Sentiment"))

    # Do not plot the lakes
    fig.update_geos(showlakes=False, lakecolor="white", landcolor="white")
    fig.show()

# Load the voting data to get the state names and abbreviations
voting_results = pd.read_csv('../data/election_results/voting.csv')

# Make dataframe of abreviations and names
states = pd.DataFrame({'STUSPS': list(voting_results['state_abr']), 'NAME': list(voting_results['state'])})
n_timestamps = 100
time_series_data = make_time_series_dataset(states, n_timestamps)

# Call the function to plot
plot_with_slider_plotly(time_series_data, "biden")
