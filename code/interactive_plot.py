import numpy as np
import pandas as pd
import plotly.express as px

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
                'biden_sentiment': np.random.uniform(-1, 1)
            })

    return pd.DataFrame(time_series_data)

# Example input dataset
states = pd.DataFrame({'STUSPS': ['AL', 'AK', 'AZ', 'AR', 'CA']})
n_timestamps = 10
time_series_data = make_time_series_dataset(states, n_timestamps)

# Plot with Plotly
def plot_with_slider_plotly(data):
    fig = px.choropleth(
        data,
        locations="state",  # State codes
        locationmode="USA-states",
        color="trump_sentiment",
        hover_name="state",
        animation_frame="timestamp",
        color_continuous_scale="RdBu",
        range_color=(-1, 1),
        scope="usa",
    )
    fig.update_layout(title="Trump Sentiment by State Over Time", coloraxis_colorbar=dict(title="Sentiment"))
    fig.show()

# Call the function to plot
plot_with_slider_plotly(time_series_data)
