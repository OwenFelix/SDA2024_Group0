"""
interactive_plot.py

DESCRIPTION:
This file contains the code to generate an interactive plot of the sentiment of each state for a given candidate over time.
The sentiment data is processed to the mean of each state and day and then plotted using Plotly Express.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from pandas import Timestamp


def process_sentiment_data(data, states, candidate):
    """
    Process the sentiment data to create a DataFrame with the sentiment of each state for each candidate over time.
    """
    state_codes = states['STUSPS'].tolist()
    all_states_mean = {}

    # Might find another way to do this, but it works for now
    all_dates = [Timestamp(2020, 10, 15), Timestamp(2020, 10, 16), Timestamp(2020, 10, 17), Timestamp(2020, 10, 18), Timestamp(2020, 10, 19), Timestamp(2020, 10, 20), Timestamp(2020, 10, 21), Timestamp(2020, 10, 22), Timestamp(2020, 10, 23), Timestamp(2020, 10, 24), Timestamp(2020, 10, 25), Timestamp(2020, 10, 26), Timestamp(
        2020, 10, 27), Timestamp(2020, 10, 28), Timestamp(2020, 10, 29), Timestamp(2020, 10, 30), Timestamp(2020, 10, 31), Timestamp(2020, 11, 1), Timestamp(2020, 11, 2), Timestamp(2020, 11, 3), Timestamp(2020, 11, 4), Timestamp(2020, 11, 5), Timestamp(2020, 11, 6), Timestamp(2020, 11, 7), Timestamp(2020, 11, 8)]
    all_dates = [x.date() for x in all_dates]

    # Loop over each state and store the sentiment scores and mean for each day
    for state in state_codes:
        state_data = data[state][candidate]
        state_dict = {}

        # Initialize the dictionary with empty lists
        state_dict = {i: [] for i, date in enumerate(
            all_dates) if date == all_dates[i]}

        # Loop over the data and store the sentiment for each day
        for (sent, timestamp) in state_data:
            for i in range(len(all_dates)):
                if timestamp.date() == all_dates[i]:
                    state_dict[i].append(sent)

        # Calculate mean if there is data for each day, if not store 0
        for day in state_dict:
            if state_dict[day] == []:
                all_states_mean[(state, day)] = 0
            else:
                all_states_mean[(state, day)] = np.mean(state_dict[day])

    return all_states_mean


def make_time_series_dataset_real(states, n_timestamps, mean_data, candidate):
    """
    Generate random time-series sentiment data for each state.
    """
    state_codes = states['STUSPS'].tolist()
    time_series_data = []

    # Loop over each state and timestamp and note the sentiment
    for state in state_codes:
        for t in range(n_timestamps):
            time_series_data.append({
                'state': state,
                'timestamp': t,
                f'{candidate}_sentiment': mean_data[(state, t)],
                'state_name': states[states['STUSPS'] == state]['NAME'].values[0]
            })

    return pd.DataFrame(time_series_data)


def plot_with_slider_plotly(data, candidate):
    """
    Plot the sentiment of each state for a given candidate over time using Plotly
    """
    # Min and max value in data
    min_val = data[f'{candidate}_sentiment'].min()
    max_val = data[f'{candidate}_sentiment'].max()
    zero_val = (0 - min_val) / (max_val - min_val)

    fig = px.choropleth(
        data,
        locations="state",
        locationmode="USA-states",
        color=f'{candidate}_sentiment',
        hover_name="state_name",
        animation_frame="timestamp",
        color_continuous_scale=[(0, "Black"), (zero_val, "White"), (1, "Blue")] if candidate == "biden" else [
            (0, "Black"), (zero_val, "White"), (1, "Red")],
        range_color=(min_val, max_val),
        scope="usa",
        title="Biden Tweet Sentiment Over Time" if candidate == "biden" else "Trump Tweet Sentiment Over Time"
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Sentiment"))

    # Do not plot the lakes
    fig.update_geos(showlakes=False, lakecolor="white", landcolor="white")
    fig.show()


# Load the voting data to get the state names and abbreviations
voting_results = pd.read_csv('../data/election_results/voting.csv')

# Make dataframe of abreviations and names
states = pd.DataFrame({'STUSPS': list(
    voting_results['state_abr']), 'NAME': list(voting_results['state'])})
n_timestamps = 25

data = pickle.load(open('../tmp/timeseries.pkl', 'rb'))

mean_data_trump = process_sentiment_data(data, states, 'trump')
mean_data_biden = process_sentiment_data(data, states, 'biden')

# Generate the time series data
time_series_data_trump = make_time_series_dataset_real(
    states, n_timestamps, mean_data_trump, "trump")
time_series_data_biden = make_time_series_dataset_real(
    states, n_timestamps, mean_data_biden, "biden")

# Call the function to plot the data in separate plots
plot_with_slider_plotly(time_series_data_trump, "trump")
plot_with_slider_plotly(time_series_data_biden, "biden")
