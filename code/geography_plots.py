"""
geography_plots.py

DESCRIPTION:
This file contains the code to make a geographical plot of the US states using the Matplotlib and Geopandas libraries.
The code generates a map of the US states and plots the results of the 2020 presidential election by state.
This code also generates random time-series sentiment data for each state and plots it in an interactive plot with a slider to change the timestamp.
The interactive plot is not used to plot the real sentiment data, because the plot was too slow, and we used Plotly instead.
See the code in interactive_plot.py for the real sentiment data plotting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely import affinity
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import sys


def fix_alaska(alaska_geom, scale, threshold=1e10):
    """
    Scale and translate Alaska to fit on the map, and filter out the small islands.
    """
    alaska_scaled = affinity.scale(alaska_geom, xfact=scale, yfact=scale)
    alaska_moved = affinity.translate(
        alaska_scaled, xoff=-7000000, yoff=-6500000)

    filtered = [poly for poly in alaska_moved.geoms if poly.area > threshold]
    return MultiPolygon(filtered)


def fix_hawaii(hawaii_geom, scale):
    """
    Scale and translate Hawaii to fit on the map.
    """
    hawaii_scaled = affinity.scale(hawaii_geom, xfact=scale, yfact=scale)

    hawaii_moved = affinity.translate(hawaii_scaled, xoff=6500000, yoff=500000)
    return hawaii_moved


def generate_map():
    """
    Generate a map of the US states and clean up the geometry for Alaska and Hawaii.
    """
    # Read the shapefile for the US states
    states = gpd.read_file('../state_borders/cb_2018_us_state_500k.shp')
    states = states.to_crs("EPSG:3395")

    # Exclude these territories from the map
    non_continental = ['VI', 'MP', 'GU', 'AS', 'PR']
    for n in non_continental:
        states = states[states.STUSPS != n]

    # Get data for alaska
    alaska = states.loc[states['STUSPS'] == 'AK', 'geometry'].values[0]
    alaska = fix_alaska(alaska, 0.3)

    # Get data for hawaii
    hawaii = states.loc[states['STUSPS'] == 'HI', 'geometry'].values[0]
    hawaii = fix_hawaii(hawaii, 2)

    # Replace Alaska's and Hawaii's geometry in the GeoDataFrame
    states.loc[states['STUSPS'] == 'AK', 'geometry'] = alaska
    states.loc[states['STUSPS'] == 'HI', 'geometry'] = hawaii

    return states


# Assign the correct color to each state and plot the map
def plot_election_results(states):
    """
    Plot the results on a map of the US.
    """
    # Read the voting data
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    trump_states = voting_results[voting_results['trump_win'] == 1]['state']

    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    # Set the colors for the states in the file
    states['COLOR'] = np.where(states['NAME'].isin(
        trump_states), republican_color, democrat_color)

    republican_patch = mpatches.Patch(
        color=republican_color, label='Trump-Winning States')
    democrat_patch = mpatches.Patch(
        color=democrat_color, label='Biden-Winning States')

    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    states.plot(color=states['COLOR'], linewidth=0.6,
                edgecolor='black', legend=True, ax=ax)

    # Set the legend
    plt.legend(handles=[republican_patch, democrat_patch],
               loc='center left', bbox_to_anchor=(0.7, -0.1))
    plt.axis('off')
    plt.title('2020 Presidential Election Results by State')

    plt.show()


def plot_sentiment_results(states, trump_data, biden_data):
    """
    Plot the sentiment results on a map of the US and compare them to the real election results.
    """
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    biden_states = []
    trump_states = []

    for state in voting_results['state_abr']:
        biden_sentiment = biden_data[biden_data['state']
                                    == state]['biden_sentiment'].mean()
        trump_sentiment = trump_data[trump_data['state']
                                    == state]['trump_sentiment'].mean()

        if biden_sentiment > trump_sentiment:
            biden_states.append(state)

        elif trump_sentiment > biden_sentiment:
            trump_states.append(state)

    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    states_sentiment = states.copy()

    print("Trump states:", trump_states)
    print("Biden states:", biden_states)

    # Set the colors for the states in the file
    states_sentiment['COLOR'] = np.where(states['STUSPS'].isin(
        trump_states), republican_color, democrat_color)
    
    trump_states_real = voting_results[voting_results['trump_win'] == 1]['state']
    states['COLOR'] = np.where(states['NAME'].isin(
        trump_states_real), republican_color, democrat_color)

    republican_patch = mpatches.Patch(
        color=republican_color, label='Trump-Winning States')
    democrat_patch = mpatches.Patch(
        color=democrat_color, label='Biden-Winning States')
    
    
    # Plot the map
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    # plot the sentiment left and the real results on the right
    states.plot(color=states_sentiment['COLOR'], linewidth=0.6,
                edgecolor='black', legend=False, ax=ax[0])
    

    states.plot(color=states['COLOR'], linewidth=0.6,
                edgecolor='black', legend=False, ax=ax[1])
    
    # Set the legend
    plt.legend(handles=[republican_patch, democrat_patch],
                loc='center left', bbox_to_anchor=(0.7, -0.1))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('Sentiment Results by State')
    ax[1].set_title('Real Results by State')

    plt.show()


def make_example_dataset(states, trump_data, biden_data):
    """
    Generate random sentiment data for each state for both candidates.
    """
    state_codes = states['STUSPS'].tolist()

    for state in state_codes:
        # random float between -1 and 1
        states.loc[states['STUSPS'] == state,
                   'trump_sentiment'] = trump_data[trump_data['state'] == state]['trump_sentiment'].values[0]
        states.loc[states['STUSPS'] == state,
                   'biden_sentiment'] = biden_data[biden_data['state'] == state]['biden_sentiment'].values[0]

    return states


def make_time_series_dataset(states, n_timestamps):
    """
    Generate random time-series sentiment data for each state.
    """
    state_codes = states['STUSPS'].tolist()

    time_series_data = []

    for state in state_codes:
        state_data = {
            'state': state,
            'trump_sentiment': np.random.uniform(-1, 1, n_timestamps),
            'biden_sentiment': np.random.uniform(-1, 1, n_timestamps)
        }
        time_series_data.append(state_data)

    return pd.DataFrame(time_series_data)


def make_time_series_dataset_plotly(states, n_timestamps):
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


def get_sentiment_color(sentiment, color):
    """
    Compute a color based on the sentiment value and a target color.
    """
    weight = (sentiment + 1) / 2
    white = mcolors.to_rgb('white')
    target = mcolors.to_rgb(color)

    # Compute the resulting color as a linear interpolation
    result_color = tuple(
        ((1 - weight) * white[i] + weight * target[i]) for i in range(3))
    return result_color


def plot_sentiment(states):
    """
    Plot the sentiment of each state for each candidate on separate plots.
    """
    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    # Create 2 plots, one for each candidate
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    for i, candidate in enumerate(['trump', 'biden']):
        # Set the colors for the states in the file
        states['COLOR'] = states[f'{candidate}_sentiment'].apply(lambda x: get_sentiment_color(
            x, republican_color if candidate == 'trump' else democrat_color))

        # Plot the map
        states.plot(color=states['COLOR'], linewidth=0.6,
                    edgecolor='black', legend=False, ax=ax[i])

        ax[i].set_title(f'{candidate.capitalize()} Sentiment by State')
        ax[i].axis('off')

    plt.show()


def plot_time_series(states, time_series_data, n_timestamps):
    """
    Plot a time series of sentiment data for each state, with a slider to change the timestamp.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    plt.subplots_adjust(bottom=0.25)

    trump_color = '#FF0803'
    biden_color = '#0000FF'

    # Function to plot the map at a given timestamp
    def get_time_series(t):
        # UNCOMMENT THIS FOR SINGLE PLOT
        # states['COLOR'] = [get_sentiment_color(x[t], "#FF0803") for x in time_series_data['trump_sentiment']]
        # states.plot(color = states['COLOR'], linewidth = 0.6, edgecolor = 'black', legend = False, ax = ax)
        # ax.axis('off')
        # ax.set_title('Trump Sentiment by State')

        # For two plots
        for i, candidate in enumerate(['trump', 'biden']):
            # Set the colors for the states in the file
            if candidate == 'trump':
                color = trump_color
            else:
                color = biden_color
            states['COLOR'] = [get_sentiment_color(
                x[t], color) for x in time_series_data[f'{candidate}_sentiment']]
            # Plot the map
            states.plot(color=states['COLOR'], linewidth=0.6,
                        edgecolor='black', legend=False, ax=ax[i])

            ax[i].set_title(f'{candidate.capitalize()} Sentiment by State')
            ax[i].axis('off')

    # Plot the initial time series
    get_time_series(0)

    # Set the axis and slider position in the plot
    axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                             facecolor='white')
    slider_position = Slider(axis_position,
                             'Timestamp', valmin=0, valmax=n_timestamps-1, valstep=1, valinit=0)

    # Update function for when the slider is moved

    def update(val):
        pos = slider_position.val
        get_time_series(int(pos))
        fig.canvas.draw_idle()

    # Update function called using on_changed() function
    slider_position.on_changed(update)

    plt.show()


# Plot the map
state_map = generate_map()

# Plot election results onto map
plot_election_results(state_map)

# Plot time series sentiment data onto map (random data)
n_timestamps = 25
# state_map_ts = make_time_series_dataset(state_map, n_timestamps)
# plot_time_series(state_map, state_map_ts, n_timestamps)

# Read data from csv
time_series_biden = pd.read_csv('../tmp/time_series_data_biden.csv')
time_series_trump = pd.read_csv('../tmp/time_series_data_trump.csv')

# only sentiment data from timestamp 19
time_series_biden = time_series_biden[time_series_biden['timestamp'] == 19]
time_series_trump = time_series_trump[time_series_trump['timestamp'] == 19]

plot_sentiment_results(state_map, time_series_trump, time_series_biden)

# Plot single sentiment values onto map (random data)
# state_map_sent = make_example_dataset(
#     state_map, time_series_trump, time_series_biden)
# plot_sentiment(state_map_sent)
