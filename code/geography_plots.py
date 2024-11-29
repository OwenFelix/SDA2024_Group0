"""
HEADER
TODO : FILL WITH DESCRIPTION OF CONTENT OF FILE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely import affinity
import matplotlib.colors as mcolors

# Scale and translate Alaska and filter out the small islands
def fix_alaska(alaska_geom, scale, threshold = 1e10):
    alaska_scaled = affinity.scale(alaska_geom, xfact=scale, yfact=scale)
    alaska_moved = affinity.translate(alaska_scaled, xoff=-7000000, yoff=-6500000)

    filtered = [poly for poly in alaska_moved.geoms if poly.area > threshold]
    return MultiPolygon(filtered)

# Scale and translate Hawaii
def fix_hawaii(hawaii_geom, scale):
    hawaii_scaled = affinity.scale(hawaii_geom, xfact=scale, yfact=scale)

    hawaii_moved = affinity.translate(hawaii_scaled, xoff=6500000, yoff=500000)
    return hawaii_moved

# Make a plain US map
def generate_map():  
    # Read the shapefile for the US states
    states = gpd.read_file('../state_borders/cb_2018_us_state_500k.shp')
    states = states.to_crs("EPSG:3395")

    # Exclude these territories from the map
    non_continental = ['VI','MP','GU','AS','PR']
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
    # Read the voting data
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    trump_states = voting_results[voting_results['trump_win'] == 1]['state']
    biden_states = voting_results[voting_results['biden_win'] == 1]['state']

    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    # Set the colors for the states in the file
    states['COLOR'] = np.where(states['NAME'].isin(trump_states), republican_color, democrat_color)

    republican_patch = mpatches.Patch(color=republican_color, label='Trump-Winning States')
    democrat_patch = mpatches.Patch(color=democrat_color, label='Biden-Winning States')

    # Plot the map 
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    states.plot(color = states['COLOR'], linewidth = 0.6, edgecolor = 'black', legend = True, ax = ax)

    # Set the legend
    plt.legend(handles=[republican_patch, democrat_patch], loc='center left', bbox_to_anchor=(0.7, -0.1))
    plt.axis('off')
    plt.title('2020 Presidential Election Results by State')

    plt.show()



def make_example_dataset(states):
    state_codes = states['STUSPS'].tolist()
    
    for state in state_codes:
        # random float between -1 and 1
        states.loc[states['STUSPS'] == state, 'trump_sentiment'] = np.random.uniform(-1, 1)
        states.loc[states['STUSPS'] == state, 'biden_sentiment'] = np.random.uniform(-1, 1)

    return states

def get_sentiment_color(sentiment, color):
    # Compute the weight of the target color and white
    weight = (sentiment + 1) / 2  # Convert sentiment (-1 to 1) to a range of 0 to 1
    white = mcolors.to_rgb('white')
    target = mcolors.to_rgb(color)

    # Compute the resulting color as a linear interpolation
    result_color = tuple(((1 - weight) * white[i] + weight * target[i]) for i in range(3))
    return result_color


def plot_sentiment(states):
    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    # Create 2 plots, one for each candidate
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    for i, candidate in enumerate(['trump', 'biden']):
        # Set the colors for the states in the file
        states['COLOR'] = states[f'{candidate}_sentiment'].apply(lambda x: get_sentiment_color(x, republican_color if candidate == 'trump' else democrat_color))

        # Plot the map 
        states.plot(color = states['COLOR'], linewidth = 0.6, edgecolor = 'black', legend = False, ax = ax[i])

        ax[i].set_title(f'{candidate.capitalize()} Sentiment by State')
        ax[i].axis('off')

    plt.show()


# Plot the map
state_map = generate_map()
plot_election_results(state_map)

state_map_sent = make_example_dataset(state_map)
plot_sentiment(state_map_sent)








