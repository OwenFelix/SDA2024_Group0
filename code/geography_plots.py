import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import MultiPolygon
from shapely import affinity

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
def plot_us_map():  
    # Read the shapefile for the US states
    states = gpd.read_file('../state_borders/cb_2018_us_state_500k.shp')
    states = states.to_crs("EPSG:3395")

    # Exclude these territories from the map
    non_continental = ['VI','MP','GU','AS','PR']
    us49 = states
    for n in non_continental:
        us49 = us49[us49.STUSPS != n]

    # Get data for alaska
    alaska = us49.loc[us49['STUSPS'] == 'AK', 'geometry'].values[0]
    alaska = fix_alaska(alaska, 0.3)

    # Get data for hawaii
    hawaii = us49.loc[us49['STUSPS'] == 'HI', 'geometry'].values[0]
    hawaii = fix_hawaii(hawaii, 2)

    # Replace Alaska's and Hawaii's geometry in the GeoDataFrame
    us49.loc[us49['STUSPS'] == 'AK', 'geometry'] = alaska
    us49.loc[us49['STUSPS'] == 'HI', 'geometry'] = hawaii

    return us49


# Assign the correct color to each state and plot the map
def plot_election_results(us49):
    # Read the voting data
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    trump_states = voting_results[voting_results['trump_win'] == 1]['state']
    biden_states = voting_results[voting_results['biden_win'] == 1]['state']

    republican_color = '#FF0803'
    democrat_color = '#0000FF'

    # Set the colors for the states in the file
    us49['COLOR'] = np.where(us49['NAME'].isin(trump_states), republican_color, democrat_color)

    republican_patch = mpatches.Patch(color=republican_color, label='Trump-Winning States')
    democrat_patch = mpatches.Patch(color=democrat_color, label='Biden-Winning States')

    # Plot the map 
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    us49.plot(color = us49['COLOR'], linewidth = 0.6, edgecolor = 'black', legend = True, ax = ax)

    # Set the legend
    plt.legend(handles=[republican_patch, democrat_patch], loc='center left', bbox_to_anchor=(0.7, -0.1))
    plt.axis('off')
    plt.title('2020 Presidential Election Results by State')

    plt.show()

# Plot the map
us49 = plot_us_map()
# plot_election_results(us49)


