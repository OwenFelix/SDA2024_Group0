import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely import affinity

def fix_alaska(alaska_geom, threshold = 1e10):
    # Scale Alaska's geometry (reduce size by 40% for both x and y axes)
    alaska_scaled = affinity.scale(alaska_geom, xfact=0.3, yfact=0.3)

    # Translate Alaska (move it to the lower-left of the contiguous U.S.)
    alaska_moved = affinity.translate(alaska_scaled, xoff=-7000000, yoff=-6500000)

    # Filter out polygons smaller than the minimum area
    filtered = [poly for poly in alaska_moved.geoms if poly.area > threshold]
    return MultiPolygon(filtered)

def fix_hawaii(hawaii_geom):
    # Scale Hawaii's geometry (reduce size by 40% for both x and y axes)
    hawaii_scaled = affinity.scale(hawaii_geom, xfact=15, yfact=15)

    # Translate Hawaii (move it to the lower-left of the contiguous U.S.)
    hawaii_moved = affinity.translate(hawaii_scaled, xoff=-3000000, yoff=-1400000)
    return hawaii_moved
    
  
states = gpd.read_file('../state_borders/cb_2018_us_state_500k.shp')
states = states.to_crs("EPSG:3395")

non_continental = ['VI','MP','GU','AS','PR']
us49 = states
for n in non_continental:
    us49 = us49[us49.STUSPS != n]

# Get data for alaska
alaska = us49.loc[us49['STUSPS'] == 'AK', 'geometry'].values[0]
alaska = fix_alaska(alaska)

hawaii = us49.loc[us49['STUSPS'] == 'HI', 'geometry'].values[0]
hawaii = fix_hawaii(hawaii)

# Replace Alaska's geometry in the GeoDataFrame
us49.loc[us49['STUSPS'] == 'AK', 'geometry'] = alaska

voting_results = pd.read_csv('../data/election_results/voting.csv')

trump_states = voting_results[voting_results['trump_win'] == 1]['state']
biden_states = voting_results[voting_results['biden_win'] == 1]['state']

republican_color = '#FF0803'
democrat_color = '#0000FF'

us49['COLOR'] = np.where(us49['NAME'].isin(trump_states), republican_color, democrat_color)

republican_patch = mpatches.Patch(color=republican_color, label='Trump-Winning States')
democrat_patch = mpatches.Patch(color=democrat_color, label='Biden-Winning States')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
us49.plot(color = us49['COLOR'], linewidth = 0.5, edgecolor = 'black', legend = True, ax=ax)

plt.legend(handles=[republican_patch, democrat_patch], loc='center left', bbox_to_anchor=(0.7, -0.1))
plt.axis('off')
plt.title('2020 Presidential Election Results by State')

plt.show()

