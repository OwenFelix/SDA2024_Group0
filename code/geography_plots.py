import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
  

states = gpd.read_file('../tl_2024_us_state/tl_2024_us_state.shp')
states = states.to_crs("EPSG:3395")

non_continental = ['HI','VI','MP','GU','AK','AS','PR']
us49 = states
for n in non_continental:
    us49 = us49[us49.STUSPS != n]

voting_results = pd.read_csv('../data/election_results/voting.csv')

trump_states = voting_results[voting_results['trump_win'] == 1]['state']
biden_states = voting_results[voting_results['biden_win'] == 1]['state']

republican_color = '#FF0803'
democrat_color = '#0000FF'

us49['COLOR'] = np.where(us49['NAME'].isin(trump_states), republican_color, democrat_color)

us49.plot(color = us49['COLOR'], linewidth = 0.5, edgecolor = 'black', legend = True)

plt.axis('off')
plt.title('2020 Presidential Election Results by State')

plt.show()

