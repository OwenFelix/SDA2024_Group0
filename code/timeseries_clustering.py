import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load in the timeseries data
with open('../data/features.pkl', 'rb') as f:
    timeseries_features = pickle.load(f)


state_color_map = {
    'CA': 'blue', 'NY': 'blue', 'IL': 'blue', 'WA': 'blue',
    'MI': 'swing', 'TX': 'red', 'FL': 'swing', 'GA': 'red',
    'OH': 'swing', 'NC': 'swing', 'PA': 'swing', 'AZ': 'swing',
    'MA': 'blue', 'NJ': 'blue', 'VA': 'blue', 'TN': 'red',
    'IN': 'red', 'MO': 'red', 'MD': 'blue', 'WI': 'swing',
    'MN': 'swing', 'CO': 'swing', 'AL': 'red', 'SC': 'red',
    'LA': 'red', 'KY': 'red', 'OR': 'blue', 'CT': 'blue',
    'IA': 'swing', 'MS': 'red', 'AR': 'red', 'UT': 'red',
    'NV': 'swing', 'KS': 'red', 'NM': 'swing', 'NE': 'red',
    'ID': 'red', 'WV': 'red', 'HI': 'blue', 'ME': 'swing',
    'NH': 'swing', 'MT': 'red', 'RI': 'blue', 'DE': 'blue',
    'ND': 'red', 'AK': 'red', 'VT': 'blue', 'WY': 'red'
}

# Use the timeseries to get the five biggest blue states and the five biggest red states
blue_states = [state for state in state_color_map.keys()
               if state_color_map[state] == 'blue']
red_states = [state for state in state_color_map.keys()
              if state_color_map[state] == 'red']


# First write the features as a matrix
X = np.array([list(timeseries_features[state].values())
              for state in blue_states + red_states])

# standardize the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Initialize the KMeans model
kmeans = KMeans(n_clusters=2, random_state=567)

# Fit the model to the features
kmeans.fit(X)

# Get the cluster assignments
cluster_assignments = kmeans.predict(X)

# # print the cluster assignments
for state, cluster in zip(blue_states + red_states, cluster_assignments):
    print(f'{state}: {cluster}')

# print the accuracy
true_labels = [0] * len(blue_states) + [1] * len(red_states)
accuracy = np.mean(cluster_assignments == true_labels)
print(f'Accuracy: {accuracy}')

# Plot the clusters. Make label 0 blue and label 1 red
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='coolwarm')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering of Blue and Red States')

# Plot the clusters with the true labels (blue being 0 and red being 1) next to them
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='coolwarm')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('True Labels of Blue and Red States')
plt.show()
