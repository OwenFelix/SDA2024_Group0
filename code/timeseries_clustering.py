"""
timeseries_clustering.py

DESCRIPTION:

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import permutation_test


def load_timeseries_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_state_color_map():
    return {
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


def get_state_groups(state_color_map):
    """
    Get the states in the blue and red groups.
    """
    blue_states = [state for state in state_color_map.keys()
                   if state_color_map[state] == 'blue']
    red_states = [state for state in state_color_map.keys()
                  if state_color_map[state] == 'red']
    return blue_states, red_states


def prepare_features(timeseries_features, blue_states, red_states):
    """
    Prepare the features for clustering by first getting the
    timeseries features for the blue and red states and then
    normalizing the features.
    """
    X = np.array([list(timeseries_features[state].values())
                 for state in blue_states + red_states])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pca = PCA()
    return pca.fit_transform(X)


def perform_clustering(X, n_clusters=2, random_state=567):
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    kmeans.fit(X)
    return kmeans.predict(X), kmeans


def calculate_accuracy(cluster_assignments, blue_states, red_states):
    true_labels = [0] * len(blue_states) + [1] * len(red_states)
    accuracy = np.mean(cluster_assignments == true_labels)
    return accuracy, true_labels


def cluster_distinction_test(X, cluster_assignments):
    """Test for distinct clustering using the silhouette score."""
    return silhouette_score(X, cluster_assignments)


def average_kmeans_clustering(X, n_iter=100):
    """
    Cluster X n_iter times with differing seeds and return the
    average cluster assignments.
    """
    cluster_assignments = np.zeros((n_iter, X.shape[0]))

    # Perform clustering n_iter times
    for i in range(n_iter):
        cluster_assignments[i], _ = perform_clustering(X, random_state=i * 10)
    mean_clustering = np.mean(cluster_assignments, axis=0)

    #  Clamp each element to 0 or 1
    mean_clustering[mean_clustering < 0.5] = 0
    mean_clustering[mean_clustering >= 0.5] = 1
    return mean_clustering


def random_clustering_test(X, cluster_assignments, true_labels,
                           n_permutations=1000):
    """Permutation test for comparing clustering to random assignments."""
    def statistic(cluster_assignments, true_labels):
        return np.mean(cluster_assignments == true_labels)

    data = (cluster_assignments, true_labels)
    p_value = permutation_test(
        data, statistic, permutation_type='pairings',
        n_resamples=n_permutations, alternative='greater'
    ).pvalue

    return p_value


def plot_clusters(X, cluster_assignments, true_labels):
    # Create a custom legend
    legend_labels = ['Biden', 'Trump']
    legend_handles = [plt.Line2D([0], [0], marker='o',
                                 color='w', markerfacecolor='blue',
                                 markersize=10),
                      plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='red', markersize=10)]
    # Plot for KMeans Clustering
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='coolwarm',
                label='Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering of Blue and Red States')
    plt.legend(legend_handles, legend_labels)

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='coolwarm',
                label='True Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('True Labels of Blue and Red States')
    plt.legend(legend_handles, legend_labels)
    plt.show()


def main():
    timeseries_features = load_timeseries_data('../data/features.pkl')
    state_color_map = get_state_color_map()
    blue_states, red_states = get_state_groups(state_color_map)
    X = prepare_features(timeseries_features, blue_states, red_states)
    cluster_assignments, _ = perform_clustering(X)

    for state, cluster in zip(blue_states + red_states, cluster_assignments):
        print(f'{state}: {cluster}')

    average_clustering = average_kmeans_clustering(X)
    accuracy, true_labels = calculate_accuracy(
        average_clustering, blue_states, red_states)
    print(f'Accuracy: {accuracy}')

    # Test the hypotheses
    silhouette_avg = cluster_distinction_test(X, average_clustering)
    p_value = random_clustering_test(X, average_clustering, true_labels)
    print(f'Silhouette Score: {silhouette_avg}')
    print(f'P-value: {p_value}')

    # Plot Clusters
    plot_clusters(X, average_clustering, true_labels)

    if silhouette_avg > 0.5:
        print('Reject the null hypothesis that'
              'the clusters are not distinct.')
    else:
        print('Fail to reject the null hypothesis'
              'that the clusters are not distinct.')

    if p_value < 0.05:
        print('Reject the null hypothesis that the'
              'clustering is not better than random.')
    else:
        print('Fail to reject the null hypothesis that'
              'the clustering is not better than random.')


if __name__ == "__main__":
    main()
