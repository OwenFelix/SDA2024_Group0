"""
timeseries_dtwclustering.py

DESCRIPTION:
This script performs dynamic time warping clustering on the sentiment time
series data for the US presidential candidates Donald Trump and Joe Biden.
The script loads the sentiment time series data from a pickle file, filters
the states based on their political leaning, computes the dynamic time warping
(DTW) matrix for the sentiment time series data, performs hierarchical
clustering on the DTW matrix, and evaluates the clustering results using
accuracy, silhouette score, and permutation tests.
"""

import numpy as np  # For numerical operations
import pickle  # For loading the model
import tslearn.metrics  # For time series analysis
import scipy.cluster.hierarchy  # For hierarchical clustering
from sklearn.metrics import silhouette_score  # For measuring cluster quality
from scipy.stats import permutation_test  # For hypothesis testing


def get_state_color_map():
    """Returns a mapping of states to their political leaning colors."""
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


def load_timeseries_data(filepath):
    """Loads the timeseries data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def filter_states_by_color(timeseries, state_color_map, colors):
    """
    Filters states based on specified colors
    and returns their timeseries data.
    """
    return {state: timeseries[state] for state, color
            in state_color_map.items() if color in colors}


def compute_sentiment_timeseries(robust_states, candidate):
    """Extracts sentiment time series data for a given candidate."""
    return {
        state: np.array(data[candidate])[:, 0]
        for state, data in robust_states.items()
    }


def compute_dtw_matrix(states, sentiment_data):
    """Computes the DTW matrix for the given states and sentiment data."""
    n_states = len(states)
    dtw_matrix = np.zeros((n_states, n_states))
    state_list = list(states)

    for i, state1 in enumerate(state_list):
        for j, state2 in enumerate(state_list):
            if i > j:
                dtw_matrix[i, j] = dtw_matrix[j, i]
            else:
                dtw_matrix[i, j] = tslearn.metrics.dtw(
                    sentiment_data[state1], sentiment_data[state2]
                )
    return dtw_matrix


def cluster_states(dtw_matrix, n_clusters=2):
    """Performs hierarchical clustering on the DTW matrix."""
    linkage_matrix = scipy.cluster.hierarchy.linkage(dtw_matrix)
    return scipy.cluster.hierarchy.fcluster(linkage_matrix,
                                            n_clusters, criterion='maxclust')


def calculate_accuracy(cluster_assignments, blue_states, red_states):
    """This function calculates the accuracy of the clustering."""
    true_labels = [0] * len(blue_states) + [1] * len(red_states)
    accuracy = np.mean(cluster_assignments == true_labels)
    return accuracy, true_labels


def cluster_distinction_test(X, cluster_assignments):
    """Test for distinct clustering using the silhouette score."""
    return silhouette_score(X, cluster_assignments)


def calculate_p_value(cluster_assignments, true_labels,
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


def random_clustering_test(X, cluster_assignments, true_labels, n_iter=100):
    """
    First adds noise to the dtw_matrix, and then perform the
    permutation test n_iter times on the matrix with each time new noise.
    """
    n_states = X.shape[0]
    p_values = np.zeros(n_iter)
    for i in range(n_iter):
        noise = np.random.normal(0, 0.1, (n_states, n_states))
        noisy_matrix = X + noise
        cluster_assignments = cluster_states(noisy_matrix)
        p_values[i] = calculate_p_value(cluster_assignments, true_labels)
    return p_values


def main():
    # Step 1: Load data and prepare state data
    timeseries = load_timeseries_data('../tmp/timeseries.pkl')
    state_color_map = get_state_color_map()

    # Step 2: Filter states
    robust_states = filter_states_by_color(
        timeseries, state_color_map, colors=['blue', 'red']
    )

    # Step 3: Precompute sentiment time series
    sentiment_biden = compute_sentiment_timeseries(robust_states, 'biden')
    sentiment_trump = compute_sentiment_timeseries(robust_states, 'trump')

    # Step 4: Compute DTW matrices
    dtw_matrix_biden = compute_dtw_matrix(robust_states, sentiment_biden)
    dtw_matrix_trump = compute_dtw_matrix(robust_states, sentiment_trump)

    # Step 5: Combine DTW matrices, standarize and cluster
    dtw_matrix = dtw_matrix_biden + dtw_matrix_trump
    dtw_matrix = (dtw_matrix - np.mean(dtw_matrix)) / np.std(dtw_matrix)
    cluster_labels = cluster_states(dtw_matrix)

    # Step 6: Calculate accuracy
    blue_states, red_states = get_state_groups(state_color_map)
    accuracy, true_labels = calculate_accuracy(
        cluster_labels, blue_states, red_states)

    print(f'Accuracy: {accuracy}')

    # Step 7: Test the hypotheses
    silhouette_avg = cluster_distinction_test(dtw_matrix, cluster_labels)
    p_value = random_clustering_test(dtw_matrix, cluster_labels, true_labels)
    print(f'Silhouette Average: {silhouette_avg}')
    print(f'P-value: {p_value.mean()}')

    # Step 8: Print the hypothesis test results
    if silhouette_avg > 0.5:
        print('Reject the null hypothesis that '
              'the clusters are not distinct.')
    else:
        print('Fail to reject the null hypothesis '
              'that the clusters are not distinct.')

    if p_value.mean() < 0.05:
        print('Reject the null hypothesis that the '
              'clustering is not better than random.')
    else:
        print('Fail to reject the null hypothesis that '
              'the clustering is not better than random.')


if __name__ == '__main__':
    main()
