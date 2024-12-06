import numpy as np  # For numerical operations
import pickle  # For loading the model
import fastdtw  # For Dynamic Time Warping
import scipy.stats as sp  # For statistical analysis
import tslearn.metrics  # For time series analysis


def difference_in_mean_sentiment(sentiment1, sentiment2):
    """
    Calculate the difference in mean sentiment between two timeseries.
    """
    return np.mean(sentiment1) - np.mean(sentiment2)


def std_ratio(sentiment1, sentiment2):
    """
    Calculate the ratio of the standard deviations of two timeseries.
    """
    return np.std(sentiment1) / np.std(sentiment2)


def dtw_cumdiff(sentiment1, sentiment2):
    """
    Calculate the cumulative difference between two timeseries
    using Dynamic Time Warping (DTW).
    """
    print(len(sentiment1), len(sentiment2))
    # Step 1: Apply DTW to the two time series
    _, path = fastdtw.fastdtw(sentiment1, sentiment2)

    # Step 2: Get the aligned series
    aligned_sentiment1 = np.array([sentiment1[i] for i, j in path])
    aligned_sentiment2 = np.array([sentiment2[j] for i, j in path])

    # Step 3: Calculate the cumulative difference between the aligned series
    cumdiff = np.cumsum(np.abs(aligned_sentiment1 - aligned_sentiment2))

    return cumdiff  # Return the final cumulative difference value


def dtw_auc(sentiment1, sentiment2):
    """
    Calculate the Area Under Curve (AUC) of two timeseries using DTW.
    """
    # Step 1: Apply DTW to the two time series
    _, path = fastdtw.fastdtw(sentiment1, sentiment2)

    # Step 2: Interpolate the aligned series onto a common time grid
    aligned_sentiment1 = np.array([sentiment1[i] for i, j in path])
    aligned_sentiment2 = np.array([sentiment2[j] for i, j in path])

    # Step 3: Calculate the AUC of the aligned series
    # Use trapezoidal integration to calculate the area under the curve
    auc_sentiment1 = np.trapz(aligned_sentiment1)
    auc_sentiment2 = np.trapz(aligned_sentiment2)

    return auc_sentiment1, auc_sentiment2


def skewness(sentiment):
    """
    Calculate the skewness of a timeseries.
    """
    return sp.skew(sentiment)


def kurtosis(sentiment):
    """
    Calculate the kurtosis of a timeseries.
    """
    return sp.kurtosis(sentiment)


def cross_correlation(sentiment1, sentiment2, lag=0):
    """
    Calculate the cross-correlation coefficient between
    two time series at a given lag.
    """
    sentiment1 = np.array(sentiment1)
    sentiment2 = np.array(sentiment2)

    if lag > 0:
        sentiment1 = sentiment1[lag:]
        sentiment2 = sentiment2[:-lag]
    elif lag < 0:
        sentiment1 = sentiment1[:lag]
        sentiment2 = sentiment2[-lag:]

    # Normalize the series
    sentiment1 = (sentiment1 - np.mean(sentiment1)) / np.std(sentiment1)
    sentiment2 = (sentiment2 - np.mean(sentiment2)) / np.std(sentiment2)

    # Compute cross-correlation coefficient
    return np.correlate(sentiment1, sentiment2)[0] / len(sentiment1)


def dtw_distance(sentiment1, sentiment2):
    """
    Calculate the Dynamic Time Warping (DTW) distance between two timeseries.
    """
    return tslearn.metrics.dtw(sentiment1, sentiment2)


def ks_test(sentiment1, sentiment2):
    """
    Perform a Kolmogorov-Smirnov test to compare two timeseries.
    """
    return sp.ks_2samp(sentiment1, sentiment2)


def cumdiff_regression(sentiment1, sentiment2):
    """
    Use linear regression to approximate the slope of the cumulative difference
    between two timeseries.
    """
    # Calculate the cumulative difference between the two timeseries
    cumdiff = dtw_cumdiff(sentiment1, sentiment2)

    # Create a time vector
    time = np.arange(len(cumdiff))

    # Perform linear regression
    slope, _ = np.polyfit(time, cumdiff, 1)

    return slope


# Load in the timeseries data
with open('../data/timeseries.pkl', 'rb') as f:
    timeseries = pickle.load(f)

# Extract features from the timeseries
features = dict()

for state in timeseries.keys():
    print(f'Extracting features for {state}...')
    """
    Extract features from the timeseries for each state.
    The following features will be extracted of the two timeseries per state:
    - difference in mean sentiment
    - std ratio of the two timeseries
    - cumulative difference in sentiment
    - AUC (Area Under Curve) of the two timeseries
    - skewness of the two timeseries
    - kurtosis of the two timeseries
    - cross-correlation between the two timeseries
    - DTW distance between the two timeseries
    - ks-test value between the two timeseries
    - slope of the cumulative difference between the two timeseries
    """
    # First extract the timeseries for the state
    biden = timeseries[state]['biden']
    trump = timeseries[state]['trump']

    biden_sentiment = [x[0] for x in biden]
    trump_sentiment = [x[0] for x in trump]

    # If any of the timeseries are empty, skip the state
    if len(biden_sentiment) == 0 or len(trump_sentiment) == 0:
        continue

    # Calculate difference in mean sentiment
    mean_diff = difference_in_mean_sentiment(biden_sentiment, trump_sentiment)

    # Calculate std ratio
    std_r = std_ratio(biden_sentiment, trump_sentiment)

    # Calculate cumulative difference in sentiment
    # cumdiff = dtw_cumdiff(biden_sentiment, trump_sentiment)[-1]

    # Calculate AUC
    # auc_biden, auc_trump = dtw_auc(biden_sentiment, trump_sentiment)

    # Calculate skewness
    skew_biden = skewness(biden_sentiment)
    skew_trump = skewness(trump_sentiment)

    # Calculate kurtosis
    kurt_biden = kurtosis(biden_sentiment)
    kurt_trump = kurtosis(trump_sentiment)

    # Calculate cross-correlation
    cross_corr = cross_correlation(biden_sentiment, trump_sentiment)

    # Calculate DTW distance
    dtw_dist = dtw_distance(biden_sentiment, trump_sentiment)

    # Calculate ks-test value
    ks_statistic, _ = ks_test(biden_sentiment, trump_sentiment)

    # Calculate slope of cumulative difference
    slope = cumdiff_regression(biden_sentiment, trump_sentiment)

    # Vectorize the features
    features[state] = {'mean_diff': mean_diff,
                       'std_ratio': std_r,
                       #    'cumdiff': cumdiff,
                       #    'auc_biden': auc_biden,
                       #    'auc_trump': auc_trump,
                       'skew_biden': skew_biden,
                       'skew_trump': skew_trump,
                       'kurt_biden': kurt_biden,
                       'kurt_trump': kurt_trump,
                       'cross_corr': cross_corr,
                       'dtw_dist': dtw_dist,
                       'ks': ks_statistic,
                       'slope': slope}


# Save the features to a pickle file
with open('../data/features.pkl', 'wb') as f:
    pickle.dump(features, f)
