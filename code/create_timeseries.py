"""
create_timeseries.py

DESCRIPTION:
This script creates time series data for each state based on the sentiment
polarity of tweets. The sentiment polarity is calculated using a weighted
mean of sentiment polarity for a group of tweets within a time period. The
weights are based on the number of likes and retweets of each tweet. We
can use this data to plot the sentiment polarity of tweets for both
candidates in each state.
"""

import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import pickle  # For saving the model
import warnings  # For handling warnings
warnings.filterwarnings('ignore')  # Ignore warnings



def weighted_mean(x, sigma=4, alpha=2, gaussian_kernel=True):
    """
    Calculate the weighted mean of sentiment polarity for a group using
    the number of likes and retweets as weights.

    :param x: DataFrame (a group of tweets within a time period)
    :return: Weighted mean of sentiment polarity
    """
    # Extract columns
    likes = x['likes']
    retweets = x['retweet_count']
    sentiment_polarity = x['sentiment_polarity']

    # Calculate sentiment-based weights
    sentiment_weight = 2 * np.log1p(likes) + 5 * np.log1p(retweets)
    sentiment_weight = np.where(sentiment_weight == 0, 1, sentiment_weight)

    if not gaussian_kernel:
        return np.sum(sentiment_polarity * sentiment_weight) /  \
            np.sum(sentiment_weight)

    # Calculate Gaussian kernel weights
    positions = np.arange(len(x))
    center = len(x) // 2
    gaussian_kernel = np.exp(-0.5 * ((positions - center) / sigma) ** 2)

    # Combine sentiment-based and Gaussian kernel weights
    sentiment_weight = alpha * sentiment_weight + (1 - alpha) * gaussian_kernel

    # Normalize weights
    sentiment_weight /= np.sum(sentiment_weight)

    # Calculate the weighted mean of sentiment polarity
    return np.sum(sentiment_polarity * sentiment_weight)


def create_timeseries(data, state_code, window_size, gaussian_kernel=True):
    """
    Create time series data for a given state based on the sentiment
    polarity of tweets.
    """
    tweets = data[data['state_code'] == state_code]

    # Convert the 'created_at' column to datetime and set it as the index
    tweets.loc[:, 'created_at'] = pd.to_datetime(tweets['created_at'])
    tweets = tweets.set_index('created_at')

    # Sort by datetime index
    tweets = tweets.sort_index()

    # Ensure sentiment_polarity is in the correct format (float)
    tweets['sentiment_polarity'] = tweets['sentiment_polarity'].astype(float)

    # Create rolling windows based on the specified window size
    rolling_tweets = tweets.rolling(window_size)

    # Initialize a list to store processed intervals (weighted means)
    intervals = []
    gaussian_free_intervals = []

    # Loop through each window and calculate the weighted mean sentiment
    # polarity
    for interval in rolling_tweets:
        intervals.append(weighted_mean(
            interval, gaussian_kernel=True))
        gaussian_free_intervals.append(weighted_mean(
            interval, gaussian_kernel=False))

    # Make a list of tuples with the intervals and the corresponding dates
    tupled_intervals = list(zip(intervals, tweets.index))
    tupled_intervals_no_gaussian = list(
        zip(gaussian_free_intervals, tweets.index))

    return intervals, tweets, tupled_intervals, tupled_intervals_no_gaussian


def plot_sentiment_polarity(biden_data, trump_data, state_code, window_size):
    """
    Plot the sentiment polarity of tweets for Joe Biden and Donald Trump
    """
    # Create time series for Joe Biden and Donald Trump
    biden_intervals, tweets_biden, _, __ = create_timeseries(
        biden_data, state_code, window_size)
    trump_intervals, tweets_trump, _, __ = create_timeseries(
        trump_data, state_code, window_size)

    # Plot the sentiment polarity
    plt.figure(figsize=(12, 6))
    plt.plot(tweets_biden.index, biden_intervals,
             label='Joe Biden', color='blue')
    plt.plot(tweets_trump.index, trump_intervals,
             label='Donald Trump', color='red')
    plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'Sentiment Polarity of Tweets in {state_code}')
    plt.legend()
    plt.show()


def main():
    # Load the datasets
    biden_tweets = pd.read_csv("../tmp/cleaned_hashtag_joebiden.csv")
    trump_tweets = pd.read_csv("../tmp/cleaned_hashtag_donaldtrump.csv")

    # Get the states codes
    voting_results = pd.read_csv('../data/election_results/voting.csv')
    state_codes = voting_results['state_abr'].tolist()

# Create time series data for each state
timeseries = {}
timeseries_no_gaussian = {}
for state in state_codes:
    _, __, b_tuples1, b_tuples2 = create_timeseries(biden_tweets, state, '24h')
    _, __, t_tuples1, t_tuples2 = create_timeseries(trump_tweets, state, '24h')
    timeseries[state] = {'biden': b_tuples1,
                         'trump': t_tuples1}
    timeseries_no_gaussian[state] = {'biden': b_tuples2,
                                     'trump': t_tuples2}

# Save the timeseries data
with open('tmp/timeseries.pkl', 'wb') as f:
    pickle.dump(timeseries, f)

# Save the timeseries data without Gaussian kernel
with open('tmp/timeseries_no_gaussian.pkl', 'wb') as f:
    pickle.dump(timeseries_no_gaussian, f)
