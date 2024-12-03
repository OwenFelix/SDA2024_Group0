import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import pickle  # For saving the time series data


def weighted_mean(x, sigma=4, alpha=2):
    """
    Calculate the weighted mean of sentiment polarity for a group, using a combination of sentiment-based
    weights and a Gaussian kernel.

    :param x: DataFrame (a group of tweets within a time period)
    :param sigma: Standard deviation for the Gaussian kernel
    :param alpha: Balance factor for sentiment vs. Gaussian weights
    :return: Weighted mean of sentiment polarity
    """
    # Extract columns
    likes = x['likes']
    retweets = x['retweet_count']
    sentiment_polarity = x['sentiment_polarity']

    # Calculate sentiment-based weights
    sentiment_weight = 2 * np.log1p(likes) + 5 * np.log1p(retweets)
    sentiment_weight = np.where(sentiment_weight == 0, 1, sentiment_weight)

    # Calculate Gaussian kernel weights
    positions = np.arange(len(x))
    center = len(x) // 2
    gaussian_kernel = np.exp(-0.5 * ((positions - center) / sigma) ** 2)

    # Combine sentiment and Gaussian weights
    weight = alpha * sentiment_weight + (1 - alpha) * gaussian_kernel
    weight /= np.sum(weight)  # Normalize weights

    # Calculate the weighted mean of sentiment polarity
    return np.sum(sentiment_polarity * weight)


def create_timeseries(data, state, window_size):
    tweets = data[data['state'] == state]

    # Convert the 'created_at' column to datetime and set it as the index
    tweets['created_at'] = pd.to_datetime(tweets['created_at'])
    tweets = tweets.set_index('created_at')

    # Sort by datetime index
    tweets = tweets.sort_index()

    # Ensure sentiment_polarity is in the correct format (float)
    tweets['sentiment_polarity'] = tweets['sentiment_polarity'].astype(float)

    # Create rolling windows based on the specified window size
    rolling_tweets = tweets.rolling(window_size)

    # Initialize a list to store processed intervals (weighted means)
    intervals = []

    # Loop through each window and calculate the weighted mean sentiment polarity
    for interval in rolling_tweets:
        intervals.append(weighted_mean(interval))

    return intervals, tweets


def plot_sentiment_polarity(biden_data, trump_data, state, window_size):
    # Create time series for Joe Biden and Donald Trump
    biden_intervals, tweets_biden = create_timeseries(
        biden_data, state, window_size)
    trump_intervals, tweets_trump = create_timeseries(
        trump_data, state, window_size)

    # Plot the sentiment polarity
    plt.figure(figsize=(12, 6))
    plt.plot(tweets_biden.index, biden_intervals,
             label='Joe Biden', color='blue')
    plt.plot(tweets_trump.index, trump_intervals,
             label='Donald Trump', color='red')
    plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'Sentiment Polarity of Tweets in {state}')
    plt.legend()
    plt.show()


# Load the biden dataset
biden_tweets = pd.read_csv("../data/tweets/cleaned_hashtag_joebiden.csv")
trump_tweets = pd.read_csv("../data/tweets/cleaned_hashtag_donaldtrump.csv")

timeseries = []
# get the five biggest red states and the five biggest blue states
states = ['California', 'New York', 'Illinois', 'Washington', 'Michigan',
          'Texas', 'Florida', 'Georgia', 'Ohio', 'North Carolina']

for state in states:
    # Save tuples of two time series (rump and biden) for each state
    timeseries.append((create_timeseries(biden_tweets, state, '24h'),
                      create_timeseries(trump_tweets, state, '24h')))

# Save the time series in a file
with open('../data/timeseries.pkl', 'wb') as f:
    pickle.dump(timeseries, f)
