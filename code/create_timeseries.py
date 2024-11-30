import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations


# Define the custom weighted mean function
def weighted_mean(x):
    """
    Calculate the weighted mean of sentiment polarity for a group.
    The weight of a single tweet is calculated by the formula:
    weight = #likes + 2 * #retweets

    :param group: DataFrame (a group of tweets within a time period)
    :return: Weighted mean of sentiment polarity
    """
    # Extract columns: likes, retweets, sentiment_polarity
    likes = x['likes']
    retweets = x['retweet_count']
    sentiment_polarity = x['sentiment_polarity']

    # Calculate the weight
    weight = np.log1p(likes) + 2 * np.log1p(retweets)
    # Make sure tweets with no likes or retweets aren't ignored
    weight = np.where(weight == 0, 1, weight)

    return np.average(sentiment_polarity, weights=weight)


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

plot_sentiment_polarity(biden_tweets, trump_tweets, 'Arizona', '24h')
