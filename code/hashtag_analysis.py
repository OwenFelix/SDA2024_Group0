import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from wordcloud import WordCloud  # For creating wordclouds
from create_timeseries import weighted_mean  # For calculating weighted mean of sentiment polarity

# Get the tweet data for both candidates
trump_tweets = pd.read_csv("../data/tweets/cleaned_hashtag_donaldtrump.csv")
biden_tweets = pd.read_csv('../data/tweets/cleaned_hashtag_joebiden.csv')


def filter_hashtags(data, candidate):
    """
    Filter out tweets that do not contain other hashtags than the candidate's name
    """
    # Extract hashtags from the tweet text
    data['hashtags'] = data['tweet'].apply(
        lambda x: [i for i in x.split() if i.startswith("#")])

    # Filter out tweets that do not contain other hashtags than the candidate's name
    data['hashtags'] = data['hashtags'].apply(
        lambda x: [i for i in x if i.lower() != f'#{candidate.lower()}'])

    # Remove tweets that do not contain any hashtags
    data = data[data['hashtags'].apply(len) > 0]

    return data

# Get the dataframes with the filtered tweets
trump_hashtags = filter_hashtags(trump_tweets, 'Trump')
biden_hashtags = filter_hashtags(biden_tweets, 'Biden')

def create_wordcloud(data, candidate):
    """
    Create a wordclou for the hashtags in the tweet data
    """
    # Combine all hashtags into a single string
    text = data['hashtags'].apply(lambda x: ' '.join(x)).str.cat(sep=' ')

    # Create the wordcloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='black').generate(text)

    # Display the wordcloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Wordcloud of Hashtags for {candidate}')
    plt.show()

# filter out hashtags that contain the string 'trump' or 'biden'
def filter_candidates(data):
    """
    Filter out hashtags that contain the string 'trump' or 'biden'
    """
    data.loc[:, 'hashtags'] = data['hashtags'].apply(
        lambda x: [i for i in x if 'trump' not in i.lower() and 'biden' not in i.lower()])

    return data

# Get the dataframes with the filtered tweets
trump_hashtags = filter_candidates(trump_hashtags)
biden_hashtags = filter_candidates(biden_hashtags)


# create_wordcloud(trump_hashtags, 'Trump')
# create_wordcloud(biden_hashtags, 'Biden')

def create_time_series_not_per_state(data, window_size):
    # Convert the 'created_at' column to datetime and set it as the index
    data.loc[:, 'created_at'] = pd.to_datetime(data['created_at'])
    data = data.set_index('created_at')
    # data = data.sort_values('created_at')

    # Sort by datetime index
    data = data.sort_index()

    # Ensure sentiment_polarity is in the correct format (float)
    data['sentiment_polarity'] = data['sentiment_polarity'].astype(float)

    # Create rolling windows based on the specified window size
    rolling_data = data.rolling(window_size)

    # Initialize a list to store processed intervals (weighted means)
    intervals = []

    # Loop through each window and calculate the weighted mean sentiment polarity
    for interval in rolling_data:
        intervals.append(weighted_mean(interval))

    # Make a list of tuples with the intervals and the corresponding dates
    tupled_intervals = list(zip(intervals, data.index))

    return tupled_intervals

def plot_sentiment_polarity_not_per_state(data, window_size):
    # Create time series for Joe Biden and Donald Trump
    intervals = create_time_series_not_per_state(data, window_size)

    # Plot the sentiment polarity
    plt.figure(figsize=(12, 6))
    plt.plot([i[1] for i in intervals], [i[0] for i in intervals],
             label='Sentiment Polarity', color='blue')
    plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title('Sentiment Polarity of Tweets')
    plt.legend()
    plt.show()

# Plan for the next steps:
# Create timeseries where we get the most popular hashtag used on a certain day
# Then find a way to link it to the events that happened on that day
pd.set_option('display.max_colwidth', None)
# Get the most popular hashtag for each day
def create_hashtag_time_series(data, window_size, candidate):
    """
    This creates a time-series where we get the most popular hashtag used
    on a certain for a certain state.
    """
    # Convert the 'created_at' column to datetime
    data.loc[:, 'created_at'] = pd.to_datetime(data['created_at'])

    data = data.explode('hashtags')

    # Group by time windows and find the most popular hashtag
    time_series = (
        data.set_index('created_at')
        .groupby(pd.Grouper(freq=window_size))['hashtags']
        .apply(lambda x: x.value_counts().idxmax() if not x.empty else None)
        .reset_index(name='most_popular_hashtag')
    )

    return time_series


plot_sentiment_polarity_not_per_state(trump_hashtags, '24h')
plot_sentiment_polarity_not_per_state(biden_hashtags, '24h')

trump_hashtags_timeseries = create_hashtag_time_series(trump_hashtags, '24h', 'Trump')
biden_hashtags_timeseries = create_hashtag_time_series(biden_hashtags, '24h', 'Biden')
print(trump_hashtags_timeseries)
print(biden_hashtags_timeseries)
print(trump_hashtags_timeseries.shape)
print(biden_hashtags_timeseries.shape)