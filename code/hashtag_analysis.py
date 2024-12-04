import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from wordcloud import WordCloud  # For creating wordclouds

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


# create_wordcloud(trump_hashtags, 'Trump')
# create_wordcloud(biden_hashtags, 'Biden')

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


    # data = data.set_index('created_at')
    # data = data.sort_index()

    # Create a rolling windows based on the specified window size
    # rolling_data = data.rolling(window_size)
    # print()

    # # Initialize a list to store the most popular hashtags
    # hashtags = []

    # # Loop through each window and get the most popular hashtag
    # for window in rolling_data:
    #     # print(window['hashtags'].value_counts().idxmax())
    #     hashtags.append(window['hashtags'].value_counts().idxmax())

    # # Get the date and hashtags columns
    # hashtags = data[['created_at', 'hashtags']].copy()

    # # Create a new column with the most popular hashtag for each day
    # hashtags['most_popular_hashtag'] = hashtags['hashtags'].apply(
    #     lambda x: pd.Series(x).value_counts().idxmax())

    # # Group by date and get the most popular hashtag
    # hashtags = hashtags.groupby('created_at')['most_popular_hashtag'].apply(
    #     lambda x: x.value_counts().idxmax()).reset_index()

    # Set the date as the index
    # hashtags = hashtags.set_index('created_at')

    # return hashtags

trump_hashtags_timeseries = create_hashtag_time_series(trump_hashtags, '24h', 'Trump')
biden_hashtags_timeseries = create_hashtag_time_series(biden_hashtags, '24h', 'Biden')
print(trump_hashtags_timeseries)
print(biden_hashtags_timeseries)
# print(trump_hashtags_timeseries.shape)

