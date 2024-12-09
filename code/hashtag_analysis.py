"""
hashtag_analysis.py

DESCRIPTION:

"""

import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
from wordcloud import WordCloud  # For creating wordclouds
import re  # For regular expressions
# For calculating weighted mean of sentiment polarity
from create_timeseries import weighted_mean
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")

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


def create_hashtag_time_series(data, window_size):
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


def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags)


def create_co_occurence_matrix(data):
    hashtag_series = data['tweet'].apply(extract_hashtags)
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 1),  stop_words='english')
    count_vec = count_vectorizer.fit_transform(hashtag_series)
    co_occurence_matrix = (count_vec.T @ count_vec)
    co_occurence_df = pd.DataFrame(co_occurence_matrix.toarray(
    ), columns=count_vectorizer.get_feature_names_out(), index=count_vectorizer.get_feature_names_out())
    np.fill_diagonal(co_occurence_df.values, 0)
    return co_occurence_df


# trump_co_occurence_df = create_co_occurence_matrix(trump_hashtags)
# biden_co_occurence_df = create_co_occurence_matrix(biden_hashtags)


def plot_co_occurence(co_occorence_df, candidate):
    plt.figure(figsize=(25, 5))
    co_occorence_df.sum(axis=1).sort_values(
        ascending=False).head(25).plot(kind='bar')
    plt.ylabel('Count')
    plt.xlabel('Hashtags')
    plt.title(f'Top 25 Co-occurring Hashtags for {candidate}')
    plt.show()


def pearson_correlation_test(data, window_size):
    """
    Calculate the p-value for the correlation between sentiment polarity and hashtag frequency.
    """
    # Create time series for sentiment polarity
    sentiment_intervals = create_time_series_not_per_state(data, window_size)

    # Create time series for hashtag frequency
    hashtag_time_series = create_hashtag_time_series(data, window_size)

    # Merge the two time series on the date
    merged_data = pd.merge(
        pd.DataFrame(sentiment_intervals, columns=[
                     'sentiment_polarity', 'date']),
        hashtag_time_series,
        left_on='date',
        right_on='created_at',
        how='inner'
    )

    # Calculate the correlation and p-value
    correlation, p_value = pearsonr(
        merged_data['sentiment_polarity'],
        merged_data['most_popular_hashtag'].apply(
            lambda x: data['hashtags'].explode().value_counts().get(x, 0))
    )

    return correlation, p_value


_, trump_p_value = pearson_correlation_test(trump_hashtags, '24h')
_, biden_p_value = pearson_correlation_test(biden_hashtags, '24h')

if trump_p_value < 0.05:
    print("The correlation between sentiment and hashtag frequency for Trump is statistically significant.")
else:
    print("The correlation between sentiment and hashtag frequency for Trump is not statistically significant.")

if biden_p_value < 0.05:
    print("The correlation between sentiment and hashtag frequency for Biden is statistically significant.")
else:
    print("The correlation between sentiment and hashtag frequency for Biden is not statistically significant.")


# plot_co_occurence(trump_co_occurence_df, 'Trump')
# plot_co_occurence(biden_co_occurence_df, 'Biden')

# plot_sentiment_polarity_not_per_state(trump_hashtags, '24h')
# plot_sentiment_polarity_not_per_state(biden_hashtags, '24h')

# trump_hashtags_timeseries = create_hashtag_time_series(trump_hashtags, '24h', 'Trump')
# biden_hashtags_timeseries = create_hashtag_time_series(biden_hashtags, '24h', 'Biden')
# print(trump_hashtags_timeseries)
# print(biden_hashtags_timeseries)
# print(trump_hashtags_timeseries.shape)
# print(biden_hashtags_timeseries.shape)
