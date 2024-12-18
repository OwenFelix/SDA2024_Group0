"""
hashtag_analysis.py

DESCRIPTION:
This script analyses the hashtags used in tweets about the US presidential
candidates Donald Trump and Joe Biden. It creates a wordcloud of the hashtags
used in the tweets, filters out hashtags that contain the candidate's name,
and creates a time series of the hashtag frequency. It also calculates the
correlation between the sentiment polarity of the tweets and the total hashtag
frequency.

Note: This code was not used in the final presentation as it lacked relevance
for the scope of the project.
"""

import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from wordcloud import WordCloud  # For creating wordclouds
# For calculating weighted mean of sentiment polarity
from create_timeseries import weighted_mean
# For co-occurrence
from scipy.stats import pearsonr  # For calculating correlation
import warnings  # For handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings


def filter_hashtags(data, candidate):
    """
    Filter out tweets that do not contain other hashtags than the candidate's
    name and hashtags that contain the candidate's name.
    """
    # Extract hashtags from the tweet text
    data['hashtags'] = data['tweet'].apply(
        lambda x: [i for i in x.split() if i.startswith("#")])

    # Filter out tweets that do not contain other hashtags than the
    # candidate's name
    data['hashtags'] = data['hashtags'].apply(
        lambda x: [i for i in x if i.lower() != f'#{candidate.lower()}'])

    # Filter out hashtags that contain the candidate's name
    data.loc[:, 'hashtags'] = data['hashtags'].apply(
        lambda x: [i for i in x if candidate.lower() not in i.lower()])

    # Remove tweets that do not contain any hashtags
    data = data[data['hashtags'].apply(len) > 0]

    return data


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


def create_time_series_not_per_state(data, window_size):
    """
    Create a time series for the sentiment polarity of the tweets
    without grouping by state.
    """
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

    # Loop through each window and calculate the weighted mean sentiment
    # polarity
    for interval in rolling_data:
        intervals.append(weighted_mean(interval))

    # Make a list of tuples with the intervals and the corresponding dates
    tupled_intervals = list(zip(intervals, data.index))

    return tupled_intervals


def plot_sentiment_polarity_not_per_state(data, window_size, candidate):
    """
    This function plots the sentiment polarity of the tweets over time.
    This is done without grouping the tweets by state.
    """
    # Create time series for Joe Biden and Donald Trump
    intervals = create_time_series_not_per_state(data, window_size)

    # Plot the sentiment polarity
    plt.figure(figsize=(12, 6))
    plt.plot([i[1] for i in intervals], [i[0] for i in intervals],
             label='Sentiment Polarity', color='blue')
    plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'Sentiment Polarity of Tweets for {candidate}')
    plt.legend()
    plt.show()


def create_hashtag_time_series(data, window_size):
    """
    This creates a time series with the total hashtag frequency used
    over time windows.
    """
    # Convert the 'created_at' column to datetime
    data.loc[:, 'created_at'] = pd.to_datetime(data['created_at'])

    # Explode the hashtags to one per row
    data = data.explode('hashtags')

    # Group by time windows and calculate total frequency of hashtags
    time_series = (
        data.set_index('created_at')
        .groupby(pd.Grouper(freq=window_size))['hashtags']
        .value_counts()
        .reset_index(name='frequency')
        .groupby('created_at')['frequency']
        .sum()
        .reset_index(name='total_hashtag_frequency')
    )

    return time_series


def pearson_correlation_test(data, window_size):
    """
    Calculate the p-value for the correlation between sentiment polarity
    and total hashtag frequency.
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
        merged_data['total_hashtag_frequency']
    )

    return correlation, p_value


def main():
    # Get the tweet data for both candidates
    trump_tweets = pd.read_csv("../tmp/cleaned_hashtag_donaldtrump.csv")
    biden_tweets = pd.read_csv('../tmp/cleaned_hashtag_joebiden.csv')

    # Get the dataframes with the filtered tweets
    trump_hashtags = filter_hashtags(trump_tweets, 'Trump')
    biden_hashtags = filter_hashtags(biden_tweets, 'Biden')

    # Create wordclouds for the hashtags (vizualization purposes)
    create_wordcloud(trump_hashtags, 'Trump')
    create_wordcloud(biden_hashtags, 'Biden')

    _, trump_p_value = pearson_correlation_test(trump_hashtags, '24h')
    _, biden_p_value = pearson_correlation_test(biden_hashtags, '24h')

    print(f"Trump p-value: {trump_p_value}")
    if trump_p_value < 0.05:
        print("The correlation between sentiment and hashtag frequency \
for Trump is statistically significant.")
    else:
        print("The correlation between sentiment and hashtag frequency \
for Trump is not statistically significant.")

    print(f"Biden p-value: {biden_p_value}")
    if biden_p_value < 0.05:
        print("The correlation between sentiment and hashtag frequency \
for Biden is statistically significant.")
    else:
        print("The correlation between sentiment and hashtag frequency \
for Biden is not statistically significant.")

    plot_sentiment_polarity_not_per_state(trump_hashtags, '24h', 'Trump')
    plot_sentiment_polarity_not_per_state(biden_hashtags, '24h', 'Biden')


if __name__ == "__main__":
    main()
