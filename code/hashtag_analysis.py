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


create_wordcloud(trump_hashtags, 'Trump')
create_wordcloud(biden_hashtags, 'Biden')
