"""
data_preprocessing.py:

DESCRIPTION:
This script is used to preprocess the data from the hashtag_donaldtrump.csv and
hashtag_joebiden.csv files. The preprocessing steps include:
1. Dropping irrelevant columns
2. Filtering out tweets that are not from the United States of America
3. Dropping rows with missing values
4. Removing tweets that are in both datasets
5. Detecting the language of the tweets and only saving the English tweets
6. Cleaning the tweets
7. Performing the polarity sentiment analysis on the tweets
"""

# Importing the required libraries
import pandas as pd  # For data manipulation
import re  # For regular expressions
import emoji  # For handling emojis
import string  # For handling string operations
from nltk.corpus import stopwords  # For stopwords
from nltk.corpus import wordnet  # For POS tagging
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
from textblob import TextBlob  # For sentiment analysis
import langid  # For language detection

import nltk  # For natural language processing
nltk.download('stopwords')  # Download the stopwords
nltk.download('wordnet')  # Download the wordnet
nltk.download('averaged_perceptron_tagger_eng')  # Download the POS tagger


def read_data(file_path):
    """
    Read the data from the given file path
    """
    data = pd.read_csv(file_path, chunksize=1000, lineterminator='\n')
    chunk_list = []
    for chunk in data:
        chunk_list.append(chunk)
    return pd.concat(chunk_list, axis=0)


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def remove_stopwords(text):
    """
    Remove stopwords from the text except negation words and at
    the same time lemmatize the words in the text so that the words are
    in their root form.
    """
    lm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words -= {"'no', 'nor', 'not', 'don't, 'aren't', 'couldn't, 'didn't, \
                   'doesn't', 'hadn't', 'hasn't', 'haven't', 'isn't', \
                   'mightn't', 'mustn't', 'needn't', 'shan't', 'shouldn't', \
                   'wasn't', 'weren't', 'won't', 'wouldn't'"}
    return " ".join(lm.lemmatize(word, get_wordnet_pos(word)) for word in
                    text.split() if word not in stop_words)


def pre_langdetect_clean(text):
    """
    This function removes the usernames and URLs from the text data before
    language detection.
    """
    # Remove usernames from the text
    text = re.sub(r'@\S+', '', text)

    # Remove URLs entirely
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    return text


def clean_tweet_data(text):
    """
    This function cleans the text data by removing HTML tags, usernames,
    URLs, emojis, punctuation, numbers, stopwords and lemmatizing the words.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace emoji with text for the expression of the emoji
    text = emoji.demojize(text)

    # Remove punctuation, but keep !, ?, # and @
    punctuation_to_replace = ''.join(
        c for c in string.punctuation if c not in "!?#")
    text = re.sub(f"[{re.escape(punctuation_to_replace)}]", " ", text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove newline characters
    text = re.sub('\n', '', text)

    # Remove the stopwords and lemmatize the words
    text = remove_stopwords(text)

    # Remove extra spaces
    text = re.sub(' +', ' ', text)

    # Remove words of lenghth less than 2
    text = ' '.join([word for word in text.split() if len(word) > 2])

    return text


def get_language(text):
    """
    This function detects the language of the text.
    """
    try:
        return langid.classify(text)[0]
    except Exception as _:
        return 'unknown'


# Load the datasets
trump_tweets = read_data("../data/tweets/hashtag_donaldtrump.csv")
biden_tweets = read_data("../data/tweets/hashtag_joebiden.csv")

# Initialize a list of irrelevant columns
irrelevant_cols = ['source', 'user_id', 'user_name', 'user_screen_name',
                   'user_description', 'user_join_date', 'city',
                   'collected_at']

# Drop irrelevant columns from the datasets
trump_tweets.drop(columns=irrelevant_cols, inplace=True)
biden_tweets.drop(columns=irrelevant_cols, inplace=True)

# Replace United States with United States of America
trump_tweets['country'] = trump_tweets['country'].replace(
    'United States', 'United States of America')
biden_tweets['country'] = biden_tweets['country'].replace(
    'United States', 'United States of America')

# Filter out tweets that are not from the United States of America
trump_tweets = trump_tweets[trump_tweets['country']
                            == 'United States of America'].copy()
biden_tweets = biden_tweets[biden_tweets['country']
                            == 'United States of America'].copy()

# Drop all rows with missing values
biden_tweets.dropna(axis=0, inplace=True)
trump_tweets.dropna(axis=0, inplace=True)

# Remove tweets that are in both datasets
tids = trump_tweets['tweet_id']
bids = biden_tweets['tweet_id']
ids_tweets_in_common = set(trump_tweets['tweet_id']).intersection(
    set(biden_tweets['tweet_id']))
trump_tweets = trump_tweets[~tids.isin(ids_tweets_in_common)]
biden_tweets = biden_tweets[~bids.isin(ids_tweets_in_common)]

# Apply the pre clean before detecting the language
trump_tweets['tweet'] = trump_tweets['tweet'].apply(pre_langdetect_clean)
biden_tweets['tweet'] = biden_tweets['tweet'].apply(pre_langdetect_clean)

# Detect language for all tweets in the dataset
trump_tweets['language'] = trump_tweets['tweet'].apply(
    lambda x: get_language(x))
biden_tweets['language'] = biden_tweets['tweet'].apply(
    lambda x: get_language(x))

# Only save the english tweets
trump_tweets = trump_tweets[trump_tweets['language'] == 'en']
biden_tweets = biden_tweets[biden_tweets['language'] == 'en']

# Clean the tweets
trump_tweets['tweet'] = trump_tweets['tweet'].apply(clean_tweet_data)
biden_tweets['tweet'] = biden_tweets['tweet'].apply(clean_tweet_data)

# Sentiment analysis of the tweets
trump_tweets['sentiment_polarity'] = trump_tweets['tweet'].apply(
    lambda x: TextBlob(x).sentiment.polarity)
biden_tweets['sentiment_polarity'] = biden_tweets['tweet'].apply(
    lambda x: TextBlob(x).sentiment.polarity)

# Save the new datasets to csv
trump_tweets.to_csv(
    '../data/tweets/cleaned_hashtag_donaldtrump.csv', index=False)
biden_tweets.to_csv(
    '../data/tweets/cleaned_hashtag_joebiden.csv', index=False)
