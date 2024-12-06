"""
HEADER
TODO : FILL WITH DESCRIPTION OF CONTENT OF FILE
"""

import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import re  # For regular expressions
import emoji  # For handling emojis
import string  # For handling string operations
from nltk.corpus import stopwords  # For stopwords
from nltk.tokenize import TweetTokenizer  # For tokenizing tweets
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
from textblob import TextBlob  # For sentiment analysis
# For detecting the language of a text
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # For reproducibility

# GenZ slang dictionary
CHAT_WORDS = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': "For What It's Worth",
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you (also a chat program)',
    'ILU': 'ILU: I Love You',
    'IMHO': 'In My Honest/Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My A.. Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'PITA': 'Pain In The A..',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My A.. Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The F...',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait...',
    '7K': 'Sick:-D Laugher',
    'TFW': 'That feeling when',
    'MFW': 'My face when',
    'MRW': 'My reaction when',
    'IFYP': 'I feel your pain',
    'TNTL': 'Trying not to laugh',
    'JK': 'Just kidding',
    'IDC': "I don't care",
    'ILY': 'I love you',
    'IMU': 'I miss you',
    'ADIH': 'Another day in hell',
    'ZZZ': 'Sleeping, bored, tired',
    'WYWH': 'Wish you were here',
    'TIME': 'Tears in my eyes',
    'BAE': 'Before anyone else',
    'FIMH': 'Forever in my heart',
    'BSAAW': 'Big smile and a wink',
    'BWL': 'Bursting with laughter',
    'BFF': 'Best friends forever',
    'CSL': "Can't stop laughing"
}


def read_data(file_path):
    """
    Read the data from the given file path
    """
    data = pd.read_csv(file_path, chunksize=1000, lineterminator='\n')
    chunk_list = []
    for chunk in data:
        chunk_list.append(chunk)
    return pd.concat(chunk_list, axis=0)


def remove_stopwords(text):
    """
    Remove stopwords from the text except negation words
    """
    stop_words = set(stopwords.words('english'))
    stop_words -= {'not', 'no'}
    return " ".join(word for word in text.split() if word not in stop_words)


def clean_tweet_data(text):
    # GenZ slang treatment
    text = " ".join(CHAT_WORDS.get(w, w) for w in text.split())

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs entirely
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Replace emoji with text for the expression of the emoji
    text = emoji.demojize(text)

    # remove punctuation, but keep !, ?, # and @
    punctuation_to_replace = ''.join(
        c for c in string.punctuation if c not in "!?#@")
    text = re.sub(f"[{re.escape(punctuation_to_replace)}]", " ", text)

    # remove the stopwords
    text = remove_stopwords(text)

    # Convert to lowercase
    text = text.lower()

    return text


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
