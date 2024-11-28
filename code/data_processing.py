"""
HEADER
TODO : FILL WITH DESCRIPTION OF CONTENT OF FILE
"""

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import re  # For regular expressions
import emoji  # For handling emojis
import string  # For handling string operations
from nltk.corpus import stopwords  # For stopwords
from nltk.tokenize import TweetTokenizer  # For tokenizing tweets
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
from wordcloud import WordCloud  # For creating wordclouds
from textblob import TextBlob  # For sentiment analysis
from langdetect import detect, DetectorFactory  # For detecting the language of a text

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
    stop_words = set(stopwords.words('english'))
    stop_words -= {'not', 'no'}
    return " ".join(word for word in text.split() if word not in stop_words)


def gen_z_slang(text):
    return " ".join(CHAT_WORDS.get(w, w) for w in text.split())


def clean_tweet_data(text):
    # GenZ slang treatment
    text = gen_z_slang(text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs entirely
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # remove punctuation, but keep !, ?, # and @
    text = text.translate(str.maketrans('', '', string.punctuation.replace(
        '!', '').replace('?', '').replace('#', '').replace('@', '')))

    # remove the stopwords
    text = remove_stopwords(text)

    # Replace emoji with text for the expression of the emoji
    text = emoji.demojize(text)

    # Convert to lowercase
    text = text.lower()

    return text


election_results_data = read_data("../data/election_results/voting.csv")
trump_tweets = read_data("../data/tweets/hashtag_donaldtrump.csv")
biden_tweets = read_data("../data/tweets/hashtag_joebiden.csv")


# Remove irrelevant columns
irrelevant_cols = ['source', 'user_id', 'user_name', 'user_screen_name',
                   'user_description', 'user_join_date', 'collected_at']

trump_tweets.drop(columns=irrelevant_cols, inplace=True)
biden_tweets.drop(columns=irrelevant_cols, inplace=True)

# Remove rows with missing values
# Maybe reconsider only dropping if there is no state data available
trump_tweets.dropna(axis=0, inplace=True)
biden_tweets.dropna(axis=0, inplace=True)

pd.set_option('display.max_columns', None)

# Filter tweets from the USA
trump_usa_tweets = trump_tweets[trump_tweets['country'] == 'United States of America'].copy()
biden_usa_tweets = biden_tweets[biden_tweets['country'] == 'United States of America'].copy()

print(trump_usa_tweets.shape)
print(biden_usa_tweets.shape)

def get_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'


# Remove tweets that are in both datasets
tids = trump_usa_tweets['tweet_id']
bids = biden_usa_tweets['tweet_id']

ids_tweets_in_common = set(trump_usa_tweets['tweet_id']).intersection(set(biden_usa_tweets['tweet_id']))

trump_usa_tweets = trump_usa_tweets[~tids.isin(ids_tweets_in_common)]
biden_usa_tweets = biden_usa_tweets[~bids.isin(ids_tweets_in_common)]

print(trump_usa_tweets.shape)
print(biden_usa_tweets.shape)

# Filter out tweets that are not in English
# trump_usa_tweets['language'] = trump_usa_tweets['tweet'].apply(lambda x: get_language(x))
# biden_usa_tweets['language'] = biden_usa_tweets['tweet'].apply(lambda x: get_language(x))

# trump_usa_tweets = trump_usa_tweets[trump_usa_tweets['language'] == 'en'].copy()
# biden_usa_tweets = biden_usa_tweets[biden_usa_tweets['language'] == 'en'].copy()

# print(trump_usa_tweets.shape)
# print(biden_usa_tweets.shape)

# Clean the tweets
trump_usa_tweets['tweet'] = trump_usa_tweets['tweet'].apply(clean_tweet_data)
biden_usa_tweets['tweet'] = biden_usa_tweets['tweet'].apply(clean_tweet_data)

# Sentiment analysis
trump_usa_tweets['sentiment_polarity'] = trump_usa_tweets['tweet'].apply(
    lambda x: TextBlob(x).sentiment.polarity)
biden_usa_tweets['sentiment_polarity'] = biden_usa_tweets['tweet'].apply(
    lambda x: TextBlob(x).sentiment.polarity)

# trump_usa_tweets['sentiment_subjectivity'] = trump_usa_tweets['tweet'].apply(
#     lambda x: TextBlob(x).sentiment.subjectivity)
# biden_usa_tweets['sentiment_subjectivity'] = biden_usa_tweets['tweet'].apply(
#     lambda x: TextBlob(x).sentiment.subjectivity)

pd.set_option('display.max_columns', None)
print(trump_usa_tweets['tweet'].head())
print(biden_usa_tweets['tweet'].head())
print(trump_usa_tweets['sentiment_polarity'].head())
print(biden_usa_tweets['sentiment_polarity'].head())
print(trump_usa_tweets['sentiment_subjectivity'].head())
print(biden_usa_tweets['sentiment_subjectivity'].head())
