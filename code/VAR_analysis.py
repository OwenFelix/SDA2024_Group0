"""
VAR_analysis.py

DESCRIPTION:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.api import VAR
from create_timeseries import weighted_mean
import warnings
warnings.filterwarnings("ignore")


def extract_hashtags(tweet):
    """
    This function extracts all the hashtags from a tweet text.
    """
    return re.findall(r"#\w+", str(tweet).lower())


def identify_dominant_hashtags(data, top_n=10):
    """
    This function identifies the most dominant hashtags based on the average
    sentiment polarity in a dynmaic way.
    """
    hashtag_sentiments = {}
    hashtag_counts = {}

    for _, row in data.iterrows():
        hashtags = extract_hashtags(row['tweet'])
        try:
            # Ensure sentiment is numeric
            sentiment = float(row['sentiment_polarity'])
        except ValueError:
            continue  # Skip if sentiment is not a valid number
        for tag in hashtags:
            if tag not in hashtag_sentiments:
                hashtag_sentiments[tag] = []
                hashtag_counts[tag] = 0
            hashtag_sentiments[tag].append(sentiment)
            hashtag_counts[tag] += 1

    # Compute average sentiment and frequency for each hashtag
    hashtag_summary = {
        tag: {
            "average_sentiment": np.mean(sentiments) if len(sentiments) > 0 else 0.0,
            "frequency": hashtag_counts[tag]
        }
        for tag, sentiments in hashtag_sentiments.items()
    }

    # Sort by frequency
    dominant_hashtags = sorted(
        hashtag_summary.items(),
        key=lambda x: x[1]['frequency'],
        reverse=True
    )
    return dominant_hashtags[:top_n]

# Compute weighted sentiment score using dynamic weights


def compute_weighted_sentiment_dynamic(tweet, sentiment_score, hashtag_weights):
    hashtags = extract_hashtags(tweet)
    weights = [hashtag_weights.get(tag, 0) for tag in hashtags]
    if any(weights):
        weighted_mean = np.average(
            [sentiment_score] * len(weights), weights=weights)
        return weighted_mean
    return sentiment_score  # Fallback to raw sentiment score

# Prepare weighted time series for a state


def prepare_weighted_time_series(data, state_code, hashtag_weights, time_interval='24h'):
    try:
        state_data = data[data['state_code'] == state_code].copy()
        state_data['created_at'] = pd.to_datetime(
            state_data['created_at'], errors='coerce')
        state_data.dropna(subset=['created_at'], inplace=True)
        state_data.set_index('created_at', inplace=True)
        state_data['weighted_sentiment'] = state_data.apply(
            lambda row: compute_weighted_sentiment_dynamic(row['tweet'],
                                                           row['sentiment_polarity'], hashtag_weights), axis=1
        )
        return state_data['weighted_sentiment'].resample(time_interval).mean().dropna()
    except Exception as e:
        print(f"Error processing time series for state {state_code}: {e}")
        return pd.Series()

# Combine Trump and Biden data for a state


def prepare_state_data(trump_data, biden_data, state_code, trump_weights, biden_weights, time_interval='24h'):
    trump_series = prepare_weighted_time_series(
        trump_data, state_code, trump_weights, time_interval)
    biden_series = prepare_weighted_time_series(
        biden_data, state_code, biden_weights, time_interval)
    combined_data = pd.concat([trump_series, biden_series], axis=1)
    combined_data.columns = ['trump_sentiment', 'biden_sentiment']
    return combined_data

# Perform VAR analysis with stationarity


def perform_var_analysis_with_stationarity(state_data, confidence=0.95):
    model = VAR(state_data)
    results = model.fit(maxlags=1)

    coefficients = results.params
    stderr = results.stderr
    z_score = 1.96  # For 95% confidence interval

    lower_bound = coefficients - z_score * stderr
    upper_bound = coefficients + z_score * stderr

    ci_summary = f"Confidence Intervals ({confidence * 100}%):\n"
    for col in coefficients.columns:
        ci_summary += f"{col}:\n"
        for idx in coefficients.index:
            coef = coefficients.loc[idx, col]
            lb = lower_bound.loc[idx, col]
            ub = upper_bound.loc[idx, col]
            ci_summary += f"  {idx}: Coefficient={coef:.4f}, CI=({lb:.4f}, {ub:.4f})\n"

    return results, ci_summary

# Visualize sentiment with confidence intervals and spike annotations


def visualize_sentiment_with_spikes(state_code, state_data, voting_results, var_results, ci_summary, trump_data, biden_data):
    plt.figure(figsize=(14, 8))
    plt.plot(state_data.index,
             state_data['trump_sentiment'], label='Trump Sentiment', color='red')
    plt.plot(state_data.index,
             state_data['biden_sentiment'], label='Biden Sentiment', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Election date and results
    election_date = pd.to_datetime('2020-11-03')
    plt.axvline(x=election_date, color='orange',
                linestyle='--', label='Election Date')
    state_results = voting_results[voting_results['state_abr'] == state_code]
    trump_pct = state_results['trump_pct'].values[0]
    biden_pct = state_results['biden_pct'].values[0]
    plt.text(election_date, 0.15,
             f"Trump: {trump_pct}%\nBiden: {biden_pct}%", color='purple', fontsize=10)

    # Mark spikes and annotate with hashtags
    for col, color, tweet_data in zip(['trump_sentiment', 'biden_sentiment'], ['red', 'blue'], [trump_data, biden_data]):
        sentiment_diff = state_data[col].diff().fillna(0)
        # Threshold for spikes
        spikes = sentiment_diff[sentiment_diff.abs() > 0.1]
        for idx in spikes.index:
            plt.scatter(idx, state_data[col][idx], color=color, s=50)

            # Find matching tweets
            matching_tweets = tweet_data[tweet_data['created_at'].dt.floor(
                'D') == idx.floor('D')]
            if not matching_tweets.empty:
                hashtags = extract_hashtags(matching_tweets.iloc[0]['tweet'])

                # Format hashtags horizontally
                formatted_hashtags = ' '.join(hashtags[:2])

                # Ensure hashtags remain inside the plot
                y_min, y_max = plt.ylim()
                y_position = np.clip(state_data[col][idx] + 0.05, y_min, y_max)

                # Annotate hashtags
                plt.text(idx, y_position, formatted_hashtags,
                         fontsize=8, color='black', ha='center', rotation=0)

    # Title and labels
    plt.title(
        f'Sentiment Analysis with Confidence Intervals and Spikes for {state_code}')
    plt.xlabel('Date')
    plt.ylabel('Weighted Sentiment Polarity')
    plt.legend()
    plt.tight_layout()

    # Add figure caption for correlation matrix and CI
    caption = f"Correlation matrix of residuals:\n{var_results.resid_corr}\n\n{ci_summary}"
    plt.figtext(0.5, -0.3, caption, wrap=True,
                horizontalalignment='center', fontsize=10)
    plt.show()

# Analyze all states with dynamic hashtag weights and visualization


def analyze_all_states_with_dynamic_weights(trump_data, biden_data, voting_results):
    trump_weights = {tag: sentiment['average_sentiment']
                     for tag, sentiment in identify_dominant_hashtags(trump_data)}
    biden_weights = {tag: sentiment['average_sentiment']
                     for tag, sentiment in identify_dominant_hashtags(biden_data)}

    for state_code in voting_results['state_abr']:
        try:
            state_data = prepare_state_data(
                trump_data, biden_data, state_code, trump_weights, biden_weights)
            var_results, ci_summary = perform_var_analysis_with_stationarity(
                state_data)
            visualize_sentiment_with_spikes(
                state_code, state_data, voting_results, var_results, ci_summary, trump_data, biden_data)
        except Exception as e:
            print(f"Error processing {state_code}: {e}")


if __name__ == "main":
    # Load datasets
    trump_tweets = pd.read_csv(
        '../data/tweets/cleaned_hashtag_donaldtrump.csv')
    biden_tweets = pd.read_csv('../data/tweets/cleaned_hashtag_joebiden.csv')
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    # Convert 'created_at' to datetime
    trump_tweets['created_at'] = pd.to_datetime(
        trump_tweets['created_at'], errors='coerce')
    biden_tweets['created_at'] = pd.to_datetime(
        biden_tweets['created_at'], errors='coerce')

    # Analyze all states
    analyze_all_states_with_dynamic_weights(
        trump_tweets, biden_tweets, voting_results)
