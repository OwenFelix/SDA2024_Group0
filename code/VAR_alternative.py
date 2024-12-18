"""
VAR_alternative.py

DESCRIPTION:
This script processes, analyzes, and visualizes sentiment trends from tweets
to investigate public opinion for Trump and Biden across various U.S. states
during the 2020 election. It applies time series methods such as Vector Auto
regression (VAR) and ARIMA models to study sentiment interactions and temporal
dynamics.

Caveat: this was not included in the presentation because we didn't have time
to check if ARIMA can help reflect the twitter sentiments for states with
insufficient data and can't be made stationary after 2 differencing attempts.
Since ARIMA focuses only on a single candidate’s sentiment trend over time,
it might lead to biased results. It would have been interesting to test and
set limitations to use it but we ran out of time to look into how we can
integrate this to our analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


# Extract hashtags from tweet text
def extract_hashtags(tweet):
    return re.findall(r"#\w+", str(tweet).lower())


# Identify dominant hashtags dynamically
def identify_dominant_hashtags(data, top_n=10):
    hashtag_sentiments = {}
    hashtag_counts = {}

    for _, row in data.iterrows():
        hashtags = extract_hashtags(row['tweet'])
        sentiment = row.get('sentiment_polarity', 0.0)
        for tag in hashtags:
            if tag not in hashtag_sentiments:
                hashtag_sentiments[tag] = []
                hashtag_counts[tag] = 0
            hashtag_sentiments[tag].append(sentiment)
            hashtag_counts[tag] += 1

    hashtag_summary = {
        tag: {
            "average_sentiment": np.mean(sentiments),
            "frequency": hashtag_counts[tag]
        }
        for tag, sentiments in hashtag_sentiments.items()
    }
    return sorted(hashtag_summary.items(), key=lambda x: x[1]['frequency'],
                  reverse=True)[:top_n]


# Compute weighted sentiment score using hashtag weights
def compute_weighted_sentiment_dynamic(tweet, sentiment_score, hashtag_weights):
    hashtags = extract_hashtags(tweet)
    weights = [hashtag_weights.get(tag, 0) for tag in hashtags]
    return np.average([sentiment_score] * len(weights), weights=weights) if weights else sentiment_score


# Prepare time series and associate hashtags for spikes
def prepare_time_series(data, state_code, hashtag_weights):
    try:
        state_data = data[data['state_code'] == state_code].copy()
        state_data['created_at'] = pd.to_datetime(state_data['created_at'])
        state_data['weighted_sentiment'] = state_data.apply(
            lambda row: compute_weighted_sentiment_dynamic(row['tweet'], row['sentiment_polarity'], hashtag_weights), axis=1
        )
        state_data['hashtags'] = state_data['tweet'].apply(extract_hashtags)
        state_data['rounded_date'] = state_data['created_at'].dt.date

        # Resample sentiment to daily averages
        time_series = state_data.set_index('created_at')['weighted_sentiment'].resample('24h').mean()
        time_series = time_series.replace([np.inf, -np.inf], np.nan).dropna().ffill().bfill()

        return time_series, state_data
    except Exception as e:
        print(f"Error preparing time series for state {state_code}: {e}")
        return None, None


# Check stationarity using ADF test with differencing
def check_stationarity_with_differencing(series, name, max_diff=2):
    for diff in range(max_diff + 1):
        result = adfuller(series.dropna())
        p_value = result[1]
        decision = "Reject H0 (Stationary)" if p_value <= 0.05 else "Fail to Reject H0 (Not Stationary)"
        print(f"{name}: ADF p-value = {p_value:.4f}. {decision}.")

        if p_value <= 0.05:
            if diff > 0:
                print(f"{name} is stationary after differencing ({diff}).")
            else:
                print(f"{name} is stationary.")
            return True, series
        print(f"{name} is not stationary. Differencing applied.")
        series = series.diff().dropna()

    print(f"{name} - Stationarity could not be achieved after {max_diff} differencing attempts.")
    return False, series


# Visualize sentiment trends with spikes and hashtags
def visualize_sentiment(state_code, combined_data, spikes, hashtags,
                        voting_results, analysis_type):
    plt.figure(figsize=(14, 8))
    plt.plot(combined_data.index, combined_data['trump_sentiment'],
             label="Trump Sentiment", color="red")
    plt.plot(combined_data.index, combined_data['biden_sentiment'],
             label="Biden Sentiment", color="blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    # Add spikes with hashtags
    for col, color in zip(['trump_sentiment', 'biden_sentiment'],
                          ['red', 'blue']):
        for idx in spikes[col].index:
            value = combined_data.at[idx, col]
            hashtags_on_date = hashtags.get(idx.date(), [])
            top_hashtags = ', '.join(hashtags_on_date[:3]) if hashtags_on_date else "[No Hashtags]"
            if np.isfinite(value):
                plt.scatter(idx, value, color=color, s=50)
                plt.text(idx, value + 0.02, top_hashtags, fontsize=8, ha='center', color="black")

    # Election and debate lines
    plt.axvline(pd.to_datetime('2020-10-15'), color='purple', linestyle='--', label='Canceled Debate (Oct 15)')
    plt.axvline(pd.to_datetime('2020-10-22'), color='green', linestyle='--', label='Final Debate (Oct 22)')
    plt.axvline(pd.to_datetime('2020-11-03'), color='orange', linestyle='--', label='Election Date')

    plt.title(f"{analysis_type} Analysis: Sentiment Trends for {state_code}")
    plt.xlabel("Date")
    plt.ylabel("Weighted Sentiment")
    plt.legend()
    plt.show()


# Main function to analyze states
def analyze_states(trump_data, biden_data, voting_results):
    trump_weights = {tag: data['average_sentiment'] for tag, data in identify_dominant_hashtags(trump_data)}
    biden_weights = {tag: data['average_sentiment'] for tag, data in identify_dominant_hashtags(biden_data)}

    for state in voting_results['state_abr']:
        print(f"\nAnalyzing {state}...")
        trump_series, trump_data_clean = prepare_time_series(trump_data, state, trump_weights)
        biden_series, biden_data_clean = prepare_time_series(biden_data, state, biden_weights)
        if trump_series is None or biden_series is None:
            print(f"{state} - Insufficient data. Skipping.")
            continue

        combined_data = pd.concat([trump_series, biden_series], axis=1)
        combined_data.columns = ['trump_sentiment', 'biden_sentiment']

        stationary_trump, trump_series_diff = check_stationarity_with_differencing(
            combined_data['trump_sentiment'], f"{state} - Trump Sentiment"
        )
        stationary_biden, biden_series_diff = check_stationarity_with_differencing(
            combined_data['biden_sentiment'], f"{state} - Biden Sentiment"
        )

        spikes = {col: combined_data[col][combined_data[col].diff().abs() > 0.1] for col in combined_data.columns}
        all_data_clean = pd.concat([trump_data_clean, biden_data_clean])
        hashtags = all_data_clean.groupby('rounded_date')['hashtags'].sum().to_dict()

        if stationary_trump and stationary_biden:
            print(f"{state}: Performing VAR analysis.")
            analysis_type = "VAR"
        else:
            print(f"{state}: Performing ARIMA analysis.")
            analysis_type = "ARIMA"
            combined_data['trump_sentiment'] = trump_series_diff
            combined_data['biden_sentiment'] = biden_series_diff

        visualize_sentiment(state, combined_data, spikes, hashtags, voting_results, analysis_type)


# Load datasets
trump_data = pd.read_csv('../tmp/cleaned_hashtag_donaldtrump.csv')
biden_data = pd.read_csv('../tmp/cleaned_hashtag_joebiden.csv')
voting_results = pd.read_csv('../data/election_results/voting.csv')

# Convert 'created_at' to datetime
trump_data['created_at'] = pd.to_datetime(trump_data['created_at'],
                                          errors='coerce')
biden_data['created_at'] = pd.to_datetime(biden_data['created_at'],
                                          errors='coerce')

# Run analysis
analyze_states(trump_data, biden_data, voting_results)
