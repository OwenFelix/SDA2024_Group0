import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import pickle
import numpy as np
import re

# Function to sanitize hashtags
def sanitize_hashtags(hashtags):
    sanitized = [re.sub(r'[^\w#]', '', tag) for tag in hashtags]
    return [tag for tag in sanitized if tag]  # Remove empty strings

# Function to annotate spikes with hashtags
def annotate_spikes_with_hashtags(state_data, hashtags, sentiment_column, color, label):
    spike_indices = state_data[sentiment_column].nlargest(3).index  # Get top 3 spikes
    for i, idx in enumerate(spike_indices):
        if idx in state_data.index:
            plt.scatter(idx, state_data.loc[idx, sentiment_column], color=color, label=label if i == 0 else "")
            if hashtags:
                plt.annotate(
                    hashtags[i % len(hashtags)],
                    (idx, state_data.loc[idx, sentiment_column]),
                    textcoords="offset points",
                    xytext=(-50, 10),
                    arrowprops=dict(arrowstyle='->', color=color)
                )

# Function to prepare time series
def prepare_time_series(data, state_code, time_interval='24h'):
    state_data = data[data['state_code'] == state_code].copy()
    state_data.loc[:, 'created_at'] = pd.to_datetime(state_data['created_at'], errors='coerce')
    state_data = state_data.dropna(subset=['created_at'])
    state_data = state_data.set_index('created_at')
    state_data = state_data.sort_index()
    time_series = state_data['sentiment_polarity'].resample(time_interval).mean().dropna()
    return time_series

# Function to prepare state data
def prepare_state_data(trump_data, biden_data, state_code, time_interval='24h'):
    trump_series = prepare_time_series(trump_data, state_code, time_interval)
    biden_series = prepare_time_series(biden_data, state_code, time_interval)
    combined_data = pd.concat([trump_series, biden_series], axis=1)
    combined_data.columns = ['trump_sentiment', 'biden_sentiment']
    return combined_data

# Visualize sentiment with election results
def visualize_sentiment_with_election_results(state_code, state_data, forecast_df, election_date, trump_percentage, biden_percentage, trump_hashtags, biden_hashtags):
    plt.figure(figsize=(14, 8))  # Increased figure size

    # Plot observed and forecasted sentiments
    plt.plot(state_data.index, state_data['trump_sentiment'], label='Trump Sentiment', color='red')
    plt.plot(state_data.index, state_data['biden_sentiment'], label='Biden Sentiment', color='blue')
    if forecast_df is not None:
        forecast_df = forecast_df[forecast_df.index <= election_date]  # Limit to pre-election
        plt.plot(forecast_df.index, forecast_df['trump_sentiment'], label='Forecasted Trump Sentiment', linestyle='--', color='red')
        plt.plot(forecast_df.index, forecast_df['biden_sentiment'], label='Forecasted Biden Sentiment', linestyle='--', color='blue')

    # Annotate spikes with hashtags
    annotate_spikes_with_hashtags(state_data, trump_hashtags, 'trump_sentiment', 'red', 'Trump Peaks')
    annotate_spikes_with_hashtags(state_data, biden_hashtags, 'biden_sentiment', 'blue', 'Biden Peaks')

    # Mark the election date
    plt.axvline(x=election_date, color='orange', linestyle=':', label='Election Date')
    plt.text(election_date, 0.1, f"Trump: {trump_percentage}%\nBiden: {biden_percentage}%", color='orange', fontsize=10, ha='center')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'Sentiment Analysis and Election Results for {state_code}')
    plt.legend()
    plt.tight_layout()  # Adjust layout for better visualization
    plt.show()

# Main function to analyze state sentiment
def analyze_state_sentiment(trump_data, biden_data, voting_data, time_interval='24h', forecast_steps=7):
    state_results = []
    for state_code in voting_data['state_abr']:
        try:
            state_data = prepare_state_data(trump_data, biden_data, state_code, time_interval)
            if state_data.empty or len(state_data) < 5:
                continue

            model = VAR(state_data)
            results = model.fit(ic='aic')
            if results.k_ar == 0:
                continue

            forecast = results.forecast(state_data.values[-results.k_ar:], steps=forecast_steps)
            forecast_dates = pd.date_range(state_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['trump_sentiment', 'biden_sentiment'])

            trump_hashtags = trump_data[trump_data['state_code'] == state_code]['hashtags'].explode().value_counts().index.tolist()[:3]
            biden_hashtags = biden_data[biden_data['state_code'] == state_code]['hashtags'].explode().value_counts().index.tolist()[:3]

            election_result = voting_data[voting_data['state_abr'] == state_code]
            trump_percentage = election_result['trump_pct'].values[0]
            biden_percentage = election_result['biden_pct'].values[0]
            election_date = pd.to_datetime('2020-11-03')

            visualize_sentiment_with_election_results(state_code, state_data, forecast_df, election_date, trump_percentage, biden_percentage, trump_hashtags, biden_hashtags)
            state_results.append({
                'state_code': state_code,
                'observed_data': state_data,
                'forecast_data': forecast_df,
                'model_results': results
            })
        except Exception as e:
            print(f"Error processing {state_code}: {e}")
    with open('./data/analysis_results.pkl', 'wb') as f:
        pickle.dump(state_results, f)

# Load datasets
trump_tweets = pd.read_csv('./data/tweets/cleaned_hashtag_donaldtrump.csv')
biden_tweets = pd.read_csv('./data/tweets/cleaned_hashtag_joebiden.csv')
voting_results = pd.read_csv('./data/election_results/cleaned_voting.csv')

# Execute the analysis
analyze_state_sentiment(trump_tweets, biden_tweets, voting_results)
