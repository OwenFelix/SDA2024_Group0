import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import pickle
import numpy as np
import re


# Function to extract hashtags
def extract_hashtags(text):
    return re.findall(r'#\w+', text.lower())

# Function to calculate the weighted mean
def weighted_mean(x, sigma=4, alpha=2):
    likes = x['likes']
    retweets = x['retweet_count']
    sentiment_polarity = x['sentiment_polarity']

    sentiment_weight = 2 * np.log1p(likes) + 5 * np.log1p(retweets)
    sentiment_weight = np.where(sentiment_weight == 0, 1, sentiment_weight)

    positions = np.arange(len(x))
    center = len(x) // 2
    gaussian_kernel = np.exp(-0.5 * ((positions - center) / sigma) ** 2)

    weight = alpha * sentiment_weight + (1 - alpha) * gaussian_kernel
    weight /= np.sum(weight)

    return np.sum(sentiment_polarity * weight)

# Create time series with rolling window
def create_timeseries(data, state_code, window_size):
    tweets = data[data['state_code'] == state_code]
    tweets['created_at'] = pd.to_datetime(tweets['created_at'])
    tweets = tweets.set_index('created_at')
    tweets = tweets.sort_index()
    tweets['sentiment_polarity'] = tweets['sentiment_polarity'].astype(float)

    rolling_tweets = tweets.rolling(window_size)
    intervals = []

    for interval in rolling_tweets:
        intervals.append(weighted_mean(interval))

    tupled_intervals = list(zip(intervals, tweets.index))
    return intervals, tweets, tupled_intervals

# Visualize sentiment with peaks and forecasts, focused on pre-election period
def visualize_sentiment_with_election_results(state_code, state_data, forecast_df, election_date, trump_percentage, biden_percentage, top_hashtags):
    plt.figure(figsize=(12, 6))
    plt.plot(state_data.index, state_data['trump_sentiment'], label='Trump Sentiment', color='red')
    plt.plot(state_data.index, state_data['biden_sentiment'], label='Biden Sentiment', color='blue')

    if forecast_df is not None:
        # Limit forecast plot to before the election date only
        forecast_df = forecast_df[forecast_df.index <= election_date]
        plt.plot(forecast_df.index, forecast_df['trump_sentiment'], label='Forecasted Trump Sentiment', linestyle='--', color='red')
        plt.plot(forecast_df.index, forecast_df['biden_sentiment'], label='Forecasted Biden Sentiment', linestyle='--', color='blue')

    # Mark the election date
    plt.axvline(x=election_date, color='orange', linestyle=':', label='Election Date')

    # Show voting percentages at election
    plt.text(election_date, 0.1, f"Trump: {trump_percentage}%\nBiden: {biden_percentage}%", color='orange', fontsize=10, ha='center')

    # Annotate the hashtags
    for i, hashtag in enumerate(top_hashtags):
        vertical_offset = 10 * (i + 1)  # Dynamic vertical offset for each hashtag
        plt.annotate(hashtag, (state_data.index[0], 0), textcoords="offset points", xytext=(0, vertical_offset), ha='center')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Polarity')
    plt.title(f'Sentiment Polarity Forecast for {state_code}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare the state data
def prepare_state_data(trump_data, biden_data, state_code, time_interval='24H'):
    trump_series = prepare_time_series(trump_data, state_code, time_interval)
    biden_series = prepare_time_series(biden_data, state_code, time_interval)
    combined_data = pd.concat([trump_series, biden_series], axis=1)
    combined_data.columns = ['trump_sentiment', 'biden_sentiment']
    return combined_data

# Main analysis function for sentiment and forecasting, focused on pre-election data
def analyze_state_sentiment_fixed(trump_data, biden_data, voting_data, time_interval='24H', forecast_steps=7):
    state_results = []
    for state_code in voting_data['state_abr']:
        print(f"Processing state: {state_code}")

        # Prepare state data (sentiment series for Trump and Biden)
        state_data = prepare_state_data(trump_data, biden_data, state_code, time_interval)

        if state_data.empty or len(state_data) < 5:
            print(f"Insufficient data for {state_code}. Skipping...")
            continue

        try:
            # Apply VAR model for sentiment forecasting, focusing on pre-election period
            model = VAR(state_data)
            results = model.fit(ic='aic')  # Automatically select lag based on AIC
            print(f"Optimal lag length for {state_code}: {results.k_ar}")

            if results.k_ar == 0:
                print(f"No valid lag length for {state_code}. Skipping...")
                continue

            forecast = results.forecast(state_data.values[-results.k_ar:], steps=forecast_steps)
            forecast_dates = pd.date_range(state_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
            forecast_df = pd.DataFrame(
                forecast, index=forecast_dates, columns=['trump_sentiment', 'biden_sentiment']
            )

            # Identify peaks (top sentiment values)
            peaks = state_data.idxmax()
            top_hashtags = analyze_hashtags_on_peaks(trump_data[trump_data['state_code'] == state_code]) + \
                           analyze_hashtags_on_peaks(biden_data[biden_data['state_code'] == state_code])

            # Fetch actual election result (vote percentages)
            election_result = voting_data[voting_data['state_abr'] == state_code]
            trump_percentage = election_result['trump_pct'].values[0]
            biden_percentage = election_result['biden_pct'].values[0]

            # Manually set the election date (since it's the same for all states)
            election_date = pd.to_datetime('2020-11-03')  # Set the election date manually

            # Visualize results with peaks and hashtags, and overlay election results
            visualize_sentiment_with_election_results(state_code, state_data, forecast_df, election_date, trump_percentage, biden_percentage, top_hashtags[:3])

            state_results.append({
                'state_code': state_code,
                'observed_data': state_data,
                'forecast_data': forecast_df,
                'model_results': results
            })

        except Exception as e:
            print(f"Error processing {state_code}: {e}")

    # Save results for further analysis
    with open('./data/analysis_results.pkl', 'wb') as f:
        pickle.dump(state_results, f)

# Execute the analysis
analyze_state_sentiment_fixed(trump_tweets, biden_tweets, voting_results)
