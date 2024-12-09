"""
var_model.py

DESCRIPTION:
This script analyzes the sentiment polarity time series for Trump and Biden tweets,
tests for stationarity, fits a Vector AutoRegression (VAR) model, and forecasts future sentiment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
# from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
from create_timeseries import weighted_mean

import warnings
warnings.filterwarnings("ignore")

# Load datasets
trump_df = pd.read_csv("../data/tweets/cleaned_hashtag_donaldtrump.csv")
biden_df = pd.read_csv("../data/tweets/cleaned_hashtag_joebiden.csv")


def create_timeseries(data, window_size):
    """
    Create a time series for the sentiment polarity of the tweets.

    Parameters:
    data (DataFrame): The input data containing tweets.
    window_size (str): The resampling window size.

    Returns:
    Series: The resampled sentiment polarity time series.
    """
    data.loc[:, 'created_at'] = pd.to_datetime(data['created_at'])
    data.set_index('created_at', inplace=True)
    data.sort_index(inplace=True)
    data['sentiment_polarity'] = data['sentiment_polarity'].astype(float)
    sentiment = data.resample(window_size).apply(weighted_mean)
    return sentiment


# Generate time series
trump_sentiment = create_timeseries(trump_df, '24h')
biden_sentiment = create_timeseries(biden_df, '24h')

print(trump_sentiment)

def stationarity_test(data, candidate):
    """
    Perform the Augmented Dickey-Fuller test to determine stationarity.

    Parameters:
    data (Series): The time series data to test.
    candidate (str): The name of the candidate (for labeling purposes).
    """
    result = adfuller(data.dropna())
    print(f"\nADF Test for {candidate}:")
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] < 0.05:
        print(f"The time series for {candidate} is stationary.")
    else:
        print(f"The time series for {candidate} is not stationary.")

# Test stationarity
stationarity_test(trump_sentiment, 'Trump')
stationarity_test(biden_sentiment, 'Biden')

# Apply differencing to make the series stationary
trump_sentiment_diff = trump_sentiment.diff().dropna()
biden_sentiment_diff = biden_sentiment.diff().dropna()

# Re-test stationarity after differencing
print("\nAfter differencing:")
stationarity_test(trump_sentiment_diff, 'Trump')
stationarity_test(biden_sentiment_diff, 'Biden')

# Combine the differenced series into a DataFrame
diff_data = pd.DataFrame({
    'Trump': trump_sentiment_diff,
    'Biden': biden_sentiment_diff
}).dropna()

# Adjust maxlags dynamically based on the size of the dataset
max_possible_lags = int(len(diff_data) / 5)
maxlags = min(15, max_possible_lags)

# Fit the VAR model
model = VAR(diff_data)
results = model.fit(maxlags=maxlags, ic='aic')
print(results.summary())

# Forecast sentiment
forecast_steps = 10
forecast = results.forecast(diff_data.values[-results.k_ar:], steps=forecast_steps)
forecast_index = pd.date_range(start=diff_data.index[-1], periods=forecast_steps + 1, freq='24H')[1:]

# Convert forecast to DataFrame
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Trump_Forecast', 'Biden_Forecast'])

# Plot the forecasts
plt.figure(figsize=(12, 6))
plt.plot(trump_sentiment_diff, label='Trump (Differenced)', color='blue')
plt.plot(biden_sentiment_diff, label='Biden (Differenced)', color='red')
plt.plot(forecast_df['Trump_Forecast'], label='Trump Forecast', linestyle='dashed', color='blue')
plt.plot(forecast_df['Biden_Forecast'], label='Biden Forecast', linestyle='dashed', color='red')
plt.legend()
plt.title('Sentiment Polarity and Forecast')
plt.xlabel('Date')
plt.ylabel('Sentiment Polarity')
plt.grid()
plt.show()
