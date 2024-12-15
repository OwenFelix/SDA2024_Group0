"""
VAR_analysis.py

DESCRIPTION:
This script performs Vector Autoregression (VAR) analysis on the sentiment
time series data for Trump and Biden in each state. It also performs Granger
causality tests to determine if one candidate's sentiment Granger-causes the
other's sentiment. The script visualizes the sentiment time series data with
confidence intervals and spike annotations.
"""

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import re  # For regular expressions
from statsmodels.tsa.api import VAR  # For Vector Autoregression (VAR)
# For Granger causality tests (hypothesis testing)
from statsmodels.tsa.stattools import grangercausalitytests
# For Augmented Dickey-Fuller (stationarity) test
from statsmodels.tsa.stattools import adfuller
import warnings  # For handling warnings
warnings.filterwarnings('ignore')  # Ignore warnings


def extract_hashtags(tweet):
    """
    Extract hashtags from a tweet text.
    """
    return re.findall(r"#\w+", str(tweet).lower())


def identify_dominant_hashtags(data, top_n=10):
    """
    Identify the most dominant hashtags in a dynamic manner based on average
    sentiment and frequency.
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
            "average_sentiment": np.mean(sentiments) if len(sentiments) > 0
            else 0.0,
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


def perform_adf_test(time_series, label):
    """
    Perform Augmented Dickey-Fuller (ADF) test on a given time series.
    """
    result = adfuller(time_series.dropna())  # Drop NaN values before testing
    adf_results = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Number of Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4]
    }

    print(f"ADF Test Results for {label}:")
    print(f"ADF Statistic: {adf_results['ADF Statistic']}")
    print(f"p-value: {adf_results['p-value']}")
    print("Critical Values:")
    for key, val in adf_results['Critical Values'].items():
        print(f"  {key}: {val}")

    # Determine stationarity
    if adf_results['p-value'] < 0.05:
        print(f"{label} is stationary (rejecting null hypothesis).")
        return time_series, False  # No differencing needed
    else:
        print(f"{label} is non-stationary (cannot reject null hypothesis).")
        print("Differencing the series to ensure stationarity...")
        differenced_series = time_series.diff().dropna()
        return differenced_series, True  # Indicate differencing was performed


# Function to analyze stationarity and difference if necessary
def analyze_stationarity_and_differencing(state_data):
    """
    Analyze stationarity for both Trump and Biden sentiment series.
    Differencing is applied if series are non-stationary.
    Returns processed (stationary or differenced) data.
    """
    print("\nAnalyzing and differencing if necessary...")

    # Process Trump sentiment
    print("\nRunning ADF test for Trump sentiment...")
    trump_series, trump_differenced = perform_adf_test(
        state_data['trump_sentiment'], "Trump Sentiment"
    )

    # Process Biden sentiment
    print("\nRunning ADF test for Biden sentiment...")
    biden_series, biden_differenced = perform_adf_test(
        state_data['biden_sentiment'], "Biden Sentiment"
    )

    # Combine both series into a DataFrame
    processed_data = pd.DataFrame({
        'trump_sentiment': trump_series,
        'biden_sentiment': biden_series
    })

    # Log changes (whether differencing was applied)
    if trump_differenced:
        print("Trump sentiment was differenced to ensure stationarity.")
    if biden_differenced:
        print("Biden sentiment was differenced to ensure stationarity.")

    return processed_data


def compute_weighted_sentiment_dynamic(tweet, sentiment_score,
                                       hashtag_weights):
    """
    Compute the weighted sentiment score based on dynamic hashtag weights.
    """
    hashtags = extract_hashtags(tweet)
    weights = [hashtag_weights.get(tag, 0) for tag in hashtags]
    if any(weights):
        weighted_mean = np.average(
            [sentiment_score] * len(weights), weights=weights)
        return weighted_mean
    return sentiment_score  # Fallback to raw sentiment score


def prepare_weighted_time_series(data, state_code, hashtag_weights,
                                 time_interval='24h'):
    """
    Prepare a weighted time series for a given state based on dynamic
    hashtag weights.
    """
    try:
        state_data = data[data['state_code'] == state_code].copy()
        state_data['created_at'] = pd.to_datetime(
            state_data['created_at'], errors='coerce')
        state_data.dropna(subset=['created_at'], inplace=True)
        state_data.set_index('created_at', inplace=True)
        state_data['weighted_sentiment'] = state_data.apply(
            lambda row: compute_weighted_sentiment_dynamic(
                row['tweet'],
                row['sentiment_polarity'],
                hashtag_weights), axis=1)
        return state_data[
            'weighted_sentiment'].resample(time_interval).mean().dropna()
    except Exception as e:
        print(f"Error processing time series for state {state_code}: {e}")
        return pd.Series()

# Combine Trump and Biden data for a state


def prepare_state_data(trump_data, biden_data, state_code, trump_weights,
                       biden_weights, time_interval='24h'):
    """
    Prepare a combined weighted time series for a given state based on dynamic
    weights for Trump and Biden.
    """
    trump_series = prepare_weighted_time_series(
        trump_data, state_code, trump_weights, time_interval)
    biden_series = prepare_weighted_time_series(
        biden_data, state_code, biden_weights, time_interval)
    combined_data = pd.concat([trump_series, biden_series], axis=1)
    combined_data.columns = ['trump_sentiment', 'biden_sentiment']
    return combined_data


def perform_var_analysis(state_data, confidence=0.95):
    """
    Perform VAR analysis with proper checks for NaN and lag selection.

    """
    # Drop NaN values to ensure proper model fitting
    state_data = state_data.dropna(how='any')

    # Check for stationarity and proceed
    if state_data.empty:
        raise ValueError("Time series data is empty after dropping NaNs.")

    # Select optimal lag length using criteria
    model = VAR(state_data)
    lag_order = model.select_order(maxlags=5)  # Select optimal lag
    print("Optimal lag order selected using criteria:")
    print(lag_order.summary())

    # Fit the VAR model using the optimal lag or just 1 as a fallback
    optimal_lag = lag_order.aic if lag_order.aic > 0 else 1
    results = model.fit(maxlags=optimal_lag)

    # Extract coefficients and confidence intervals
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
            ci_summary += f"  {idx}: Coefficient={coef:.4f}, CI=({lb:.4f}, \
{ub:.4f})\n"

    # print(ci_summary)  # For debugging
    return results, ci_summary


def visualize_sentiment_with_spikes(state_code, state_data, voting_results,
                                    var_results, ci_summary, trump_data,
                                    biden_data):
    """
    Vizualize sentiment time series with confidence intervals and
    spike annotations.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(state_data.index, state_data['trump_sentiment'],
             label='Trump Sentiment', color='red')
    plt.plot(state_data.index, state_data['biden_sentiment'],
             label='Biden Sentiment', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Key events dates and results
    canceled_debate_dat = pd.to_datetime('2020-10-15')
    plt.axvline(x=canceled_debate_dat, color='purple',
                linestyle='--', label='Canceled Debate')
    final_debate_date = pd.to_datetime('2020-10-22')
    plt.axvline(x=final_debate_date, color='green',
                linestyle='--', label='Final Debate')
    election_date = pd.to_datetime('2020-11-03')
    plt.axvline(x=election_date, color='orange',
                linestyle='--', label='Election Date')
    state_results = voting_results[voting_results['state_abr'] == state_code]
    trump_pct = state_results['trump_pct'].values[0]
    biden_pct = state_results['biden_pct'].values[0]
    plt.text(election_date, 0.15,
             f"Trump: {trump_pct}%\nBiden: {biden_pct}%", color='purple',
             fontsize=10)

    # Mark spikes and annotate with hashtags
    for col, color, tweet_data in zip(['trump_sentiment', 'biden_sentiment'],
                                      ['red', 'blue'],
                                      [trump_data, biden_data]):
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
    plt.title(f'Sentiment Analysis with Confidence Intervals and Spikes for \
{state_code}')
    plt.xlabel('Date')
    plt.ylabel('Weighted Sentiment Polarity')
    plt.legend()
    plt.tight_layout()

    # Add figure caption for correlation matrix and CI
    caption = f"Correlation matrix of residuals:\n{var_results.resid_corr} \
\n\n{ci_summary}"
    plt.figtext(0.5, -0.3, caption, wrap=True,
                horizontalalignment='center', fontsize=10)
    plt.show()


def analyze_all_states_with_dynamic_weights(trump_data, biden_data,
                                            voting_results):
    """
    Analyze all states with dynamic hashtag weights and visualize results.
    """
    trump_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(trump_data)}
    biden_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(biden_data)}

    for state_code in voting_results['state_abr']:
        try:
            state_data = prepare_state_data(trump_data, biden_data, state_code,
                                            trump_weights, biden_weights)
            var_results, ci_summary = perform_var_analysis(state_data)
            visualize_sentiment_with_spikes(state_code, state_data,
                                            voting_results, var_results,
                                            ci_summary, trump_data, biden_data)
        except Exception as e:
            print(f"Error processing {state_code}: {e}")


def perform_granger_causality_test(state_data, max_lag=5):
    """
    Perform Granger causality tests to determine if one sentiment time series
    Granger causes the other.
    """
    # Ensure data is fully prepared and NaNs are dropped
    state_data = state_data.dropna(how='any')

    if state_data.empty:
        raise ValueError("Time series data is empty after dropping NaNs.")

    # Perform Granger causality test
    print("\nTesting if Trump's sentiment Granger-causes Biden's sentiment...")
    results_trump_to_biden = grangercausalitytests(
        state_data, maxlag=max_lag, verbose=None
    )

    print("\nTesting if Biden's sentiment Granger-causes Trump's sentiment...")
    results_biden_to_trump = grangercausalitytests(
        state_data[['biden_sentiment', 'trump_sentiment']
                   ], maxlag=max_lag, verbose=None
    )

    return results_trump_to_biden, results_biden_to_trump


def check_granger_causality_results(granger_results, candidate1, candidate2):
    """
    Check Granger causality results for all tests.
    """
    tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    for test in tests:
        if all([result[1][0][test][1] < 0.05 for result in granger_results]):
            print(f"{candidate1}'s sentiment Granger-causes \
{candidate2}'s sentiment (rejecting null hypothesis).")
        else:
            print(f"{candidate1}'s sentiment does not Granger-cause \
{candidate2}'s sentiment (cannot reject null hypothesis).")


def analyze_single_state(trump_data, biden_data, state_code,
                         voting_results):
    """
    Analyze a single state with Granger causality tests and visualize results.
    """
    trump_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(trump_data)}
    biden_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(biden_data)}

    try:
        # Prepare weighted data
        state_data = prepare_state_data(trump_data, biden_data, state_code,
                                        trump_weights, biden_weights)

        # Perform ADF Tests and differencing if necessary
        state_data = analyze_stationarity_and_differencing(state_data)

        # Run VAR analysis
        var_results, ci_summary = perform_var_analysis(state_data)

        # Perform Granger causality test
        granger_results = perform_granger_causality_test(state_data)

        # Visualize results
        visualize_sentiment_with_spikes(state_code, state_data, voting_results,
                                        var_results, ci_summary, trump_data,
                                        biden_data)

        # Do hypothesis testing on Granger causality results
        check_granger_causality_results(granger_results, candidate1='Trump',
                                        candidate2='Biden')
        # TODO: fix the biden -> trump causality test
        check_granger_causality_results(granger_results, candidate1='Biden',
                                        candidate2='Trump')

    except Exception as e:
        print(f"Error processing {state_code}: {e}")


def perform_granger_causality_with_early_exit(trump_data, biden_data,
                                              voting_results, max_lag=5):
    """
    Perform Granger causality testing with early stopping upon sufficient
    evidence to reject the null hypothesis.
    """
    trump_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(trump_data)}
    biden_weights = {tag: sentiment['average_sentiment'] for tag, sentiment in
                     identify_dominant_hashtags(biden_data)}

    # Loop over all states
    for state_code in voting_results['state_abr']:
        try:
            # Prepare weighted sentiment data for the given state
            state_data = prepare_state_data(trump_data, biden_data, state_code,
                                            trump_weights, biden_weights)

            # Perform ADF differencing/stationarity tests
            state_data = analyze_stationarity_and_differencing(state_data)

            # Run Granger causality tests
            results = perform_granger_causality_test(
                state_data, max_lag=max_lag)

            # Extract p-values
            p_values = {
                'ssr_ftest': results[0][1][0]['ssr_ftest'][1],
                'ssr_chi2test': results[0][1][0]['ssr_chi2test'][1],
                'lrtest': results[0][1][0]['lrtest'][1],
                'params_ftest': results[0][1][0]['params_ftest'][1]
            }
            p_values_biden_to_trump = {
                'ssr_ftest': results[1][1][0]['ssr_ftest'][1],
                'ssr_chi2test': results[1][1][0]['ssr_chi2test'][1],
                'lrtest': results[1][1][0]['lrtest'][1],
                'params_ftest': results[1][1][0]['params_ftest'][1]
            }

            # Check for sufficient evidence to reject null hypothesis
            if all(p < 0.05 for p in p_values.values()) and \
                    all(p < 0.05 for p in p_values_biden_to_trump.values()):
                print(f"Evidence found in state {state_code}: \
                      Trump_to_Biden p-values={p_values}, "
                      f"Biden_to_Trump p-values={p_values_biden_to_trump}")
                print("Null hypothesis rejected. Exiting early.")
                return {
                    'state': state_code,
                    'Trump_to_Biden_pvalues': p_values,
                    'Biden_to_Trump_pvalues': p_values_biden_to_trump
                }

        except Exception as e:
            print(f"Error analyzing state {state_code}: {e}")

    print("No evidence found in any state. Thus the null hypothesis cannot be \
rejected.")
    return None


def main():
    # Load datasets
    trump_tweets = pd.read_csv('../tmp/cleaned_hashtag_donaldtrump.csv')
    biden_tweets = pd.read_csv('../tmp/cleaned_hashtag_joebiden.csv')
    voting_results = pd.read_csv('../data/election_results/voting.csv')

    # Convert 'created_at' to datetime
    trump_tweets['created_at'] = pd.to_datetime(
        trump_tweets['created_at'], errors='coerce')
    biden_tweets['created_at'] = pd.to_datetime(
        biden_tweets['created_at'], errors='coerce')

    # Analyze single state with Granger causality
    analyze_single_state(trump_tweets, biden_tweets, 'MN', voting_results)

    # Perform Granger causality for every state with early exit
    # if a null hypothesis is rejected
    perform_granger_causality_with_early_exit(trump_tweets, biden_tweets,
                                              voting_results)

    # Analyze all states
    analyze_all_states_with_dynamic_weights(trump_tweets, biden_tweets,
                                            voting_results)


if __name__ == '__main__':
    main()
