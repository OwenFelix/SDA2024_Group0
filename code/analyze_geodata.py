"""
analyze_geodata.py

DESCRIPTION:
This file contains the code to analyze the sentiment data and compare it to the election results.
A bar plot is generated to show the accuracy of the sentiment analysis for each day.
"""

import matplotlib.pyplot as plt
import pandas as pd


time_series_biden = pd.read_csv('../tmp/time_series_data_biden.csv')
time_series_trump = pd.read_csv('../tmp/time_series_data_trump.csv')

voting_results = pd.read_csv('../data/election_results/voting.csv')

trump_states = list(
    voting_results[voting_results['trump_win'] == 1]['state_abr'])
biden_states = list(
    voting_results[voting_results['biden_win'] == 1]['state_abr'])

trump_states_results = []
biden_states_results = []

# Check who 'won' each day
for day in range(25):
    trump_day_results = []
    biden_day_results = []
    for state in voting_results['state_abr']:
        biden_sent = time_series_biden[(time_series_biden['state'] == state) & (
            time_series_biden['timestamp'] == day)]['biden_sentiment'].values[0]
        trump_sent = time_series_trump[(time_series_trump['state'] == state) & (
            time_series_trump['timestamp'] == day)]['trump_sentiment'].values[0]

        if biden_sent > trump_sent:
            biden_day_results.append(state)
        else:
            trump_day_results.append(state)

    trump_states_results.append(trump_day_results)
    biden_states_results.append(biden_day_results)

# Calculate how accurate the sentiment analysis was for each day
accuracy = []

# Count how many states were correctly predicted each day
for day in range(25):
    correct = 0
    for state in trump_states_results[day]:
        if state in trump_states:
            correct += 1

    for state in biden_states_results[day]:
        if state in biden_states:
            correct += 1

    accuracy.append(correct / 51)

plt.xlabel('Day')
plt.ylabel('Accuracy')
plt.title('Accuracy of sentiment analysis per day')
plt.bar(range(25), accuracy, color='blue')

plt.show()
