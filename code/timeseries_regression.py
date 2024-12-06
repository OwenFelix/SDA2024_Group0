import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Load in the timeseries data
with open('../data/features.pkl', 'rb') as f:
    timeseries_features = pickle.load(f)

# Load in election results
with open('../data/election_results/voting.csv', 'rb') as f:
    voting = pd.read_csv(f)

# Make use of the voting data to make a state color map for the results
definitive_results = voting[voting['trump_win'] != voting['biden_win']]
definitive_results = definitive_results[[
    'state_abr', 'trump_win', 'biden_win']]
definitive_results['color'] = 'red'
definitive_results.loc[definitive_results['trump_win'] == 1, 'color'] = 'blue'
election_results = dict(
    zip(definitive_results['state_abr'], definitive_results['color']))

state_color_map = {
    'CA': 'blue', 'NY': 'blue', 'IL': 'blue', 'WA': 'blue',
    'MI': 'swing', 'TX': 'red', 'FL': 'swing', 'GA': 'red',
    'OH': 'swing', 'NC': 'swing', 'PA': 'swing', 'AZ': 'swing',
    'MA': 'blue', 'NJ': 'blue', 'VA': 'blue', 'TN': 'red',
    'IN': 'red', 'MO': 'red', 'MD': 'blue', 'WI': 'swing',
    'MN': 'swing', 'CO': 'swing', 'AL': 'red', 'SC': 'red',
    'LA': 'red', 'KY': 'red', 'OR': 'blue', 'CT': 'blue',
    'IA': 'swing', 'MS': 'red', 'AR': 'red', 'UT': 'red',
    'NV': 'swing', 'KS': 'red', 'NM': 'swing', 'NE': 'red',
    'ID': 'red', 'WV': 'red', 'HI': 'blue', 'ME': 'swing',
    'NH': 'swing', 'MT': 'red', 'RI': 'blue', 'DE': 'blue',
    'ND': 'red', 'AK': 'red', 'VT': 'blue', 'WY': 'red'
}

# First we need to extract the swing states
train_states = [state for state in timeseries_features.keys()
                if state_color_map[state] == 'blue' or state_color_map[state] == 'red']

swing_states = [state for state in timeseries_features.keys()
                if state_color_map[state] == 'swing']

# Now we need to extract the features for the red and blue states
X_train = np.array([list(timeseries_features[state].values())
                    for state in train_states])
y_train = np.array([state_color_map[state] for state in train_states])

X_swing = np.array([list(timeseries_features[state].values())
                    for state in swing_states])
y_swing = np.array([election_results[state] for state in swing_states])

# Encode the labels numerically
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_swing = label_encoder.transform(y_swing)

# Scale features using a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {np.mean(scores):.2f}')

# Hyperparameter tuning
param_grid = {
    'model__C': [0.15, 0.175, 0.2, 0.1],
    'model__penalty': ['l2']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")

# Train the final model using the best parameters
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

# Predict swing states
swing_predictions = final_model.predict(X_swing)
swing_predictions_decoded = label_encoder.inverse_transform(swing_predictions)
print(f"Predicted swing states: {swing_predictions_decoded}")

# Calculate accuracy on swing states
accuracy = np.mean(swing_predictions == y_swing)
print(f'Swing state accuracy: {accuracy:.2f}')
