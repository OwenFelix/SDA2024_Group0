"""
timeseries_regression.py

DESCRIPTION:
This script reads in the timeseries features and election results data
and trains a logistic regression model to predict the winning candidate's
color in the swing states. The script also evaluates the model on the
swing states and tests the robustness of the model by adding noise to
the features.
"""

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import pickle  # For loading the data
from sklearn.linear_model import LogisticRegression  # For logistic regression
# For model evaluation
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
# For data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline  # For building a pipeline
from scipy.stats import ttest_1samp  # For hypothesis testing


def load_data():
    """
    Load in the timeseries features and election results data.

    Returns
    -------
    dict, pd.DataFrame
        The timeseries features and election
    """
    with open('../tmp/features.pkl', 'rb') as f:
        timeseries_features = pickle.load(f)
    with open('../data/election_results/voting.csv', 'rb') as f:
        voting = pd.read_csv(f)
    return timeseries_features, voting


def get_election_results(voting):
    """
    Extract the definitive election results from the voting data.

    Parameters
    ----------
    voting : pd.DataFrame
        The voting data.

    Returns
    -------
    dict
        A dictionary mapping state abbreviations to
        the winning candidate's color.
    """
    definitive_results = voting[voting['trump_win'] != voting['biden_win']]
    definitive_results = definitive_results[[
        'state_abr', 'trump_win', 'biden_win']]
    definitive_results['color'] = 'red'
    definitive_results.loc[definitive_results['trump_win']
                           == 1, 'color'] = 'blue'
    return dict(zip(definitive_results['state_abr'],
                    definitive_results['color']))


def get_state_color_map():
    """
    Get a map of states to their respective colors.
    """
    return {
        'CA': 'blue', 'NY': 'blue', 'IL': 'blue', 'WA': 'blue',
        'MI': 'swing', 'TX': 'red', 'FL': 'red', 'GA': 'red',
        'OH': 'red', 'NC': 'swing', 'PA': 'swing', 'AZ': 'red',
        'MA': 'blue', 'NJ': 'blue', 'VA': 'blue', 'TN': 'red',
        'IN': 'red', 'MO': 'red', 'MD': 'blue', 'WI': 'swing',
        'MN': 'swing', 'CO': 'swing', 'AL': 'red', 'SC': 'red',
        'SD': 'red', 'OK': 'red', 'DC': 'blue', 'LA': 'red',
        'LA': 'red', 'KY': 'red', 'OR': 'blue', 'CT': 'blue',
        'IA': 'swing', 'MS': 'red', 'AR': 'red', 'UT': 'red',
        'NV': 'swing', 'KS': 'red', 'NM': 'swing', 'NE': 'red',
        'ID': 'red', 'WV': 'red', 'HI': 'blue', 'ME': 'swing',
        'NH': 'swing', 'MT': 'red', 'RI': 'blue', 'DE': 'blue',
        'ND': 'red', 'AK': 'red', 'VT': 'blue', 'WY': 'red', 'SD': 'red',
        'OK': 'red', 'DC': 'blue'
    }


def extract_features(timeseries_features, state_color_map, election_results):
    """
    Extract the features from the timeseries data read via the pickle file.

    Parameters
    ----------
    timeseries_features : dict
        The timeseries features data.
    state_color_map : dict
        A map of states to their respective colors.
    election_results : dict
        A dictionary mapping state abbreviations to
        the winning candidate's color.

    Returns
    -------
    np.array, np.array, np.array, np.array
        The training features, training labels, swing features,
        and swing labels.
    """
    train_states = [state for state in timeseries_features.keys()
                    if state_color_map[state] == 'blue' or
                    state_color_map[state] == 'red']
    swing_states = [state for state in timeseries_features.keys()
                    if state_color_map[state] == 'swing']

    X_train = np.array([list(timeseries_features[state].values())
                       for state in train_states])
    y_train = np.array([state_color_map[state] for state in train_states])

    X_swing = np.array([list(timeseries_features[state].values())
                       for state in swing_states])
    y_swing = np.array([election_results[state] for state in swing_states])

    return X_train, y_train, X_swing, y_swing, swing_states


def encode_labels(y_train, y_swing):
    """
    Encode the labels using a LabelEncoder.

    Parameters
    ----------
    y_train : np.array
        The training labels.
    y_swing : np.array
        The swing labels.

    Returns
    -------
    np.array, np.array, LabelEncoder
        The encoded training labels, encoded swing labels,
        and the label encoder.
    """
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_swing = label_encoder.transform(y_swing)
    return y_train, y_swing, label_encoder


def build_pipeline():
    """
    Build a scikit-learn pipeline for the model.

    Returns
    -------
    Pipeline
        The scikit-learn pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])


def hyperparameter_tuning(pipeline, X_train, y_train):
    """
    Perform hyperparameter tuning for the model. The hyperparameters
    to tune are the regularization strength C and the penalty type
    for the logistic regression model.

    Parameters
    ----------
    pipeline : Pipeline
        The scikit-learn pipeline.
    X_train : np.array
        The training features.
    y_train : np.array
        The training labels.
    """
    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l2']
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(final_model, X_train, y_train,
                   X_swing, y_swing, label_encoder, swing_states):
    """
    Evaluate the model on the swing states.

    Parameters
    ----------
    final_model : Pipeline
        The final model.
    X_train : np.array
        The training features.
    y_train : np.array
        The training labels.
    X_swing : np.array
        The swing features.
    y_swing : np.array
        The swing labels.
    label_encoder : LabelEncoder
        The label encoder.
    """
    swing_predictions = final_model.predict(X_swing)
    swing_predictions_decoded = label_encoder.inverse_transform(
        swing_predictions)
    print(f"Predicted swing states: {swing_predictions_decoded}")

    # Compare decoded predictions with y_swing and return the indices of
    # the incorrect predictions to print the corresponding swing states
    incorrect_predictions = np.where(swing_predictions != y_swing)[0]
    incorrect_swing_states = [swing_states[i] for i in incorrect_predictions]
    print(f"Incorrect swing states: {incorrect_swing_states}")

    accuracy = np.mean(swing_predictions == y_swing)
    print(f'Swing state accuracy: {accuracy:.2f}')

    cv_accuracy = cross_val_score(final_model, X_train, y_train, cv=5,
                                  scoring='accuracy', n_jobs=-1)
    print(f'Cross-validation accuracy: {np.mean(cv_accuracy):.2f}')

    f1 = f1_score(y_swing, swing_predictions)
    print(f'F1 score: {f1:.2f}')


def test_robustness(final_model, X_train, y_train,
                    X_swing, y_swing, swing_states):
    """
    Test the robustness of the model by adding noise to the features.
    Perform hypothesis testing to check if the model performs significantly
    better than random guessing.
    """
    n = 1000
    n_swing_states = 14
    accuracy = []
    cv_accuracy = []
    f1 = []
    average_predictions_per_state = []

    for _ in range(n):
        # Add noise to the features
        X_train_noisy = X_train + np.random.normal(0, 0.1, X_train.shape)
        X_swing_noisy = X_swing + np.random.normal(0, 0.1, X_swing.shape)

        # Fit the model
        final_model.fit(X_train_noisy, y_train)

        # Evaluate the model
        swing_predictions = final_model.predict(X_swing_noisy)
        incorrect_predictions = np.where(swing_predictions != y_swing)[0]
        average_predictions_per_state.append(swing_predictions)

        accuracy.append(1.0 - len(incorrect_predictions) / n_swing_states)
        cv_accuracy.append(np.mean(cross_val_score(
            final_model, X_train_noisy, y_train, cv=2)))
        f1.append(f1_score(y_swing, swing_predictions))

    # Convert to numpy arrays
    accuracy = np.array(accuracy)
    cv_accuracy = np.array(cv_accuracy)
    f1 = np.array(f1)

    # Calculate average predictions per state
    average_color_predictions = [sum(x) / len(x)
                                 for x in zip(*average_predictions_per_state)]
    # Now for each element, clamp to 0 or 1
    average_color_predictions = [
        0 if x < 0.5 else 1 for x in average_color_predictions]
    incorrect_predictions = np.where(average_color_predictions != y_swing)[0]
    incorrect_swing_states = [swing_states[i] for i in incorrect_predictions]
    print(f"Incorrect swing states: {incorrect_swing_states}")

    # Calculate average metrics
    prediction_accuracy = 1.0 - len(incorrect_predictions) / n_swing_states
    print(f'Average swing state accuracy: {prediction_accuracy:.2f}')
    print(f'Average cross-validation accuracy: {np.mean(cv_accuracy):.2f}')
    print(f'Average F1 score: {np.mean(f1):.2f}')

    # Calculate 95% confidence intervals
    print(
        f'Swing state accuracy 95% confidence interval:'
        f'{np.percentile(accuracy, [2.5, 97.5])}'
    )
    print(
        f'Cross-validation accuracy 95% confidence interval:'
        f'{np.percentile(cv_accuracy, [2.5, 97.5])}')
    print(
        f'F1 score 95% confidence interval: {np.percentile(f1, [2.5, 97.5])}')

    # Perform hypothesis testing (H₀: mean = 0.50, H₁: mean > 0.50)
    prediction_accuracy_p_value = ttest_1samp(
        accuracy, popmean=0.50, alternative='greater').pvalue
    cv_accuracy_p_value = ttest_1samp(
        cv_accuracy, popmean=0.50, alternative='greater').pvalue
    f1_p_value = ttest_1samp(f1, popmean=0.50, alternative='greater').pvalue

    print(f'Swing state accuracy p-value: {prediction_accuracy_p_value:.4f}')
    print(f'Cross-validation accuracy p-value: {cv_accuracy_p_value:.4f}')
    print(f'F1 score p-value: {f1_p_value:.4f}')

    if prediction_accuracy_p_value < 0.05:
        print('Reject null hypothesis for swing state accuracy')
    else:
        print('Fail to reject null hypothesis for swing state accuracy')
    if cv_accuracy_p_value < 0.05:
        print('Reject null hypothesis for cross-validation accuracy')
    else:
        print('Fail to reject null hypothesis for cross-validation accuracy')

    if f1_p_value < 0.05:
        print('Reject null hypothesis for F1 score')
    else:
        print('Fail to reject null hypothesis for F1 score')


def main():
    # Load the data
    timeseries_features, voting = load_data()
    election_results = get_election_results(voting)
    state_color_map = get_state_color_map()

    # Extract features and labels
    X_train, y_train, X_swing, y_swing, swing_states = extract_features(
        timeseries_features, state_color_map, election_results)

    # Encode labels
    y_train, y_swing, _ = encode_labels(y_train, y_swing)

    # Build the logistic regression pipeline
    pipeline = build_pipeline()
    best = hyperparameter_tuning(pipeline, X_train, y_train)

    # Test the robustness of the model and evaluate it
    test_robustness(best, X_train,
                    y_train, X_swing, y_swing, swing_states)


if __name__ == "__main__":
    main()
