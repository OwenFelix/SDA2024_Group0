"""
HEADER
TODO : FILL WITH DESCRIPTION OF CONTENT OF FILE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

trump_df = pd.read_csv("../data/tweets/cleaned_hashtag_joebiden.csv")
biden_df = pd.read_csv("../data/tweets/cleaned_hashtag_donaldtrump.csv")

# First we apply a stationarity test to the time series data
def stationarity_test():
    pass


# - **Normalize or Scale:** (Optional) Normalize the sentiment time series for better model performance.