# SDA2024_Group0

This project is part of the course "Scientific Data Analysis" at the University of Amsterdam.

## Group members
* Remco Hogerwerf
* Mabel Traube
* Owen Poort
* Annabelle Donato

## Project description
This project aims to analyze the sentiment of tweets during the Americam Presidential Elections of 2020. We use a dataset of tweets, which we filter to only have English tweets that were sent from the United States. We then analyze the sentiment of each tweet and divide them into the different states. We created time series plots of the sentiment of each state. To the time series, we have applied various statistical models and methods, including k-means clustering, logistic regression and Vector Autoregression (VAR).

## Environment creation
To ensure compatibility and proper functioning of the code, we recommend creating a dedicated environment using anaconda. Follow these steps:

1. Create an environment using the provided ``environment.yml`` file:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate sdagroup0
```
3. When finished, deactivate the environment:
```bash
conda deactivate
```

## How to use
1. Clone the repository
2. Use the `startup.py` to create and process the data. Run the following command in the terminal if you want to download the cleaned data directly:
```bash
python3 startup.py False
```
Run the following command if you want to download the raw data and clean it yourself (takes about 15 - 20 minutes):
```bash
python3 startup.py True
```

3. To see the interactive sentiment time series plots, run the following command in the terminal:
```bash
python3 interactive_plot.py
```
This script also creates a video of the sentiment time series plots.

4. To see a static plot for the results of the elections, run the following command in the terminal:
```bash
python3 geography_plots.py
```
This command also shows the sentiment plots using Matplotlib, which do not use the real data.

5. To see a plot for the accuracy of the sentiment data compared to the actual outcome of the elections, run the following command:
```bash
python3 analyze_geodata.py
```

6. To see the result of the k-means clustering of the time series data, run the following command:
```bash
python3 timeseries_clustering.py
```

7. To see the result of the DTW-clustering of the time series data, run the following command:
```bash
python3 timeseries_dtwclustering.py
```

8. To see the results of the logistic regression, run the following command:
```bash
python3 timeseries_regression.py
```

9. To see the results of the VAR analysis, run the following command:
```bash
python3 VAR_analysis.py
```

## Code Style and Formatting
All code is formatted according to the PEP8 style guide and adheres to flake8 standards. Proper documentation is provided throughout the codebase to ensure readability and maintainability.

## Sources
- [Election results dataset](https://www.kaggle.com/datasets/callummacpherson14/2020-us-presidential-election-results-by-state)
- [Tweets dataset](//www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
- [UK General election 2017 twitter analysis](https://arxiv.org/abs/1706.02271)
- [Sentiment analysis of tweets based on US presidential elections of 2016 and UK General election 2017](https://dl.acm.org/doi/10.1145/3339909)
- [2020 US election key events](https://edition.cnn.com/election/2020/events)


