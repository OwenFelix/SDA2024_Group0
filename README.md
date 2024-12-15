# SDA2024_Group0

This project is part of the course "Scientific Data Analysis" at the University of Amsterdam.

## Group members
* Remco Hogerwerf
* Mabel Traube
* Owen Poort 
* Annabelle Donato

## Project description
This project aims to analyze the sentiment of tweets during the Americam Presidential Elections of 2020. We use a dataset of tweets, which we filter to only have English tweets that were sent from the United States. We then analyze the sentiment of each tweet and divide them into the different states. We created time series plots od the sentiment of each state. 

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

## Dependencies
* Python 3.7 (or higher)
* pandas
* numpy
* matplotlib
* shutil
* pathlib
* requests
* zipfile
* PyQt5
* kaleido

