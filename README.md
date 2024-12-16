# SDA2024_Group0

This project is part of the course "Scientific Data Analysis" at the University of Amsterdam.

## Group members
* Remco Hogerwerf
* Mabel Traube
* Owen Poort
* Annabelle Donato

## Project description
lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec elit
nec nunc ultricies ultricies.

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


## Environment creation
To be able to run our code a variety of libraries are needed. The easiest way to manage installations is through the use of an environment. We recommend creating an environment with anaconda by doing the following steps:

First we will create the environment using:
```bash
conda env create -f environment.yml
```

Then you will need to activate the environment using:
```console
conda activate sdagroup0
```

Whenever you are done with running the code you may deactivate the environment using:
```console
conda activate sdagroup0
```

## Dependencies
* Python 3.8 (or higher)
* pandas
* numpy
* matplotlib
* shutil
* pathlib
* requests
* zipfile
* PyQt5
* kaleido

