"""
    Name : Remco Hogerwerf
    UvAnetID : 14348462
    Study : Bsc Computer Science

    create_data_folder.py

    This script downloads the necessary data for our project and
    extracts it to the correct folders. The data is downloaded from
    a shorturl link and extracted to the data folder in the project
    directory.

    DEPENDENCIES:
    - requests
    - zipfile
    - shutil
"""

import zipfile
import pandas as pd
import requests
import shutil
from pathlib import Path


def read_and_extract_data(URL, fn):
    if fn.exists():
        print("Data already downloaded")
        return

    # Download the data
    r = requests.get(URL, stream=True)
    if not r.status_code == 200:
        print("Failed to download data")
        return

    # Save the data to a zip file
    with open(fn.with_suffix(".zip"), 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)

    # Extract the zip file and remove it
    with zipfile.ZipFile(fn.with_suffix(".zip"), 'r') as zip_ref:
        zip_ref.extractall(fn.parent)
    fn.with_suffix(".zip").unlink()

    print("Downloaded and extracted data")


if __name__ == "__main__":
    # First make sure the data folder and its subfolders exist
    Path(__file__).parent.joinpath("data", "election_results").mkdir(
        parents=True, exist_ok=True)
    Path(__file__).parent.joinpath("data", "tweets").mkdir(
        parents=True, exist_ok=True)
    Path(__file__).parent.joinpath("data", "covid").mkdir(
        parents=True, exist_ok=True)

    # Define the file names and URLs
    fn_results = Path(__file__).parent / "data" / \
        "election_results" / "election_results.csv"
    fn_tweets = Path(__file__).parent / "data" / "tweets" / "tweets.csv"
    fn_covid = Path(__file__).parent / "data" / "covid" / "covid.csv"

    URL_results = r"https://shorturl.at/Tf8WV"
    URL_tweets = r"https://shorturl.at/sStbw"
    URL_covid = r"https://shorturl.at/HJBnO"

    # Download and extract the data
    # read_and_extract_data(URL_results, fn_results)
    print("Downloading tweets. Warning: This may take a minute or two")
    # read_and_extract_data(URL_tweets, fn_tweets)
    read_and_extract_data(URL_covid, fn_covid)

    # Filter out a lot of the covid data

    path = Path(__file__).parent / "data" / "covid" / "us-counties.csv"
    df = pd.read_csv(path)

    # We only want dates between 2020-10-15 and 2020-11-8
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].between('2020-10-15', '2020-11-7')]
    df.to_csv(path, index=False)

    print("Done!")
