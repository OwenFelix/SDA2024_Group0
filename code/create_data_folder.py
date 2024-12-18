"""
    create_data_folder.py

    DESCRIPTION:
    This script downloads the necessary data for our project and
    extracts it to the correct folders. The data is downloaded from
    a shorturl link and extracted to the data folder in the project
    directory.
"""

import zipfile
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
    Path(__file__).parent.joinpath("..", "data", "election_results").mkdir(
        parents=True, exist_ok=True)
    Path(__file__).parent.joinpath("..", "data", "tweets").mkdir(
        parents=True, exist_ok=True)

    # Define the file names and URLs
    fn_results = Path(__file__).parent / ".." / "data" / \
        "election_results" / "election_results.csv"
    print(fn_results)
    fn_tweets = Path(__file__).parent / ".." / "data" / "tweets" / "tweets.csv"

    URL_results = r"https://drive.usercontent.google.com/download?id=10XQTzZJ8YA71moJRG-9_G5d4ksAfXtMF&export=download"
    URL_tweets = r"https://drive.usercontent.google.com/download?id=1zJdIYZGsjPztCL4UUD1cZKmTPmkHFG61&export=download&confirm=t&uuid=abd7d491-8765-4d14-affb-9540315ce4ed"

    # Check if files are already downloaded
    if (Path(__file__).parent / "data" /
            "election_results" / "voting.csv").exists():
        print("Data already downloaded")
    else:
        # Download and extract the data
        read_and_extract_data(URL_results, fn_results)

    if (Path(__file__).parent / ".." / "data" / "tweets" /
        "hashtag_donaldtrump.csv").exists() and \
            (Path(__file__).parent / ".." / "data" / "tweets" /
             "hashtag_joebiden.csv").exists():
        print("Data already downloaded")
    else:
        print("Downloading tweets. Warning: This may take a minute or two")
        read_and_extract_data(URL_tweets, fn_tweets)
