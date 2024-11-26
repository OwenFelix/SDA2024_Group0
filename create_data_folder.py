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
import requests
import shutil
from pathlib import Path


def read_and_extract_data(URL, fn):
    if not fn.exists():
        r = requests.get(URL, stream=True)
        if r.status_code == 200:
            with open(fn.with_suffix(".zip"), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(fn.with_suffix(".zip"), 'r') as zip_ref:
                zip_ref.extractall(fn.parent)
            fn.with_suffix(".zip").unlink()
            print("Downloaded and extracted data")
        else:
            print("Failed to download data")
    else:
        print("Data already downloaded")


if __name__ == "__main__":
    # First make sure the data folder and its subfolders exist
    Path(__file__).parent.joinpath("data", "election_results").mkdir(
        parents=True, exist_ok=True)
    Path(__file__).parent.joinpath("data", "tweets").mkdir(
        parents=True, exist_ok=True)

    # Define the file names and URLs
    fn_results = Path(__file__).parent / "data" / \
        "election_results" / "election_results.csv"
    fn_tweets = Path(__file__).parent / "data" / "tweets" / "tweets.csv"

    URL_results = r"https://shorturl.at/Tf8WV"
    URL_tweets = r"https://shorturl.at/sStbw"

    # Download and extract the data
    read_and_extract_data(URL_results, fn_results)
    print("Downloading tweets. Warning: This may take a minute or two")
    read_and_extract_data(URL_tweets, fn_tweets)

    print("Done!")
