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

    # Define the file names and URLs
    fn_results = Path(__file__).parent / "data" / \
        "election_results" / "election_results.csv"
    fn_tweets = Path(__file__).parent / "data" / "tweets" / "tweets.csv"

    URL_results = r"https://storage.googleapis.com/kaggle-data-sets/1139788/1911744/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241211%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241211T113249Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8866e916621aaef49ab04f762e537dd484f8e0bfac0f67f19859174b121235c7e689bd54b43dcc46ff0afcd45abdaa6ed85cc15552d8e0fd3282e059ea8bd9568d24fbe46111f6e68918c9a8b884e785bc9399a803a0cdf0949b1a8825cf508f20dec6c6e4e300ad6107898bc7c633137a16258c64b4397b45992ff17876509ea51b0935dff5cb037e544100d86bd30e2b0859138fd4bd56d229c1f74b76e1d10ba6749c69827d8b5eff15c16a0b5c28aec38a3b5e188fb8134f85e7a79dab9fbcf3083e1078700b7b47ed0420bb0de4cf6b3de7422e870bcbf1fcdfbcfcce316ac89b84d61a670c93ec9126c175033b8bbc2c102e8a6a6b99091a086039bc99"
    URL_tweets = r"https://storage.googleapis.com/kaggle-data-sets/935914/1632066/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241211%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241211T113210Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2843edb09ce4e461ffbed13999adb43d875b129a648e1df2fd1c5dc42755b327578ba0e197e1a8055ddd87000888a08da0526ef70a45a60ab2b16ea98779dc3e92266bc5bfdf70351805205fbfafae26a51280d9c7dda23f90b3168d6ec341bf181e5c941568fc5d8b9acfc045fc9caa6d4e3daa13d6020deb24fd0bf5803cbbec10827e5528b2706a7e059f8bc027aa919e03af2ad8925ea24cf3fcda13c3b1b6c0989c69d5d112dadb9dcf71518505a93dd5cf96861febe10d07790e05ac3195bb5e469f0e00d711fcdbf2cf7f678cf380f79a15049b16b275accacd202d1430e195ea1e5e2a8b9a22683d2e3f9e41943f0a29772ae7a36a6644ad1c738a02"

    # Check if files are already downloaded
    if (Path(__file__).parent / "data" /
            "election_results" / "voting.csv").exists():
        print("Data already downloaded")
    else:
        # Download and extract the data
        read_and_extract_data(URL_results, fn_results)

    if (Path(__file__).parent / "data" / "tweets" / "hashtag_donaldtrump.csv").exists() and \
    (Path(__file__).parent / "data" / "tweets" / "hashtag_joebiden.csv").exists():
        print("Data already downloaded")
    else:
        print("Downloading tweets. Warning: This may take a minute or two")
        read_and_extract_data(URL_tweets, fn_tweets)

    print("Done!")
