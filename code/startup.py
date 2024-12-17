import subprocess
import sys
import requests
from pathlib import Path
import shutil

def read_and_extract_data(URL, fn):
    if fn.exists():
        print("Data already downloaded")
        return

    # Download the data
    r = requests.get(URL, stream=True)
    if not r.status_code == 200:
        print("Failed to download data")
        return

    # Unpack tar file
    with open(fn.with_suffix(".tar"), 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)

    # Extract the tar file and remove it
    shutil.unpack_archive(fn.with_suffix(".tar"), fn.parent)
    fn.with_suffix(".tar").unlink()

    print("Downloaded and extracted data\n")


# Run create_data_folder.py
print("Downloading and extracting data...")
subprocess.run(["python", "create_data_folder.py"])
print("Data downloaded and extracted successfully!\n")

# Check if the user provided an argument
if len(sys.argv) < 2:
    print("Please provide an argument to specify whether to preprocess the data or not.")
    sys.exit()

# Download the clean data or preprocess the raw data
if sys.argv[1] == "True":
    # Run data_preprocessing.py
    print("Preprocessing the data...")
    print("Warning: This is going to take quite a while!")
    subprocess.run(["python", "code/data_preprocessing.py"])
elif sys.argv[1] == "False":
    print("Downloading clean data...")

    # First make sure the data folder and its subfolders exist
    Path(__file__).parent.joinpath("tmp").mkdir(
        parents=True, exist_ok=True)

    # Define the file names and URLs
    fn_clean = Path(__file__).parent / ".." / "tmp" / \
        "cleaned_data.csv"

    # Define the URL
    URL_cleaned = "https://drive.usercontent.google.com/download?id=1m-FqDOUcoZOfBkMqnX5OAfpzEQYjQSjo&export=download&authuser=0"

    if (Path(__file__).parent / ".." / "tmp" / "cleaned_hashtag_joebiden.csv").exists() and (Path(__file__).parent / "tmp" / "cleaned_hashtag_donaldtrump.csv").exists():
        print("Data already downloaded")
    else:
        # Download the data
        read_and_extract_data(URL_cleaned, fn_clean)

# Run create_timeseries.py
print("Creating timeseries data...")
print("This will take a minute or two...")
subprocess.run(["python", "code/create_timeseries.py"])
print("Timeseries data created successfully!\n")

# Run analyze_timeseries.py
print("Analyzing timeseries data...")
subprocess.run(["python", "code/analyze_timeseries.py"])
print("Timeseries data analyzed successfully\n!")
