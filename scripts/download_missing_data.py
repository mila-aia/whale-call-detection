import os
from datetime import datetime
import pandas as pd
import wget
from tqdm import tqdm
from download_data_ftp import MSEED_to_SAC
from pathlib import Path


MISSING_FILE_LOCATION = (
    "/network/projects/aia/whale_call/list_missing_files.txt"
)
OUTPUT_DIR = "/network/projects/aia/whale_call"
FTP_URL = "ftp://ftp.seismo.nrcan.gc.ca/"
NETWORK = "CN"
YEAR_TO_DIR = {
    "2020": "wfdata5",
    "2021": "wfdata6",
    "2022": "wfdata6",
}

# Load list of .icloud files
with open(MISSING_FILE_LOCATION) as f:
    lines = f.readlines()

# Check wether some of these files already exist
files_not_on_cluster = []
for line in lines:
    # Get name
    line = line.rstrip("\n")
    line = line[:-7]
    listline = line.split("/")
    listline[-1] = listline[-1][1:]
    without_cloud = "/".join(listline)
    # Check if file exists
    if os.path.isfile(without_cloud):
        a = 0
    else:
        files_not_on_cluster.append(without_cloud.split("/"))

# Collect name of files to download from ftp server
file_names = [line.strip("\n\r").split("/") for line in lines]

# Loop omn list of files and download them
for file in files_not_on_cluster:
    full_name = file[6].split(".")
    station_name = full_name[4]
    coordinates = full_name[6]
    date_str = full_name[0] + "-" + full_name[1] + "-" + full_name[2]
    date_p = full_name[0] + "." + full_name[1] + "." + full_name[2]

    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = datetime.strptime(date_str, "%Y-%m-%d")
    sta_list = station_name
    channel_list = coordinates

    output_dir = Path(OUTPUT_DIR)

    MSEED_output = output_dir / "MSEED"
    SAC_output = output_dir / "SAC"
    MSEED_output.mkdir(parents=True, exist_ok=True)
    SAC_output.mkdir(parents=True, exist_ok=True)

    for date in tqdm(pd.date_range(start_date, end_date)):
        year_requested = str(date.year)
        day_of_year = date.strftime("%j")
        for station in sta_list:
            for channel in channel_list:
                mseed_file_name = (
                    NETWORK
                    + "."
                    + station
                    + ".."
                    + channel
                    + ".D."
                    + year_requested
                    + "."
                    + day_of_year
                )
                file_url = (
                    FTP_URL
                    + YEAR_TO_DIR[year_requested]
                    + "/"
                    + year_requested
                    + "/"
                    + NETWORK
                    + "/"
                    + station
                    + "/"
                    + channel
                    + ".D/"
                    + mseed_file_name
                )
                try:  # nosec
                    wget.download(file_url, out=str(output_dir / "MSEED"))
                except:  # noqa
                    pass
                else:
                    MSEED_to_SAC(
                        str(output_dir / "MSEED" / mseed_file_name),
                        out_dir=str(output_dir / "SAC"),
                    )
