from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from datetime import datetime
from pathlib import Path
import pandas as pd
import wget
from obspy.core import read
from tqdm import tqdm


FTP_URL = "ftp://ftp.seismo.nrcan.gc.ca/"
NETWORK = "CN"
YEAR_TO_DIR = {
    "2020": "wfdata5",
    "2021": "wfdata6",
    "2022": "wfdata6",
}  # Mapping from year requested to director on FTP server


def MSEED_to_SAC(MSEED_file: str = None, out_dir: str = "./data") -> None:
    """
    Convert MSEED file to SAC file.
    Arguments
    ---------
    MSEED_file : str
        MSEED file name.
    out_dir : int
       Path to save the converted SAC file. (default : ./data)
    """
    st = read(MSEED_file)
    st.merge(method=0, fill_value=0)
    year = st[0].stats.starttime.year
    month = "%02d" % (st[0].stats.starttime.month)
    day = "%02d" % (st[0].stats.starttime.day)
    network = st[0].stats.network
    station = st[0].stats.station
    location = st[0].stats.location
    channel = st[0].stats.channel
    fname = "%s.%s.%s.%s.%s.%s.%s.SAC" % (
        year,
        month,
        day,
        network,
        station,
        location,
        channel,
    )
    st.write(out_dir + "/" + fname, format="SAC")


def main() -> None:

    args = parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    sta_list = args.stations.split(",")
    channel_list = args.channels.split(",")

    output_dir = Path(args.output_dir).expanduser().resolve()
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


def parse_args() -> Namespace:
    description = (
        "Script for downloading seismic data from CN ftp server"
        "The data is in MSEED format."
    )
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--output-dir",
        default="data/RAW/",
        type=str,
        help="path to the output directory",
    )
    arg_parser.add_argument(
        "--start-date",
        default=None,
        type=str,
        help="the starting date (yyyy-mm-dd) of the request window",
    )
    arg_parser.add_argument(
        "--end-date",
        default=None,
        type=str,
        help="the starting date (yyyy-mm-dd) of the request window",
    )
    arg_parser.add_argument(
        "--stations",
        default="PMAQ,ICQ,SNFQ,RISQ,SMQ,CNQ",
        type=str,
        help="the stations requested (seperated by comma)",
    )
    arg_parser.add_argument(
        "--channels",
        default="HHE,HHN,HHZ,HNE,HNN,HNZ,EHZ",
        type=str,
        help="the channels requested (seperated by comma)",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
