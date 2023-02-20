from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    Namespace,
)
from pathlib import Path
import pandas as pd
import scipy.io
from datetime import timedelta
import random
import yaml


def read_stadir(mat_file: dict, whale_type: str) -> dict:
    """
    Return name of all stations stored in stadir var
    in the .mat file

    @param: mat_file (String): path + name of .mat file
    @param: whale_type (String): type of whales

    return: dict containing station names with index as key
    and value as station name
    """
    list_stations = [
        name_station[0] for name_station in mat_file[whale_type][:, 0]
    ]
    return {v + 1: k for v, k in enumerate(list_stations)}


def preprocess_(dataset: pd.DataFrame, stadir: dict) -> pd.DataFrame:
    """Preprocess labels

    @param: dataset (pd.DataFrame): labels
    @param: stadir (dict): name and index of stations

    return: pd.Dataframe preprocessed labels
    """

    # Time
    dataset["datetime"] = pd.to_datetime(dataset["Datenum"] - 719529, unit="D")
    dataset["time"] = dataset["datetime"].dt.time
    dataset["date"] = dataset["datetime"].dt.date
    dataset["datetime_start"] = pd.to_datetime(
        dataset["time_start"].round(), unit="s"
    ).dt.time

    # Convert column types
    integer_columns = [
        "num_calls_in_detection",
        "group_index",
        "station_number",
        "detection_id",
    ]
    dataset[integer_columns] = dataset[integer_columns].astype(int)

    # Station number
    dataset["station_name"] = dataset["station_number"].map(stadir)

    return dataset


def read_list_raw_files(sac_file_path: str) -> pd.DataFrame:
    """Build a dataframe with the list of the SAC files

    @param: sac_file_path (string): PATH to the txt file containing the list

    return: pd.Dataframe with 2 columns: folder containing files, and PATH
    """
    # Read list of raw data files
    with open(sac_file_path) as file:
        list_files = file.readlines()
        list_files = [line.rstrip("\n") for line in list_files]
        list_files_df = pd.Series(list_files)

    # Create df from list of files
    list_files_detailled = pd.DataFrame(
        list_files_df.str.split("/", n=7, expand=True)[[6, 7]]
    ).rename(columns={6: "folder_date", 7: "file_name"})

    # Rename columns
    list_files_detailled = list_files_detailled["file_name"].str.split(
        ".", n=7, expand=True
    )
    list_files_detailled = list_files_detailled.rename(
        {
            0: "year",
            1: "month",
            2: "day",
            3: "network",
            4: "station",
            5: "empty",
            6: "coordinates",
            7: "SAC",
        },
        axis=1,
    )
    # Extract folder name
    list_files_detailled["folder"] = list_files_df.str.split(
        "/", n=7, expand=True
    )[[6]]
    # Add list of names to path
    list_files_detailled["file_path"] = list_files_df

    return list_files_detailled


def main() -> None:
    """ """

    # Arguments parsing
    args = parse_args()

    # Load config
    with open("config/config.yml", "r") as file:
        param_data = yaml.safe_load(file)

    # Output
    labels_output = Path(param_data["paths"]["whale_data_cluster"]) / "LABELS"

    labels_output.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist

    # Input
    input_file = Path(args.input_file)

    # Load .mat file
    mat = scipy.io.loadmat(
        input_file, variable_names=["FWC", "BWC", "stadir", "stadir_bw"]
    )

    # Load Fin & Blue whale calls from .mat
    colnames_WC = [
        "Datenum",
        "station_number",
        "R",
        "SNR",
        "group_index",
        "time_start",
        "num_calls_in_detection",
        "detection_id",
    ]

    # Get list of files in dataframe
    if args.bandpass_filter == "True":
        list_files_detailled = read_list_raw_files(
            "/network/projects/aia/whale_call/SAC_FILES_FILT.txt"
        )
    else:
        list_files_detailled = read_list_raw_files(
            "/network/projects/aia/whale_call/SAC_FILES_RAW.txt"
        )

    # Loop for the 2 whale types
    for whale_type in ["bw", "fw"]:

        # Get data from matlab .mat matrix
        df_calls = pd.DataFrame(
            mat[whale_type.upper() + "C"], columns=colnames_WC
        )

        # Load name and index of stations
        stadir = read_stadir(
            mat, param_data["whale_constant"][whale_type]["stadir_name"]
        )

        # Preprocess labels
        labels = preprocess_(df_calls, stadir)

        # Plot counts of calls
        print(
            "Number of {} calls detected: {}".format(
                param_data["whale_constant"][whale_type]["name"],
                labels.detection_id.nunique(),
            )
        )

        # Add start and end time of calls
        labels["time_call_start"] = pd.to_datetime(labels["datetime"])
        labels["time_call_end"] = labels["time_call_start"] + timedelta(
            seconds=param_data["whale_constant"][whale_type]["call_duration"]
        )

        # Reformat folder name
        labels["folder_date"] = (
            labels["date"].astype(str).apply(lambda x: "".join(x.split("-")))
        )

        # Add column with Whale type
        labels["whale_type"] = whale_type

        # Merge labels and SAC file PATHs to same dataframe
        final_df = pd.merge(
            labels,
            list_files_detailled,
            left_on=["folder_date", "station_name"],
            right_on=["folder", "station"],
        ).rename(
            columns={
                "coordinates": "component",
                "station": "station_code",
                "detection_id": "group_id",
            }
        )

        # Save results to dataframe
        if args.bandpass_filter == "True":
            csv_name = whale_type + "_filt.csv"
            csv_name_grouped = whale_type + "_component_grouped_filt.csv"
        else:
            csv_name = whale_type + "_raw.csv"
            csv_name_grouped = whale_type + "_component_grouped_raw.csv"

        # Add random value to create start and end time of window
        list_randoms = []
        for _ in final_df.index:
            rand_num = random.uniform(  # nosec
                0,
                param_data["whale_constant"][whale_type]["window_size"]
                - param_data["whale_constant"][whale_type]["call_duration"],
            )
            list_randoms.append(rand_num)

        final_df["random_t"] = list_randoms
        final_df["time_window_start"] = final_df.apply(
            lambda x: x.time_call_start - timedelta(seconds=x.random_t), axis=1
        )
        final_df["time_window_end"] = final_df[
            "time_window_start"
        ] + timedelta(seconds=5)

        grouped_df = final_df.groupby(["station_code", "time_call_start"]).agg(
            list_components=("component", lambda x: " ".join(x)),
            number_components=("component", "count"),
            R=("R", "max"),
            SNR=("SNR", "max"),
            file_path=(
                "file_path",
                lambda x: [a[:-7] + "CHANNEL" + a[-4:] for a in x][0],
            ),
            time_window_start=(
                "time_window_start",
                lambda x: [a for a in x][0],
            ),
            time_window_end=("time_window_end", lambda x: [a for a in x][0]),
            time_call_end=("time_call_end", "min"),
            whale_type=("whale_type", "min"),
        )

        grouped_df.to_csv(
            labels_output / whale_type.upper() / csv_name_grouped, index=False
        )

        final_df[
            [
                "file_path",
                "time_window_start",
                "time_window_end",
                "time_call_start",
                "time_call_end",
                "R",
                "SNR",
                "group_id",
                "station_code",
                "whale_type",
                "component",
            ]
        ].to_csv(labels_output / whale_type.upper() / csv_name, index=False)


def parse_args() -> Namespace:
    """Parse arguments"""
    description = "Script for preprocessing the labels coming from .mat matrix"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--bandpass_filter",
        default="True",
        type=str,
        help="True if you want to use data with applied bandpass filter",
    )

    arg_parser.add_argument(
        "--input_file",
        default="/network/projects/aia/whale_call/"
        + "MATLAB_OUTPUT/WhaleDetectionsLSZ_new.mat",
        type=str,
        help="path to the input file",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
