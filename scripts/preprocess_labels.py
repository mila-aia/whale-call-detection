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
import numpy as np
from obspy import UTCDateTime
import itertools


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


def timeShift(input_time: str, shift: int) -> str:
    # shift the input_time by shift seconds
    shifted_time = UTCDateTime(input_time) + shift
    return UTCDateTime.strftime(shifted_time, "%Y-%m-%d %H:%M:%S.%f")


def timeDiff(input_time: str, ref_time: str) -> float:
    # calculate the time difference between input_time and ref_time
    delta_t = UTCDateTime(input_time) - UTCDateTime(ref_time)
    return delta_t


def remove_overlapping_noise_samples(
    df_noise: pd.DataFrame, full_data: pd.DataFrame
) -> pd.Series:

    df_noise["startdate"] = pd.to_datetime(df_noise["time_window_start"])
    df_noise["enddate"] = pd.to_datetime(df_noise["time_window_end"])
    full_data["startdate"] = pd.to_datetime(full_data["time_window_start"])
    full_data["enddate"] = pd.to_datetime(full_data["time_window_end"])
    df_noise = df_noise.rename(
        columns={"startdate": "startdate_noise", "enddate": "enddate_noise"}
    )
    full_data = full_data.rename(
        columns={"startdate": "startdate_fw", "enddate": "enddate_fw"}
    )

    total = []
    for df in np.array_split(df_noise, 5):

        ixs = (
            df[["startdate_noise", "enddate_noise", "file_path"]]
            .reset_index()
            .merge(
                full_data[["startdate_fw", "enddate_fw", "file_path"]],
                on=["file_path"],
            )
            .query(
                "(startdate_noise < enddate_fw) "
                + "& (enddate_noise > startdate_fw)"
            )
        )["index"].values
        total.append(ixs)

    return list(itertools.chain.from_iterable(total))


def generate_noise_samples(
    fw_filt_hq: pd.DataFrame,
    full_data: pd.DataFrame,
    noise_shift: int,
    noise_duration: int,
) -> pd.DataFrame:

    # get noise_windonw_end by shifting time_window_start back
    # by noise_shift seconds
    fw_filt_hq["noise_window_end"] = fw_filt_hq["time_window_start"].apply(
        lambda x: timeShift(x, -1 * noise_shift)
    )
    # get noise_windonw_start by shifting noise_windonw_end back
    # by noise_duration seconds
    fw_filt_hq["noise_window_start"] = fw_filt_hq["noise_window_end"].apply(
        lambda x: timeShift(x, -1 * noise_duration)
    )

    # split fw_filt_hq into two dataframes:
    # one without noise_window_start and noise_window_end;
    # the other without time_window_start and time_window_end
    fw_filt_hq_fw = fw_filt_hq[
        [
            "file_path",
            "time_window_start",
            "time_window_end",
            "time_R_max",
            "time_call_start",
            "time_call_end",
            "R",
            "SNR",
            "group_id",
            "station_code",
            "whale_type",
            "component",
        ]
    ].copy()
    fw_filt_hq_noise = fw_filt_hq[
        [
            "file_path",
            "noise_window_start",
            "noise_window_end",
            "time_R_max",
            "time_call_start",
            "time_call_end",
            "R",
            "SNR",
            "group_id",
            "station_code",
            "whale_type",
            "component",
        ]
    ].copy()
    # rename noise_window_start to time_window_start
    # and noise_window_end to time_window_end
    fw_filt_hq_noise.rename(
        columns={
            "noise_window_start": "time_window_start",
            "noise_window_end": "time_window_end",
        },
        inplace=True,
    )
    fw_filt_hq_noise["time_R_max"] = fw_filt_hq_noise["time_window_start"]
    fw_filt_hq_noise["time_call_start"] = fw_filt_hq_noise["time_window_start"]
    fw_filt_hq_noise["time_call_end"] = fw_filt_hq_noise["time_window_end"]
    fw_filt_hq_noise["R"] = 0
    fw_filt_hq_noise["SNR"] = -99
    fw_filt_hq_noise["whale_type"] = "noise"
    fw_filt_hq_noise["group_id"] = -1

    # Check if noise samples overlap with whale calls
    before = fw_filt_hq_noise.shape[0]
    ixs = remove_overlapping_noise_samples(fw_filt_hq_noise, full_data)
    fw_filt_hq_noise = fw_filt_hq_noise[~fw_filt_hq_noise.index.isin(ixs)]
    after = fw_filt_hq_noise.shape[0]
    print("Number of overlapping noise samples removed: ", before - after)

    # concat fw_filt_hq_fw and fw_filt_hq_noise
    fw_filt_hq_mixed = pd.concat(
        [fw_filt_hq_fw, fw_filt_hq_noise], ignore_index=True
    )
    # shuffle the rows and save to out_file
    fw_filt_hq_mixed = fw_filt_hq_mixed.sample(frac=1).reset_index(drop=True)

    return fw_filt_hq_mixed


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
        list_files_df.str.split("/", expand=True)[[3]]
    ).rename(columns={3: "file_name"})

    # Create df from list of files
    # list_files_detailled = pd.DataFrame(
    #     list_files_df.str.split("/", n=7, expand=True)[[6, 7]]
    # ).rename(columns={6: "folder_date", 7: "file_name"})

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

    # Create df from list of files
    temp_df = pd.DataFrame(
        list_files_df.str.split("/", expand=True)[[3]]
    ).rename(columns={3: "file_name"})

    list_files_detailled["folder_date"] = temp_df["file_name"].apply(
        lambda x: x[0:11].replace(".", "")
    )
    list_files_detailled["folder"] = list_files_detailled["folder_date"]

    # Extract folder name
    # list_files_detailled["folder"] = list_files_df.str.split(
    #     "/", n=7, expand=True
    # )[[6]]

    # Add list of names to path
    list_files_detailled["file_path"] = list_files_df

    return list_files_detailled


def pass_checks(list_files: pd.DataFrame) -> pd.DataFrame:

    # Load dataframe containing all statistics
    stats_df = pd.read_csv("data/ISSUES/file_stats.csv")

    # Reformat data
    stats_df["starttime"] = pd.to_datetime(
        stats_df["starttime"]
    ).dt.tz_localize(None)
    stats_df["endtime"] = pd.to_datetime(stats_df["endtime"]).dt.tz_localize(
        None
    )
    stats_df["file_date"] = pd.to_datetime(
        stats_df["file_date"]
    ).dt.tz_localize(None)
    stats_df["filepath_raw"] = stats_df["filepath"]
    stats_df["filepath_filt"] = stats_df["filepath"].replace(
        {"RAW": "FILT"}, regex=True
    )

    # REMOVE FILES WITH WRONG NUMBER OF PTS
    files_npts = stats_df[
        ~stats_df.npts.isin([9000201, 8640000])
    ].filename.values

    # REMOVE FILES WITH START AND END DATE NOT THE SAME
    files_wrong_dates = stats_df[
        (stats_df.file_date > stats_df.endtime)
        | (stats_df.file_date < stats_df.starttime)
    ].filename.values

    # REMOVE FILES WITH NULL VALUES
    blue_whales_prblms = pd.read_csv("data/ISSUES/bw_0_values.csv")
    fin_whales_prblms = pd.read_csv("data/ISSUES/fw_0_values.csv")

    list_problematics_files = np.concatenate(
        (
            fin_whales_prblms[
                (fin_whales_prblms["min"] == 0)
                & (fin_whales_prblms["max"] == 0)
            ].filename.unique(),
            blue_whales_prblms[
                (blue_whales_prblms["min"] == 0)
                & (blue_whales_prblms["max"] == 0)
            ].filename.unique(),
        )
    )
    list_problematics_files = [
        a.split("/")[-1] for a in list_problematics_files
    ]

    # Only keep files
    list_files = list_files[
        (~list_files.file.isin(files_npts))
        & (~list_files.file.isin(files_wrong_dates))
        & (~list_files.file.isin(list_problematics_files))
    ]

    print("Number of files after removal:", list_files.shape[0])

    return list_files


def main() -> None:
    """ """

    # Arguments parsing
    args = parse_args()

    # Load config
    with open("config/config.yml", "r") as file:
        param_data = yaml.safe_load(file)

    # Output
    labels_output = Path("data") / "LABELS"
    bw_labels_output = labels_output / "BW"
    fw_labels_output = labels_output / "FW"
    bw_labels_output_mixed = labels_output / "BW" / "MIXED"
    fw_labels_output_mixed = labels_output / "FW" / "MIXED"

    labels_output.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist
    fw_labels_output.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist
    bw_labels_output.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist
    bw_labels_output_mixed.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist
    fw_labels_output_mixed.mkdir(
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

    for bandpass_filter in ["True", "False"]:

        print(
            "--------> Using filtered data: {} <--------".format(
                bandpass_filter
            )
        )

        # Get list of files in dataframe
        if bandpass_filter == "True":
            list_files_detailled = read_list_raw_files(
                "data/SAC_FILES_FILT.txt"
            )
        else:
            list_files_detailled = read_list_raw_files(
                "data/SAC_FILES_RAW.txt"
            )

        # Get only the name of the files
        list_files_detailled["file"] = list_files_detailled["file_path"].apply(
            lambda x: x.split("/")[-1]
        )
        print("Original # of raw data files:", list_files_detailled.shape[0])

        list_files = pass_checks(list_files_detailled)

        # Loop for the 2 whale types
        for whale_type in ["bw", "fw"]:
            # for whale_type in ["fw"]:

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
                "Number of {} detections: {} | Number of {} calls: {}".format(
                    param_data["whale_constant"][whale_type]["name"],
                    labels.detection_id.nunique(),
                    param_data["whale_constant"][whale_type]["name"],
                    labels.shape[0],
                )
            )

            # Add start and end time of calls
            labels["time_R_max"] = pd.to_datetime(labels["datetime"])

            labels["time_call_start"] = labels["time_R_max"] - timedelta(
                seconds=param_data["whale_constant"][whale_type][
                    "call_duration"
                ]
                / 2
            )
            labels["time_call_end"] = labels["time_R_max"] + timedelta(
                seconds=param_data["whale_constant"][whale_type][
                    "call_duration"
                ]
                / 2
            )

            # Reformat folder name
            labels["folder_date"] = (
                labels["date"]
                .astype(str)
                .apply(lambda x: "".join(x.split("-")))
            )

            # Add column with Whale type
            labels["whale_type"] = whale_type

            # Merge labels and SAC file PATHs to same dataframe
            final_df = pd.merge(
                labels,
                list_files,
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
            if bandpass_filter == "True":
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
                    - param_data["whale_constant"][whale_type][
                        "call_duration"
                    ],
                )
                list_randoms.append(rand_num)

            final_df["random_t"] = list_randoms
            final_df["time_window_start"] = final_df.apply(
                lambda x: x.time_call_start - timedelta(seconds=x.random_t),
                axis=1,
            ).round("10ms")

            final_df["time_window_end"] = final_df[
                "time_window_start"
            ] + timedelta(
                seconds=param_data["whale_constant"][whale_type]["window_size"]
            )

            # Convert time to pd.datetime.dt.date
            final_df["time_window_start_date"] = pd.to_datetime(
                final_df["time_window_start"]
            ).dt.date
            final_df["time_window_end_date"] = pd.to_datetime(
                final_df["time_window_end"]
            ).dt.date

            # Check to see if window date start and end on same day
            # Load dataset with stats per file
            stats_df = pd.read_csv("data/ISSUES/file_stats.csv")
            # Convert datetimes
            stats_df["starttime"] = pd.to_datetime(
                stats_df["starttime"]
            ).dt.tz_localize(None)
            stats_df["endtime"] = pd.to_datetime(
                stats_df["endtime"]
            ).dt.tz_localize(None)
            stats_df["file_date"] = pd.to_datetime(
                stats_df["file_date"]
            ).dt.tz_localize(None)
            if bandpass_filter == "True":
                stats_df["filepath"] = stats_df["filepath"].replace(
                    {"RAW": "FILT_10_32"}, regex=True
                )
            # Merge stat df and df
            merged = final_df.merge(
                stats_df, left_on="file_path", right_on="filepath"
            )
            # Convert time windows
            merged["time_window_start"] = pd.to_datetime(
                merged["time_window_start"]
            )
            merged["time_window_end"] = pd.to_datetime(
                merged["time_window_end"]
            )
            # Drop sample where
            # [time_window_end > endtime]
            # or [time_window_start < starttime]
            # or time_window_start_date != merged.time_window_end_date
            final_df.drop(
                index=merged[
                    (merged["time_window_end"] > merged["endtime"])
                    | (merged["time_window_start"] < merged["starttime"])
                    | (
                        merged.time_window_start_date
                        != merged.time_window_end_date
                    )
                ].index.tolist(),
                inplace=True,
            )

            # Group by and Save datasets
            grouped_df = (
                final_df.groupby(["station_code", "time_call_start"])
                .agg(
                    component=("component", lambda x: " ".join(x)),
                    number_components=("component", "count"),
                    time_R_max=("time_R_max", lambda x: list(x)[0]),
                    R=("R", "max"),
                    SNR=("SNR", "max"),
                    group_id=("group_id", "max"),
                    file_path=(
                        "file_path",
                        lambda x: list(x)[0][:-7]
                        + "CHANNEL"
                        + list(x)[0][-4:],
                    ),
                    time_window_start=(
                        "time_window_start",
                        lambda x: list(x)[0],
                    ),
                    time_window_end=(
                        "time_window_end",
                        lambda x: list(x)[0],
                    ),
                    time_call_end=("time_call_end", "min"),
                    whale_type=("whale_type", "min"),
                )
                .reset_index()
            )

            # Save dataframe
            final_df = final_df[
                [
                    "file_path",
                    "time_window_start",
                    "time_window_end",
                    "time_R_max",
                    "time_call_start",
                    "time_call_end",
                    "R",
                    "SNR",
                    "group_id",
                    "station_code",
                    "whale_type",
                    "component",
                ]
            ].copy()

            # HQ : R0 > 5 and SNR0 > 5
            # Grouped
            extension = csv_name_grouped[:2] + "_HQ" + csv_name_grouped[2:]
            grouped_hq = grouped_df[
                (grouped_df.R > 5) & (grouped_df.SNR > 5)
            ].copy()
            grouped_hq.to_csv(
                labels_output / whale_type.upper() / extension,
                index=False,
            )
            hq_mixed = generate_noise_samples(
                grouped_hq,
                grouped_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            hq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )
            del grouped_hq
            del hq_mixed

            # Normal
            extension = csv_name[:2] + "_HQ" + csv_name[2:]
            final_df_hq = final_df[
                (final_df.R > 5) & (final_df.SNR > 5)
            ].copy()
            final_df_hq.to_csv(
                labels_output / whale_type.upper() / extension, index=False
            )
            hq_mixed = generate_noise_samples(
                final_df_hq,
                final_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            hq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )
            del final_df_hq
            del hq_mixed

            # MQ : R0 > 3 and SNR0 > 1
            # Grouped
            extension = csv_name_grouped[:2] + "_MQ" + csv_name_grouped[2:]
            grouped_mq = grouped_df[
                (grouped_df.R > 3) & (grouped_df.SNR > 1)
            ].copy()
            grouped_mq.to_csv(
                labels_output / whale_type.upper() / extension,
                index=False,
            )
            mq_mixed = generate_noise_samples(
                grouped_mq,
                grouped_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            mq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )
            del grouped_mq
            del mq_mixed

            # Normal
            extension = csv_name[:2] + "_MQ" + csv_name[2:]
            final_df_mq = final_df[
                (final_df.R > 3) & (final_df.SNR > 1)
            ].copy()
            final_df_mq.to_csv(
                labels_output / whale_type.upper() / extension, index=False
            )
            mq_mixed = generate_noise_samples(
                final_df_mq,
                final_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            mq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )

            # LQ : All data
            # Grouped
            extension = csv_name_grouped[:2] + "_LQ" + csv_name_grouped[2:]
            grouped_lq = grouped_df.copy()
            grouped_lq.to_csv(
                labels_output / whale_type.upper() / extension,
                index=False,
            )
            lq_mixed = generate_noise_samples(
                grouped_lq,
                grouped_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            lq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )
            # Normal
            extension = csv_name[:2] + "_LQ" + csv_name[2:]
            final_df_lq = final_df.copy()
            final_df_lq.to_csv(
                labels_output / whale_type.upper() / extension, index=False
            )
            lq_mixed = generate_noise_samples(
                final_df_lq,
                final_df,
                param_data["whale_constant"][whale_type]["noise_shift"],
                param_data["whale_constant"][whale_type]["window_size"],
            )
            lq_mixed.to_csv(
                labels_output / whale_type.upper() / "MIXED" / extension,
                index=False,
            )


def parse_args() -> Namespace:
    """Parse arguments"""
    description = "Script for preprocessing the labels coming from .mat matrix"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--input_file",
        default="data/WhaleDetections_matlab_output.mat",
        type=str,
        help="path to the input file",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
