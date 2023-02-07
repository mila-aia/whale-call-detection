from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from pathlib import Path
import pandas as pd
import scipy.io


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


def main() -> None:
    """ """

    # Arguments parsing
    args = parse_args()
    # Output
    output_dir = Path(args.output_dir).expanduser().resolve()
    labels_output = output_dir / "LABELS"
    labels_output.mkdir(
        parents=True, exist_ok=True
    )  # Create folder if not exist
    # Input
    input_file = Path(args.input_file)

    # Load .mat file
    mat = scipy.io.loadmat(input_file)

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

    fw_calls = pd.DataFrame(mat["FWC"], columns=colnames_WC)
    bw_calls = pd.DataFrame(mat["BWC"], columns=colnames_WC)

    # Load name and index of stations
    stadir_fw = read_stadir(mat, "stadir")
    stadir_bw = read_stadir(mat, "stadir_bw")

    # Preprocess
    fw_ds = preprocess_(fw_calls, stadir_fw)
    bw_ds = preprocess_(bw_calls, stadir_bw)

    # Plot counts
    print(
        "Number of Fin Whale calls detected: {}".format(
            fw_ds.detection_id.nunique()
        )
    )
    print(
        "Number of Blue Whale calls detected: {}".format(
            bw_ds.detection_id.nunique()
        )
    )

    # Save datasets
    fw_ds.to_csv(labels_output / "fin_whales.csv")
    bw_ds.to_csv(labels_output / "blue_whales.csv")


def parse_args() -> Namespace:
    description = "Script for preprocessing the labels coming from .mat matrix"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--output_dir",
        default="data/",
        type=str,
        help="path to the output directory",
    )

    arg_parser.add_argument(
        "--input_file",
        default="/network/projects/aia/whale_call/calls_data \
        /WhaleDetectionsLSZ.mat",
        type=str,
        help="path to the input file",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
