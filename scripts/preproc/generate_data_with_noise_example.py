import pandas as pd
from obspy import UTCDateTime
from argparse import (
    ArgumentParser,
    ArgumentDefaultsHelpFormatter,
    Namespace,
)
from pathlib import Path


def timeShift(input_time: str, shift: int) -> str:
    # shift the input_time by shift seconds
    shifted_time = UTCDateTime(input_time) + shift
    return UTCDateTime.strftime(shifted_time, "%Y-%m-%d %H:%M:%S.%f")


def timeDiff(input_time: str, ref_time: str) -> float:
    # calculate the time difference between input_time and ref_time
    delta_t = UTCDateTime(input_time) - UTCDateTime(ref_time)
    return delta_t


def main() -> None:

    args = parse_args()
    input_file = Path(args.input_file)
    out_file = Path(args.out_file)
    noise_duration = args.noise_duration
    noise_shift = args.noise_shift
    fw_filt_hq = pd.read_csv(input_file)

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
    ]
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
    ]
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
    # concat fw_filt_hq_fw and fw_filt_hq_noise
    fw_filt_hq_mixed = pd.concat(
        [fw_filt_hq_fw, fw_filt_hq_noise], ignore_index=True
    )
    # shuffle the rows and save to out_file
    fw_filt_hq_mixed = fw_filt_hq_mixed.sample(frac=1).reset_index(drop=True)
    fw_filt_hq_mixed.to_csv(out_file, index=False)


def parse_args() -> Namespace:
    """Parse arguments"""
    description = "Script to generate data mixed with noise examples"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--input_file",
        default="data.csv",
        type=str,
        help="path to the input file",
    )
    arg_parser.add_argument(
        "--out_file",
        default="data_mixed.csv",
        type=str,
        help="path to the output file",
    )
    arg_parser.add_argument(
        "--noise_duration",
        default=2,
        type=int,
        help="duration of noise window in seconds",
    )
    arg_parser.add_argument(
        "--noise_shift",
        default=5,
        type=int,
        help="time shift of data window to get noise window in seconds",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
