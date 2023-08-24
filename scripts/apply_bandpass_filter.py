from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from pathlib import Path
from obspy.core import read
from tqdm import tqdm
from os import path


def main() -> None:

    args = parse_args()

    sac_files = args.input_sac_files
    if not path.exists(sac_files):
        raise (sac_files + " does not exist!")

    sac_lists = []
    with open(sac_files) as f:
        for line in f:
            line = line.strip()
            sac_lists.append(line)

    freq_min = args.freq_min
    freq_max = args.freq_max

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sac_file in tqdm(sac_lists):
        st = read(sac_file)
        st.filter(
            "bandpass", freqmin=freq_min, freqmax=freq_max, zerophase=True
        )
        datestring = sac_file.split("/")[-2]
        output_name = sac_file.split("/")[-1]
        date_folder = output_dir / datestring
        date_folder.mkdir(parents=True, exist_ok=True)
        st.write(str(date_folder / output_name), format="SAC")


def parse_args() -> Namespace:
    description = "Script to apply bandpass filter to a list of SAC files"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--input-sac-files",
        default="data/SAC_FILES_RAW.txt",
        type=str,
        help="path to a .txt file including a list of SAC files"
        + " to be filtered.",
    )

    arg_parser.add_argument(
        "--output-dir",
        default="data/FILT_10_32/",
        type=str,
        help="path to the save filtered SAC file",
    )

    arg_parser.add_argument(
        "--freq-min",
        default=10,
        type=int,
        help="low frequency of the bandpass filter",
    )

    arg_parser.add_argument(
        "--freq-max",
        default=32,
        type=int,
        help="high frequency of the bandpass filter",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
