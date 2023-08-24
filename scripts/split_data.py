import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def main() -> None:

    args = parse_args()

    input_dataset = pd.read_csv(args.input_file)
    output_dir = Path(args.output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # split inp_df into train, valid, test
    train = input_dataset.sample(frac=0.8, random_state=200)
    test = input_dataset.drop(train.index)
    valid = test.sample(frac=0.5, random_state=200)
    test = test.drop(valid.index)
    # save train, valid, test
    train.to_csv(output_dir / "train.csv", index=False)
    valid.to_csv(output_dir / "valid.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def parse_args() -> Namespace:
    description = "Script to apply bandpass filter to a list of SAC files"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--input-file",
        default="data/LABELS/FW/MIXED/fw_HQ_component_grouped_filt.csv",
        type=str,
        help="Path to dataset (.csv)",
    )

    arg_parser.add_argument(
        "--output-path",
        default="data/datasets/FWC_HQ_3CH_FILT/",
        type=str,
        help="Path to output folder.",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
