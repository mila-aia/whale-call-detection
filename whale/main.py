#!/usr/bin/env python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def main() -> None:
    """Main entry point of the program."""
    parse_args()


def parse_args() -> Namespace:
    description = "Whalle Call Detection"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--input-file",
        default="data/raw/enron/emails.csv",
        type=str,
        help="path to the input CSV file",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
