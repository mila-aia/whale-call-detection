import whale
from pytorch_lightning import Trainer
from pathlib import Path
import torch
import pandas as pd
import obspy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from whale.utils import cal_spectrogram
from typing import List, Union


class TestDataset(Dataset):
    """Dataset class for iterating over the test data."""

    # labels: pd.DataFrame
    # fs: float
    # normalize: bool
    def __init__(
        self: Dataset,
        labels_file: Union[str, Path],
        stage: str = "test",
        fs: int = 100,
        normalize: bool = True,
    ) -> None:
        """Initialize Whale Call Dataset.
        Args:
            labels_file (str): Path to the labels.
            stage (str): train, valid, or test
            target_time_max_val (float): maximum value of target time
            target_time_min_val (float): minimum value of target time
            fs (int): Sampling rate of the seismic signal.
            normalize (bool): Normalize the waveform into [0,1].
        """
        super().__init__()

        self.labels = pd.read_csv(labels_file)
        self.stage = stage
        self.fs = fs
        self.normalize = normalize

    def __len__(self: Dataset) -> int:
        """Return the number of data items in MyDataset."""
        return len(self.labels)

    def __getitem__(
        self: Dataset,
        index: int,
    ) -> dict:
        """__getitem__.
        Args:
            index (int): Get index item from the dataset.
        """
        sac_path = self.labels["file_path"][index]
        meta_data = dict(self.labels.iloc[index])
        component = self.labels["component"][index].split(" ")
        start_time = obspy.UTCDateTime(self.labels["time_window_start"][index])
        end_time = obspy.UTCDateTime(self.labels["time_window_end"][index])

        list_spec = []

        for comp in component:
            tr = obspy.read(
                sac_path.replace("CHANNEL", comp),
                starttime=start_time,
                endtime=end_time,
            )
            input_waveform = tr[0].data

            # Calculate spectrogram with a shape of (n_freq, n_time).
            input_spec, _, _ = cal_spectrogram(
                input_waveform,
                samp_rate=self.fs,
                per_lap=0.9,
                wlen=0.5,
                mult=4,
            )
            list_spec.append(input_spec)
        input_spec = np.average(list_spec, axis=0)

        if self.normalize:
            min_val = input_spec.min()
            max_val = input_spec.max()
            input_spec = (input_spec - min_val) / (max_val - min_val)

            min_val = input_waveform.min()
            max_val = input_waveform.max()
            input_waveform = (input_waveform - min_val) / (max_val - min_val)

        input_waveform = np.expand_dims(input_waveform, axis=0)
        input_waveform = torch.from_numpy(input_waveform).float()

        input_spec = input_spec.T  # transpose to (n_time, n_freq)
        input_spec = torch.from_numpy(input_spec).float()
        return {
            "data_index": index,
            "sig": input_waveform,
            "spec": input_spec,
            "meta_data": meta_data,
        }


def main() -> None:
    args = parse_args()

    cpkt_path = Path(args.model_ckpt).expanduser().resolve()
    inp_csv_path = Path(args.inp_csv).expanduser().resolve()
    out_csv_path = Path(args.out_csv).expanduser().resolve()
    batch_size = args.batch_size
    test_dataset = TestDataset(labels_file=inp_csv_path)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    model = whale.models.LSTM.load_from_checkpoint(cpkt_path)
    trainer = Trainer()
    predictions = trainer.predict(model, test_dataloader)
    # write predictions to csv
    labels_pred: List[int] = []
    times_pred: List[float] = []
    # predictions is a list of dictionaries
    for i in range(len(predictions)):
        labels_pred = labels_pred + [
            val for val in predictions[i]["label"].numpy()
        ]
        times_pred = times_pred + [
            val for val in predictions[i]["time"].squeeze(1).numpy()
        ]

    inp_df = pd.read_csv(inp_csv_path)
    inp_df["label_pred"] = pd.Series(labels_pred)
    inp_df["time_pred"] = pd.Series(times_pred)
    inp_df.to_csv(out_csv_path, index=False)


def parse_args() -> Namespace:
    description = "Make prediction using a pretrained model"
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--model-ckpt",
        default="model.ckpt",
        type=str,
        help="path to the pretrained model checkpoint",
    )
    arg_parser.add_argument(
        "--inp-csv",
        default="samples.csv",
        type=str,
        help="path to the input csv file",
    )
    arg_parser.add_argument(
        "--out-csv",
        default="predictions.csv",
        type=str,
        help="path to the predictions csv file",
    )
    arg_parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="batch size for prediction",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
