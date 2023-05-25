import pandas as pd
import numpy as np
import obspy
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from whale.utils.spectrogram import cal_spectrogram

# TODO Add Data Parser for 3-components seismic waves


class WhaleDataset(Dataset):
    """Dataset class for iterating over the data."""

    # labels: pd.DataFrame
    # fs: float
    # normalize: bool
    def __init__(
        self: Dataset, labels_file: str, fs: int = 100, normalize: bool = True
    ) -> None:
        """Initialize Whale Call Dataset.
        Args:
            labels_file (str): Path to the labels.
            fs (int): Sampling rate of the seismic signal.
            normalize (bool): Normalize the waveform into [0,1].
        """
        super().__init__()

        self.labels = pd.read_csv(labels_file)
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
        start_time = obspy.UTCDateTime(self.labels["time_window_start"][index])
        end_time = obspy.UTCDateTime(self.labels["time_window_end"][index])
        tr = obspy.read(sac_path, starttime=start_time, endtime=end_time)
        input_example = tr[0].data

        if self.normalize:
            min_val = input_example.min()
            max_val = input_example.max()
            input_example = (input_example - min_val) / (max_val - min_val)

        time_call_start = obspy.UTCDateTime(
            self.labels["time_call_start"][index]
        )
        time_call_end = obspy.UTCDateTime(self.labels["time_call_end"][index])
        index_call_start = int((time_call_start - start_time) * self.fs)
        index_call_end = int((time_call_end - start_time) * self.fs)
        target_example = np.zeros(input_example.shape, dtype=np.int64)
        target_example[index_call_start : index_call_end + 1] = 1  # noqa: E203

        input_example = np.expand_dims(
            input_example, axis=0
        )  # add channel dimension

        input_example = torch.from_numpy(input_example)
        target_example = torch.from_numpy(target_example)

        return {
            "data_index": index,
            "sig": input_example,
            "target": target_example,
            "meta_data": meta_data,
        }


class WhaleDatasetSpec(Dataset):
    """Dataset class for iterating over the data."""

    # labels: pd.DataFrame
    # fs: float
    # normalize: bool
    def __init__(
        self: Dataset, labels_file: str, fs: int = 100, normalize: bool = True
    ) -> None:
        """Initialize Whale Call Dataset.
        Args:
            labels_file (str): Path to the labels.
            fs (int): Sampling rate of the seismic signal.
            normalize (bool): Normalize the waveform into [0,1].
        """
        super().__init__()

        self.labels = pd.read_csv(labels_file)
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
        time_R_max = obspy.UTCDateTime(self.labels["time_R_max"][index])
        start_time = obspy.UTCDateTime(self.labels["time_window_start"][index])
        end_time = obspy.UTCDateTime(self.labels["time_window_end"][index])
        call_type = self.labels["whale_type"][index]

        tr = obspy.read(sac_path, starttime=start_time, endtime=end_time)
        input_waveform = tr[0].data

        # Calculate spectrogram with a shape of (n_freq, n_time).
        input_spec, _, _ = cal_spectrogram(
            input_waveform, samp_rate=self.fs, per_lap=0.9, wlen=0.5, mult=4
        )
        if self.normalize:
            min_val = input_spec.min()
            max_val = input_spec.max()
            input_spec = (input_spec - min_val) / (max_val - min_val)

            min_val = input_waveform.min()
            max_val = input_waveform.max()
            input_waveform = (input_waveform - min_val) / (max_val - min_val)

        target_label = 0 if call_type == "noise" else 1

        target_time_R_max = time_R_max - start_time
        target_time_R_max = np.float32(target_time_R_max)

        input_waveform = np.expand_dims(input_waveform, axis=0)
        input_spec = input_spec.T  # transpose to (n_time, n_freq)

        input_waveform = torch.from_numpy(input_waveform).float()
        input_spec = torch.from_numpy(input_spec).float()

        return {
            "data_index": index,
            "sig": input_waveform,
            "spec": input_spec,
            "target_time": target_time_R_max,
            "target_label": target_label,
            "meta_data": meta_data,
        }


class WhaleDataModule(pl.LightningDataModule):
    """Data module class that prepares dataset parsers
    and instantiates data loaders.
    """

    def __init__(
        self: pl.LightningDataModule,
        data_dir: str,
        batch_size: int = 16,
        sampling_rate: int = 100,
        data_type: str = "waveform",  # "spec" or "waveform"
    ) -> None:
        """Initialize WhaleDataModule.
        Args:
            data_dir (str): Path to the data directory containing
                            train/valid/test splits.
            batch_size (int): Batch size. Defaults to 16.
            sampling_rate (int): Sampling rate of the seismic signal.
                                Defaults to 100.
            data_type (str): "spec" or "waveform". Defaults to "waveform"
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fs = sampling_rate
        self.train_ds, self.valid_ds, self.test_ds = None, None, None
        self.data_type = data_type

    def prepare_data(self: pl.LightningDataModule) -> None:
        """Downloads/extracts/unpacks the data if needed."""
        pass

    def setup(self: pl.LightningDataModule, stage: str = None) -> None:
        """Parses and splits all samples across
        the train/valid/test parsers."""
        # here, we will actually assign train/val datasets
        # for use in dataloaders
        if stage == "fit" or stage is None:
            if self.data_type == "spec":
                self.train_ds = WhaleDatasetSpec(
                    labels_file=self.data_dir + "/train.csv", fs=self.fs
                )
                self.valid_ds = WhaleDatasetSpec(
                    labels_file=self.data_dir + "/valid.csv", fs=self.fs
                )
            elif self.data_type == "waveform":
                self.train_ds = WhaleDataset(
                    labels_file=self.data_dir + "/train.csv", fs=self.fs
                )
                self.valid_ds = WhaleDataset(
                    labels_file=self.data_dir + "/valid.csv", fs=self.fs
                )
            else:
                raise ValueError("data_type must be 'spec' or 'waveform'")

        if stage == "test" or stage is None:
            if self.data_type == "spec":
                self.test_ds = WhaleDatasetSpec(
                    labels_file=self.data_dir + "/test.csv", fs=self.fs
                )
            elif self.data_type == "waveform":
                self.test_ds = WhaleDataset(
                    labels_file=self.data_dir + "/test.csv", fs=self.fs
                )
            else:
                raise ValueError("data_type must be 'spec' or 'waveform'")

    def train_dataloader(self: pl.LightningDataModule) -> DataLoader:
        """Creates the training dataloader using the
        training data parser."""
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self: pl.LightningDataModule) -> DataLoader:
        """Creates the validation dataloader using
        the validation data parser."""
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self: pl.LightningDataModule) -> DataLoader:
        """Creates the testing dataloader using the testing data parser."""
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )
