import pandas as pd
import numpy as np
import obspy
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# TODO Add Data Parser for 3-components seismic waves


class WhaleDataset(Dataset):
    """Dataset class for iterating over the data."""

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """__getitem__.
        Args:
            index (int): Get index item from the dataset.
        """
        sac_path = self.labels["file_path"][index]
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
        target_example = np.zeros(input_example.shape)
        target_example[index_call_start : index_call_end + 1] = 1  # noqa: E203

        input_example = np.expand_dims(
            input_example, axis=0
        )  # add channel dimension

        return input_example, target_example


class WhaleDataModule(pl.LightningDataModule):
    """Data module class that prepares dataset parsers
    and instantiates data loaders."""

    def __init__(
        self: pl.LightningDataModule,
        data_dir: str,
        batch_size: int = 16,
        sampling_rate: int = 100,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fs = sampling_rate
        self.train_ds, self.valid_ds, self.test_ds = None, None, None

    def prepare_data(self: pl.LightningDataModule) -> None:
        """Downloads/extracts/unpacks the data if needed."""
        pass

    def setup(self: pl.LightningDataModule, stage: str = None) -> None:
        """Parses and splits all samples across
        the train/valid/test parsers."""
        # here, we will actually assign train/val datasets
        # for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = WhaleDataset(
                labels_file=self.data_dir + "/train.csv", fs=self.fs
            )
            self.valid_ds = WhaleDataset(
                labels_file=self.data_dir + "/valid.csv", fs=self.fs
            )
        if stage == "test" or stage is None:
            self.test_ds = WhaleDataset(
                labels_file=self.data_dir + "/test.csv", fs=self.fs
            )

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
