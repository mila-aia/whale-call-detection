import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any
from whale.utils.metrics import SegmentationMetrics


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self: nn.Module,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self: nn.Module, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self: nn.Module,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce
        # the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self: nn.Module, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:

        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self: nn.Module, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(pl.LightningModule):
    def __init__(
        self: nn.Module,
        n_channels: int,
        n_classes: int,
        bilinear: bool = False,
    ) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self: pl.LightningModule, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Unnormalized logits for each class.
        return logits

    def compute_metrics(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
    ) -> int:
        """Computes the metrics.
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted Mask.
        targets : torch.Tensor
            Ground Truth Mask.
        Returns
        -------
        torch.Tensor
            Segmentation mask.
        """
        seg_metrics = SegmentationMetrics(predictions, targets, self.n_classes)
        acc = seg_metrics.compute_acc()

        return acc

    def training_step(
        self: pl.LightningModule, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> dict:
        """Runs a prediction step for training, returning the loss.
        Parameters
        ----------
        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        sig = batch["sig"]
        target = batch["target"]
        logits = self.forward(sig)
        loss = self.loss_fn(logits, target)
        self.log("train_loss", loss)
        preds = torch.softmax(logits, dim=1)
        acc = self.compute_metrics(preds, target)

        return {"loss": loss, "acc": acc}

    def training_epoch_end(
        self: pl.LightningModule, training_step_outputs: list[dict]
    ) -> None:
        """Is called at the end of each epoch.
        Parameters
        ----------
        training_step_outputs : typing.List[float]
            A list of training metrics\
                    produced by the training step.
        """
        for training_step_output in training_step_outputs:
            train_loss_mean = training_step_output["loss"].mean()

        self.log("overall_train_loss", train_loss_mean)

    def validation_step(
        self: pl.LightningModule, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> dict:
        """Runs a prediction step for validation, returning the loss.
        Parameters
        ----------
        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        sig = batch["sig"]
        target = batch["target"]
        logits = self.forward(sig)
        loss = self.loss_fn(logits, target)
        preds = torch.softmax(logits, dim=1)
        acc = self.compute_metrics(preds, target)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(
        self: pl.LightningModule, validation_step_outputs: list[dict]
    ) -> None:
        """Is called at the end of each epoch.
        Parameters
        ----------
        validation_step_outputs : typing.List[float]
            A list of validation metrics\
                    produced by the training step.
        """
        for validation_step_output in validation_step_outputs:
            val_loss_mean = validation_step_output["val_loss"].mean()

        self.log("overall_val_loss", val_loss_mean)

    def test_step(
        self: pl.LightningModule, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> dict:
        """Runs a prediction step for testing, returning the loss.
        Parameters
        ----------
        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        sig = batch["sig"]
        target = batch["target"]
        logits = self.forward(sig)
        loss = self.loss_fn(logits, target)
        preds = torch.softmax(logits, dim=1)
        acc = self.compute_metrics(preds, target)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self: pl.LightningModule) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = UNet(n_channels=1, n_classes=2)
    model = model.to(device)

    sig = torch.zeros([1, 1, 501]).to(device)
    target = torch.empty(1, 501, dtype=torch.long).random_(2).to(device)

    print("input shape: ", sig.shape)
    print("target shape: ", target.shape)

    logits = model.forward(sig)
    print("output shape: ", logits.shape)
    print(model.loss_fn(logits, target))

    preds = torch.softmax(logits, dim=1)
    acc = model.compute_metrics(preds, target)
    print(acc)
