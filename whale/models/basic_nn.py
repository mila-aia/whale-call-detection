import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from typing import Any


# Add an pytorch LSTM model
class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 129,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        num_classes: int = 2,
        bidirectional: bool = False,
    ) -> None:

        super(LSTM, self).__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden2class = nn.Linear(hidden_dim, num_classes)
        self.hidden2time = nn.Linear(hidden_dim, 1)

        self.activation_fn = nn.ReLU()

        # CrossEntropyLoss for classification and MSELoss for regression
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.MSELoss()

        return None

    def forward(self: pl.LightningModule, x: torch.Tensor) -> torch.Tensor:

        # Forward propagate LSTM
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])

        # Decode the hidden state of the last time step

        # Unnormalized logits for each class.
        class_logits = self.hidden2class(out)  # (batch_size, n_classes)
        # regression output
        reg_out = self.hidden2time(out)  # (batch_size, 1)

        return class_logits, reg_out

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
        acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        ).to(self.device)

        return acc(predictions, targets)

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
        spec = batch["spec"]
        r_time = batch["target_time"]
        label = batch["target_label"]
        class_logits, reg_out = self.forward(spec)

        loss_cls = self.class_loss_fn(class_logits, label)
        loss_reg = self.reg_loss_fn(reg_out, r_time)
        loss = loss_cls + loss_reg

        self.log("train_loss", loss)
        self.log("train_loss_cls", loss_cls)
        self.log("train_loss_reg", loss_reg)

        preds = torch.softmax(class_logits, dim=1)

        acc = self.compute_metrics(preds, label)

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
            train_acc_mean = training_step_output["acc"].mean()

        self.log("overall_train_loss", train_loss_mean)
        self.log("overall_train_acc", train_acc_mean)

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
            val_acc_mean = validation_step_output["val_acc"].mean()

        self.log("overall_val_loss", val_loss_mean)
        self.log("overall_val_acc", val_acc_mean)

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

    model = LSTM(input_dim=50, hidden_dim=32)

    model = model.to(device)

    sig = torch.zeros([1, 2, 50]).to(device)
    # target = torch.empty(1, 501, dtype=torch.long).random_(2).to(device)

    print("input shape: ", sig.shape)
    # print("target shape: ", target.shape)

    logits, out = model.forward(sig)
    print("logits output shape: ", logits.shape)
    print("reg output shape: ", out.shape)
