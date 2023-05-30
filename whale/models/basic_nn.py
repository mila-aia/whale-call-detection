import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from typing import Any, Dict, Tuple


class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 129,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        num_classes: int = 2,
        bidirectional: bool = False,
        reg_loss_weight: float = 0.5,
        lr: float = 1e-3,
    ) -> None:
        """
        LSTM model for whale classification and time prediction.
        Parameters
        ----------
        input_dim : int, optional
            Dimension of the input sequence to the LSTM, by default 129
        hidden_dim : int, optional
            Dimension of the hidden state of the LSTM, by default 128
        num_layers : int, optional
            Number of layers in the LSTM, by default 1
        dropout : float, optional
            Dropout rate, by default 0.5
        num_classes : int, optional
            Number of classes, by default 2
        bidirectional : bool, optional
            Whether the LSTM is bidirectional, by default False
        reg_loss_weight : float, optional
            Weight for the regression loss, by default 0.5
        lr : float, optional
            Learning rate, by default 1e-3
        """

        super(LSTM, self).__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.reg_loss_weight = reg_loss_weight
        self.lr = lr

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
        """
        Forward pass of the model.
        """

        # Forward propagate LSTM
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])

        # Decode the hidden state of the last time step

        # Unnormalized logits for each class.
        class_logits = self.hidden2class(out)  # (batch_size, n_classes)
        # regression output
        reg_out = self.hidden2time(out)  # (batch_size, 1)

        return class_logits, reg_out

    def compute_loss(
        self,
        logits_pred: torch.tensor,
        time_pred: torch.tensor,
        label: torch.tensor,
        r_time: torch.tensor,
        w: float = 0.5,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Computes the loss.
        Parameters
        ----------
        logits_pred : torch.Tensor
            Predicted class unormalized logits.
        time_pred : torch.Tensor
            Predicted time.
        label : torch.Tensor
            Ground Truth Labels.
        r_time : torch.Tensor
            Ground Truth Time.
        w : float, optional
            Weight for the regression loss, by default 0.5.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of total loss, classification loss and regression loss.

        """
        loss_cls = self.class_loss_fn(logits_pred, label)

        # write weighted MSE loss, where the weight is the true class label
        # TODO: this only works for binary classification
        def weighted_mse_loss(
            input: torch.Tensor, target: torch.Tensor, weight: float
        ) -> torch.Tensor:
            return (weight * (input - target) ** 2).mean()

        loss_reg = weighted_mse_loss(time_pred, r_time, label)
        # loss_reg = self.reg_loss_fn(time_pred, r_time)

        loss = loss_cls + w * loss_reg

        return loss, loss_cls, loss_reg

    def compute_metrics(
        self,
        predictions: torch.tensor,
        targets: torch.tensor,
    ) -> Dict[str, Any]:
        """Computes the metrics.
        Parameters
        ----------
        predictions : torch.Tensor
            Predicted Labels.
        targets : torch.Tensor
            Ground Truth Labels.
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics.
        """
        acc_scorer = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        ).to(self.device)
        accuracy = acc_scorer(predictions, targets)

        f1_micro_scorer = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            average="micro",
            num_classes=self.num_classes,
        ).to(self.device)
        f1_score_micro = f1_micro_scorer(predictions, targets)

        f1_macro_scorer = torchmetrics.F1Score(
            task="multiclass",
            threshold=0.5,
            average="macro",
            num_classes=self.num_classes,
        ).to(self.device)
        f1_score_macro = f1_macro_scorer(predictions, targets)

        metrics_dict = dict(
            {
                "acc": accuracy,
                "f1_micro": f1_score_micro,
                "f1_macro": f1_score_macro,
            }
        )
        return metrics_dict

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
        # loss_cls = self.class_loss_fn(class_logits, label)
        # loss_reg = self.reg_loss_fn(reg_out.squeeze(1), r_time)
        # loss = loss_cls + loss_reg
        loss, loss_cls, loss_reg = self.compute_loss(
            class_logits, reg_out.squeeze(1), label, r_time
        )

        output_dict = {}
        output_dict["loss"] = loss
        self.log("train_loss", loss)
        self.log("train_loss_cls", loss_cls)
        self.log("train_loss_reg", loss_reg)

        preds = torch.argmax(class_logits, dim=1)
        train_metrics = self.compute_metrics(preds, label)

        for key, value in train_metrics.items():
            self.log(f"train_{key}", value)
            output_dict[f"train_{key}"] = value

        return output_dict

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
            train_f1_macro_mean = training_step_output["train_f1_macro"].mean()
            train_f1_micro_mean = training_step_output["train_f1_micro"].mean()
            train_acc_mean = training_step_output["train_acc"].mean()

        self.log("overall_train_loss", train_loss_mean)
        self.log("overall_train_f1_micro", train_f1_micro_mean)
        self.log("overall_train_f1_macro", train_f1_macro_mean)
        self.log("overall_train_acc", train_acc_mean)

    def validation_step(
        self: pl.LightningModule, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> dict:
        """Runs a prediction step for validation, returning the loss and metrics.
        Parameters
        ----------
        Returns
        -------
        Dict[str, torch.Tensor]
            loss and metrics for the validation step.
        """
        spec = batch["spec"]
        r_time = batch["target_time"]
        label = batch["target_label"]
        class_logits, reg_out = self.forward(spec)

        # loss_cls = self.class_loss_fn(class_logits, label)
        # loss_reg = self.reg_loss_fn(reg_out.squeeze(1), r_time)
        # loss = loss_cls + loss_reg
        loss, loss_cls, loss_reg = self.compute_loss(
            class_logits, reg_out.squeeze(1), label, r_time
        )

        output_dict = {}
        output_dict["val_loss"] = loss
        self.log("val_loss", loss)
        self.log("val_loss_cls", loss_cls)
        self.log("val_loss_reg", loss_reg)

        preds = torch.argmax(class_logits, dim=1)
        val_metrics = self.compute_metrics(preds, label)

        for key, value in val_metrics.items():
            self.log(f"val_{key}", value)
            output_dict[f"val_{key}"] = value

        return output_dict

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
            val_f1_micro_mean = validation_step_output["val_f1_micro"].mean()
            val_f1_macro_mean = validation_step_output["val_f1_macro"].mean()

        self.log("overall_val_loss", val_loss_mean)
        self.log("overall_val_acc", val_acc_mean)
        self.log("overall_val_f1_micro", val_f1_micro_mean)
        self.log("overall_val_f1_macro", val_f1_macro_mean)

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
        spec = batch["spec"]
        r_time = batch["target_time"]
        label = batch["target_label"]
        class_logits, reg_out = self.forward(spec)

        # loss_cls = self.class_loss_fn(class_logits, label)
        # loss_reg = self.reg_loss_fn(reg_out.squeeze(1), r_time)
        # loss = loss_cls + loss_reg
        loss, loss_cls, loss_reg = self.compute_loss(
            class_logits, reg_out.squeeze(1), label, r_time
        )

        output_dict = {}
        output_dict["loss"] = loss
        self.log("test_loss", loss)
        self.log("test_loss_cls", loss_cls)
        self.log("test_loss_reg", loss_reg)

        preds = torch.argmax(class_logits, dim=1)
        train_metrics = self.compute_metrics(preds, label)

        for key, value in train_metrics.items():
            self.log(f"test_{key}", value)
            output_dict[f"test_{key}"] = value

        return output_dict

    def predict_step(
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
        spec = batch["spec"]
        class_logits, reg_out = self.forward(spec)
        preds = torch.argmax(class_logits, dim=1)
        output_dict = {"label": preds, "time": reg_out}

        return output_dict

    def configure_optimizers(self: pl.LightningModule) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


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
