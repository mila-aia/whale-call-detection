import torch
import torchmetrics


class SegmentationMetrics:
    """Class that implements segmentation metrics."""

    def __init__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 2,
    ) -> None:
        """Initialize the class.
        Parameters
        ----------
        targets : torch.Tensor
            True labels.
        predictions : torch.Tensor
            Labels (probability over labels) predicted from the model.
        num_classes: int
            Number of classes, default is 2.
        """
        if type(targets) != torch.Tensor:
            self.targets = torch.tensor(targets)
        else:
            self.targets = targets

        if type(predictions) != torch.Tensor:
            self.predictions = torch.tensor(predictions)
        else:
            self.predictions = predictions

        if self.predictions.device != self.targets.device:
            self.targets = self.targets.to(self.predictions.device)

        self.device = self.predictions.device
        self.num_classes = num_classes

    def compute_acc(self) -> torch.Tensor:
        """Computes the accuracy.
        Returns
        -------
        torch.Tensor
            Accuracy score.
        """

        acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        ).to(self.device)

        return acc(self.predictions, self.targets)

    def compute_prec(self) -> torch.Tensor:
        """Computes the Precision.
        Returns
        -------
        torch.Tensor
            Precision score.
        """

        prec = torchmetrics.Precision(
            task="multiclass",
            threshold=0.5,
            average="micro",
            num_classes=self.num_classes,
        ).to(self.device)

        return prec(self.predictions, self.targets)

    def compute_recall(self) -> torch.Tensor:
        """Computes the Recall.
        Returns
        -------
        torch.Tensor
            Recall score.
        """

        recall = torchmetrics.Recall(
            task="multiclass",
            threshold=0.5,
            average="micro",
            num_classes=self.num_classes,
        ).to(self.device)

        return recall(self.predictions, self.targets)

    def compute_IoU(self) -> torch.Tensor:
        """Computes the IoU.
        Returns
        -------
        torch.Tensor
            IoU score.
        """

        iou = torchmetrics.JaccardIndex(
            task="multiclass",
            threshold=0.5,
            average="micro",
            num_classes=self.num_classes,
        ).to(self.device)
        return iou(self.predictions, self.targets)

    def get_metrics_dict(self) -> dict:
        """Computes all metrics and puts in dict.
        Returns
        -------
        dict
            Dictionary of metrics names and values
        """
        metrics_dict = dict(
            {
                "acc": self.compute_acc(),
                "prec": seg_metrics.compute_prec(),
                "recall": seg_metrics.compute_recall(),
                "IoU": seg_metrics.compute_IoU(),
            }
        )
        return metrics_dict


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    sample_lenth = 501
    pred = torch.rand([2, num_classes, sample_lenth]).to(device)
    target = torch.zeros([2, sample_lenth], dtype=torch.int32).to(device)

    seg_metrics = SegmentationMetrics(pred, target, num_classes=num_classes)
    acc = seg_metrics.compute_acc()
    prec = seg_metrics.compute_prec()
    recall = seg_metrics.compute_recall()
    iou = seg_metrics.compute_IoU()

    print(seg_metrics.get_metrics_dict())
