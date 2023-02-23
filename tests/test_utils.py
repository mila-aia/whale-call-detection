import torch
from whale.utils.metrics import SegmentationMetrics


def test_SegmentationMetrics() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    sample_lenth = 10
    pred = torch.rand([2, num_classes, sample_lenth]).to(device)
    target = torch.ones([2, sample_lenth], dtype=torch.int32).to(device)

    seg_metrics = SegmentationMetrics(pred, target, num_classes=num_classes)
    acc = seg_metrics.compute_acc()

    assert acc < 1.0
