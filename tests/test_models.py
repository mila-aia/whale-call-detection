import torch
from whale.models import UNet


def test_unet() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=2)
    model = model.to(device)

    sig = torch.zeros([1, 1, 501]).to(device)
    pred = model.forward(sig)
    assert pred.shape == (1, 2, 501)
