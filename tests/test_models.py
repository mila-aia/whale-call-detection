import torch
from whale.models import UNet


def test_unet() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=2)
    model = model.to(device)

    sig = torch.zeros([1, 1, 501]).to(device)
    target = torch.empty(1, 501, dtype=torch.long).random_(2).to(device)
    pred = model.forward(sig)
    loss = model.loss_fn(pred, target)

    assert pred.shape == (1, 2, 501)
    assert type(loss) == torch.Tensor
