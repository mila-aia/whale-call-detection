import torch
from whale.models import UNet, LSTM


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


def test_lstm() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM(input_dim=4, hidden_dim=6)
    model = model.to(device)

    sig = torch.zeros([5, 2, 4]).to(device)
    class_logits, reg_out = model.forward(sig)

    assert class_logits.shape == (5, 2)
    assert reg_out.shape == (5, 1)
