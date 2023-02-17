from whale.data_io.data_loader import WhaleDataModule


def test_init_datamodule() -> None:

    data_path = "tests/sample/"
    MyDataModule = WhaleDataModule(data_dir=data_path, batch_size=2)
    MyDataModule.setup()
    train_ds = MyDataModule.train_ds
    train_loader = MyDataModule.train_dataloader()
    dataiter = iter(train_loader)
    inp, target = next(dataiter)
    assert train_ds is not None and len(train_ds) == 4
    assert inp.shape == (2, 1, 501)
    assert target.shape == (2, 501)
