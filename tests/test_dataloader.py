from whale.data_io.data_loader import WhaleDataModule


def test_init_datamodule() -> None:

    data_path = "tests/sample/"
    MyDataModule = WhaleDataModule(data_dir=data_path, batch_size=2)
    MyDataModule.setup()
    train_loader = MyDataModule.train_dataloader()
    dataiter = iter(train_loader)
    inp, target = next(dataiter)
    assert (
        MyDataModule.train_ds is not None and len(MyDataModule.train_ds) == 9
    )
    assert MyDataModule.valid_ds is not None and len(MyDataModule.vali_ds) == 9
    assert inp.shape == (2, 1, 501)
    assert target.shape == (2, 501)
