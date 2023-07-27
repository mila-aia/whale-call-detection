from whale.data_io.data_loader import WhaleDataModule
from whale.utils.loggers import CustomMLFLogger
from whale.models import LSTM
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    experiment_name = args.exp_name
    run_name = args.run_name
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    epoch_num = args.num_epochs
    metric_to_optimize = args.metric_to_optimize
    mlruns_dir = Path(args.mlruns_dir).expanduser().resolve()

    whale_dm = WhaleDataModule(
        data_dir=str(data_path),
        batch_size=batch_size,
        data_type=args.data_type,
    )
    whale_dm.setup()
    train_loader = whale_dm.train_dataloader()
    valid_loader = whale_dm.val_dataloader()
    test_loader = whale_dm.test_dataloader()

    exp_logger = CustomMLFLogger(
        experiment_name=experiment_name,
        save_dir=str(mlruns_dir),
        run_name=run_name,
        log_model="all",
    )

    early_stopper = EarlyStopping(
        monitor=metric_to_optimize, patience=5, mode="min", verbose=True
    )
    checkpoint_saver = ModelCheckpoint(
        monitor=metric_to_optimize,
        mode="min",
        auto_insert_metric_name=True,
        save_on_train_epoch_end=True,
        save_top_k=2,
    )

    trainer = Trainer(
        max_epochs=epoch_num,
        accelerator="auto",
        move_metrics_to_cpu=True,
        log_every_n_steps=10,
        fast_dev_run=False,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        logger=exp_logger,
        callbacks=[early_stopper, checkpoint_saver],
    )

    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=args.num_classes,
        bidirectional=args.bidirectional,
        reg_loss_weight=args.reg_loss_weight,
        lr=args.lr,
        call_time_min_val=whale_dm.target_time_min_val,
        call_time_max_val=whale_dm.target_time_max_val,
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


def parse_args() -> Namespace:
    description = "Train an LSTM model for whall call recognition."
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--exp-name",
        default="test_exp",
        type=str,
        help="name of the MLflow experiment",
    )

    arg_parser.add_argument(
        "--run-name",
        default="test_run",
        type=str,
        help="name of the MLflow run",
    )
    arg_parser.add_argument(
        "--data-path",
        default="./data/",
        type=str,
        help="string name of the data directory,"
        + " containing train/val/test splits",
    )
    arg_parser.add_argument(
        "--input-dim",
        default=129,
        type=int,
        help="dimension of the input sequene to the LSTM",
    )
    arg_parser.add_argument(
        "--hidden-dim",
        default=128,
        type=int,
        help="dimension of the hidden state of the LSTM",
    )
    arg_parser.add_argument(
        "--num-layers",
        default=3,
        type=int,
        help="number of layers in the LSTM",
    )
    arg_parser.add_argument(
        "--num-classes",
        default=2,
        type=int,
        help="number of classes",
    )
    arg_parser.add_argument(
        "--bidirectional",
        default=False,
        action="store_true",
        help="whether the LSTM is bidirectional",
    )
    arg_parser.add_argument(
        "--reg-loss-weight",
        default=0.5,
        type=float,
        help="weight for the regression loss. Note that the total loss is"
        + " the sum of the cross entropy loss and the regression loss",
    )
    arg_parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="learning rate",
    )
    arg_parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="number of instances in a single training batch",
    )
    arg_parser.add_argument(
        "--num-epochs",
        default=30,
        type=int,
        help="the total number of epochs",
    )
    arg_parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="integer value seed for global random state",
    )
    arg_parser.add_argument(
        "--mlruns-dir",
        default="mlruns/",
        type=str,
        help="path to the MLflow mlruns directory",
    )
    arg_parser.add_argument(
        "--data-type",
        choices=["spec", "waveform"],
        default="spec",
        type=str,
        help="data type to use for training",
    )
    arg_parser.add_argument(
        "--metric-to-optimize",
        default="overall_val_loss",
        type=str,
        help="metric to optimize for model selection",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
