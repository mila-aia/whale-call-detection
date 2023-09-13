import optuna
from datetime import datetime
from whale.data_io.data_loader import WhaleDataModule
from whale.utils import read_yaml, LSTMTuningObjective
from whale.models import LSTM
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from pytorch_lightning.loggers import CometLogger, CSVLogger


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()

    hparams_space = read_yaml(Path(args.hparams_space).expanduser().resolve())
    project_name = args.project
    experiment_name = args.exp_name
    batch_size = args.batch_size
    search_epoch_num = args.search_num_epochs
    epoch_num = args.num_epochs
    metric_to_optimize = args.metric_to_optimize
    optimize_direction = args.optimize_direction
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    optuna_db = Path(args.optuna_db).expanduser().resolve()
    whale_dm = WhaleDataModule(
        data_dir=str(data_path),
        batch_size=batch_size,
        data_type=args.data_type,
    )
    whale_dm.setup()
    train_loader = whale_dm.train_dataloader()
    valid_loader = whale_dm.val_dataloader()
    test_loader = whale_dm.test_dataloader()

    optim_objective = LSTMTuningObjective(
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        epoch_num=search_epoch_num,
        train_loader=train_loader,
        valid_loader=valid_loader,
        hparams_space=hparams_space,
        project_name=project_name,
        experiment_name=f"{experiment_name}",
        save_dir=str(save_dir),
        direction=optimize_direction,
        metric_to_optimize=metric_to_optimize,
    )
    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db}",
        study_name=f"{experiment_name}",
        direction=optimize_direction,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True,
    )
    start_time = datetime.now()
    study.optimize(optim_objective, n_trials=args.n_trials)
    best_model_conf = hparams_space
    for hparam, value in study.best_trial.params.items():
        best_model_conf[hparam] = value

    model = LSTM(
        input_dim=args.input_dim,
        hidden_dim=best_model_conf["hidden_dim"],
        num_layers=best_model_conf["num_layers"],
        num_classes=args.num_classes,
        bidirectional=best_model_conf["bidirectional"],
        reg_loss_weight=best_model_conf["reg_loss_weight"],
        lr=best_model_conf["lr"],
    )

    exp_logger = CometLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        save_dir=str(save_dir),
    )
    exp_logger.experiment.log_parameter("run_name", "best_trial")
    csv_logger = CSVLogger(
        save_dir=str(save_dir),
        name=experiment_name,
    )
    early_stopper = EarlyStopping(
        monitor=metric_to_optimize,
        patience=2,
        mode=optimize_direction[0:3],
        verbose=True,
    )
    checkpoint_saver = ModelCheckpoint(
        monitor=metric_to_optimize,
        mode=optimize_direction[0:3],  # min or max
        auto_insert_metric_name=True,
        save_on_train_epoch_end=True,
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=epoch_num,
        accelerator="auto",
        move_metrics_to_cpu=True,
        log_every_n_steps=10,
        fast_dev_run=False,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        logger=[exp_logger, csv_logger],
        callbacks=[early_stopper, checkpoint_saver],
    )

    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    print(
        "Hyper-parameters search completed!"
        + f"Duration:{datetime.now() - start_time}"
    )
    print("Evaluating the best model...")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


def parse_args() -> Namespace:
    description = "Train an LSTM model for whall call recognition."
    arg_parser = ArgumentParser(
        description=description, formatter_class=ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="integer value seed for global random state",
    )
    arg_parser.add_argument(
        "--project",
        default="whale-call-detection",
        type=str,
        help="name of the project",
    )
    arg_parser.add_argument(
        "--exp-name",
        default="test_exp",
        type=str,
        help="name of the experiment",
    )
    arg_parser.add_argument(
        "--save-dir",
        default="./wandb_log/",
        type=str,
        help="path to the wandb logging directory",
    )
    arg_parser.add_argument(
        "--input-dim",
        default=129,
        type=int,
        help="dimension of the input sequene to the LSTM",
    )
    arg_parser.add_argument(
        "--num-classes",
        default=2,
        type=int,
        help="number of classes",
    )
    arg_parser.add_argument(
        "--data-path",
        default="./data/",
        type=str,
        help="string name of the data directory,"
        + " containing train/val/test splits",
    )
    arg_parser.add_argument(
        "--data-type",
        choices=["spec", "waveform"],
        default="spec",
        type=str,
        help="data type to use for training",
    )
    arg_parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="number of instances in a single training batch",
    )
    arg_parser.add_argument(
        "--search-num-epochs",
        default=15,
        type=int,
        help="the total number of epochs for hyper-parameters search phase",
    )
    arg_parser.add_argument(
        "--num-epochs",
        default=30,
        type=int,
        help="the total number of epochs",
    )
    arg_parser.add_argument(
        "--optuna-db",
        default="optuna.sqlite3",
        type=str,
        help="path to Optuna logs database",
    )
    arg_parser.add_argument(
        "--hparams-space",
        default="hyperparam.yaml",
        type=str,
        help="path to the YAML file containing the hyperparameter space",
    )
    arg_parser.add_argument(
        "--n-trials",
        default=20,
        type=int,
        help="number trials for the hyper-parameters search",
    )
    arg_parser.add_argument(
        "--metric-to-optimize",
        default="overall_val_loss",
        type=str,
        help="metric to optimize for model selection",
    )
    arg_parser.add_argument(
        "--optimize-direction",
        choices=["minimize", "maximize"],
        default="minimize",
        type=str,
        help="direction of optimization",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
