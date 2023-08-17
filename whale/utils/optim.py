from pathlib import Path
import yaml  # type: ignore
from optuna.trial import Trial
from pytorch_lightning import Trainer
from whale.models import LSTM
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from typing import Literal
from torch.utils.data import DataLoader


def read_yaml(yaml_fp: Path) -> dict:
    with open(yaml_fp) as yml_file:
        data = yaml.safe_load(yml_file)
    return data


def get_params(trial: Trial, hparams_space: dict) -> dict:
    model_conf: dict = dict()
    for hparam, space in hparams_space.items():
        if isinstance(space, dict):
            if space["type"] == "categorical":
                model_conf[hparam] = trial.suggest_categorical(
                    hparam, space["categories"]
                )
            elif space["type"] == "float":
                if "use_log" in space:
                    model_conf[hparam] = trial.suggest_float(
                        hparam,
                        space["min"],
                        space["max"],
                        log=space["use_log"],
                    )
                else:
                    model_conf[hparam] = trial.suggest_float(
                        hparam, space["min"], space["max"], step=space["step"]
                    )
            elif space["type"] == "int":
                model_conf[hparam] = trial.suggest_int(
                    hparam, space["min"], space["max"], step=space["step"]
                )
        else:
            model_conf[hparam] = space
    return model_conf


class LSTMTuningObjective:
    def __init__(
        self,
        hparams_space: dict,
        num_classes: int = 2,
        input_dim: int = 129,
        epoch_num: int = 20,
        train_loader: DataLoader = None,
        valid_loader: DataLoader = None,
        project_name: str = None,
        experiment_name: str = None,
        save_dir: str = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        metric_to_optimize: str = "overall_val_loss",
    ) -> None:
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.epoch_num = epoch_num
        self.hparams_space = hparams_space
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.direction = direction
        self.metric_to_optimize = metric_to_optimize

    def __call__(self, trial: Trial) -> float:
        model_conf = get_params(trial, self.hparams_space)
        model = LSTM(
            input_dim=self.input_dim,
            hidden_dim=model_conf["hidden_dim"],
            num_layers=model_conf["num_layers"],
            num_classes=self.num_classes,
            bidirectional=model_conf["bidirectional"],
            reg_loss_weight=model_conf["reg_loss_weight"],
            lr=model_conf["lr"],
        )
        early_stopper = EarlyStopping(
            monitor=self.metric_to_optimize,
            patience=2,
            mode=self.direction[0:3],
            verbose=True,
        )

        _logger = WandbLogger(
            project=self.project_name,
            group=self.experiment_name,
            log_model=False,  # avoid uploading the model to wandb
            save_dir=self.save_dir,
            name=f"trial_{trial.number}",
        )

        trainer = Trainer(
            max_epochs=self.epoch_num,
            accelerator="auto",
            move_metrics_to_cpu=True,
            log_every_n_steps=10,
            fast_dev_run=False,
            enable_checkpointing=True,
            check_val_every_n_epoch=1,
            logger=_logger,
            callbacks=[early_stopper],
        )

        trainer.fit(
            model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.valid_loader,
        )
        metric_history = _logger.experiment.get_metric_history(
            _logger.run_id, self.metric_to_optimize
        )
        metric_history_array = np.array(
            [metric_value.value for metric_value in metric_history]
        )
        # get the best value of the metric from metric history
        if self.direction == "minimize":
            metric_to_optimize = metric_history_array.min()
        elif self.direction == "maximize":
            metric_to_optimize = metric_history_array.max()
        return metric_to_optimize
