import os
from jsonargparse import Namespace
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning import LightningModule, Trainer


class LogConfigCallback(SaveConfigCallback):
    """Logs the config file as an artifact
    Parameters
    ----------
    SaveConfigCallback : SaveConfigCallback
        SaveConfigCallback callback class as base class.
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = True,
        multifile: bool = False,
    ) -> None:
        """Initializes the callback.
        Parameters
        ----------
        parser : LightningArgumentParser
            Parser of the arguments object.
        config : Namespace
            Configuration namespace to be used to extract configs
        config_filename : str, optional
            Configuration file name, by default "config.yaml"
        overwrite : bool, optional
            Option to overwrite existing config, by default True
        multifile : bool, optional
            To be used if several config files exists, by default False
        """
        super().__init__(
            config_filename=config_filename,
            config=config,
            parser=parser,
            overwrite=overwrite,
            multifile=multifile,
        )

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        log_dir = trainer.log_dir  # this broadcasts the directory
        fs = get_filesystem(log_dir)

        fs.makedirs(log_dir, exist_ok=True)
        config_path = os.path.join(
            log_dir,
            trainer.logger._experiment_id,
            trainer.logger._run_id,
            "artifacts",
            self.config_filename,
        )
        self.parser.save(
            self.config,
            config_path,
            skip_none=False,
            overwrite=False,
            multifile=False,
        )
