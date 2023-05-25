import os
from pytorch_lightning.loggers.mlflow import MLFlowLogger


class CustomMLFLogger(MLFlowLogger):
    """Custom logger expands the mlflow logger."""

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str = None,
        tracking_uri: str = None,
        tags: dict = None,
        save_dir: str = "./mlruns",
        prefix: str = "",
        artifact_location: str = None,
        run_id: str = None,
        log_model: bool = True,
    ) -> None:
        """ """
        if tracking_uri is None:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        super().__init__(
            experiment_name,
            run_name,
            tracking_uri,
            tags,
            save_dir,
            log_model,
            prefix,
            artifact_location,
            run_id,
        )
