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
            prefix,
            artifact_location,
            run_id,
        )

    def log_model(self, local_path: str, file_name: str) -> None:
        self.experiment.log_artifact(
            run_id=self.run_id, local_path=local_path, artifact_path=file_name
        )
