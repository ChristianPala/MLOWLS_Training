from typing import Any, Optional

import mlflow


class MLflowLogger:
    """Wrapper around MLflow Tracking API for convenient experiment logging."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None) -> None:
        """Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for the specific run
        """
        mlflow.set_experiment(experiment_name)
        self.run_name = run_name
        self.run: Optional[mlflow.ActiveRun] = None

    def __enter__(self) -> "MLflowLogger":
        """Start MLflow run."""
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End MLflow run."""
        mlflow.end_run()

    def log_param(self, key: str, value: str | int | float | bool) -> None:
        """Log a parameter."""
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: int | float, step: Optional[int] = None) -> None:
        """Log a metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact."""
        mlflow.log_artifact(path, artifact_path)

    def log_config(self, config: Any) -> None:
        """Log all configuration parameters.

        Args:
            config: Configuration object with to_dict() method or __dict__
        """
        config_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__
        for key, value in config_dict.items():
            # Convert non-serializable values to strings
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            self.log_param(key, value)

    def log_dataset_stats(self, stats: dict[str, int | float]) -> None:
        """Log dataset statistics as metrics.

        Args:
            stats: Dictionary of dataset statistics
        """
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.log_metric(f"dataset_{key}", value)

    def log_batch_metrics(
        self, metrics_dict: dict[str, int | float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric name -> value pairs
            step: Optional step number for time series tracking
        """
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.log_metric(key, value, step=step)

    def get_run_id(self) -> Optional[str]:
        """Get current run ID for reference.

        Returns:
            Run ID if run is active, None otherwise
        """
        return self.run.info.run_id if self.run is not None else None
