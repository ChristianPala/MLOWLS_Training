import mlflow

class MLflowLogger:
    """Wrapper around MLflow Tracking API."""
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def __enter__(self):
        self.run = mlflow.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path, artifact_path=None):
        mlflow.log_artifact(path, artifact_path)