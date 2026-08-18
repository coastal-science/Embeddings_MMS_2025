import mlflow

from encoder_pipeline.common.config import PipelineConfig


def configure_mlflow(config: PipelineConfig) -> None:
    """Sets the MLflow tracking URI (if given) and active experiment."""
    if config.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)


def flatten_params(prefix: str, obj: dict) -> dict:
    """Flattens a nested dict into dotted-key: value pairs, for mlflow.log_params."""
    flat = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_params(full_key, value))
        else:
            flat[full_key] = value
    return flat
