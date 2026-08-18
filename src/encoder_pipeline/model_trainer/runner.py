import argparse
from pathlib import Path

import mlflow

from encoder_pipeline.common.config import load_pipeline_config
from encoder_pipeline.common.tracking import configure_mlflow, flatten_params
from encoder_pipeline.model_trainer.config import ModelTrainerConfig
from encoder_pipeline.model_trainer.data_loader import build_dataloaders
from encoder_pipeline.model_trainer.train import train_model


def run_model_trainer(config: ModelTrainerConfig, dataset_path: str) -> None:
    """train_model wrapped in an MLflow run, nested under the
    preprocessor run that produced dataset_path (found by matching its
    dataset_path param) so training sub-runs are grouped under their
    dataset's run. Falls back to a plain top-level run if no match is found.

    Training runs also log dataset_path for traceability, so more than one
    run can match -- ordering by start_time ASC picks the preprocessor run
    that produced the dataset (always the earliest), not a sibling training
    run that merely trained on it."""
    matches = mlflow.search_runs(
        filter_string=f"params.dataset_path = '{dataset_path}'", order_by=["start_time ASC"], max_results=1,
    )
    parent_run_id = matches.iloc[0]["run_id"] if not matches.empty else None

    def _train() -> None:
        with mlflow.start_run(nested=parent_run_id is not None, run_name=config.run_name):
            mlflow.log_params(flatten_params("model_trainer", config.model_dump()))
            mlflow.log_param("dataset_path", dataset_path)
            dataloaders = build_dataloaders(dataset_path, config.dataloader)
            train_model(config, dataloaders)

    if parent_run_id is not None:
        with mlflow.start_run(run_id=parent_run_id):
            _train()
    else:
        _train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset produced by preprocessor.")
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)

    configure_mlflow(pipeline_config)
    run_model_trainer(pipeline_config.model_trainer, args.dataset_path)


if __name__ == "__main__":
    main()
