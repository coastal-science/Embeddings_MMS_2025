"""Runs the encoder pipeline end-to-end. Each stage is skipped if its output
is already given explicitly in the config (e.g. dataset_path), and runs
otherwise, logging an MLflow run under config.experiment_name."""

import argparse
from pathlib import Path

from encoder_pipeline.common.config import PipelineConfig, load_pipeline_config
from encoder_pipeline.common.tracking import configure_mlflow
from encoder_pipeline.model_trainer.runner import run_model_trainer
from encoder_pipeline.preprocessor.runner import run_preprocessor


def run_pipeline(config: PipelineConfig) -> None:
    configure_mlflow(config)

    dataset_path = run_preprocessor(config.preprocessor)
    run_model_trainer(config.model_trainer, dataset_path)
    # Future stages (evaluation, embeddings) plug in here the same way:
    # check config for an explicit output, skip and reuse it if given,
    # otherwise run the stage and log a nested run under dataset_path.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    args = parser.parse_args()
    config = load_pipeline_config(args.config, args.override)
    run_pipeline(config)


if __name__ == "__main__":
    main()
