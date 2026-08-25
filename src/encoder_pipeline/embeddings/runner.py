import argparse
from pathlib import Path

import h5py
import mlflow
from torch.utils.data import DataLoader

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, flatten_params, download_artifact
from encoder_pipeline.embeddings.config import EmbeddingsConfig
from encoder_pipeline.embeddings.embed import HALLOEmbeddingModel
from encoder_pipeline.evaluation.linear_probe import LinearProbe
from encoder_pipeline.model_trainer.data_loader import build_dataloaders


def generate_embeddings(
    config: EmbeddingsConfig, dataloaders: list[dict[str, DataLoader]], run_id: str, data_dir: str,
) -> None:
    """"""
    checkpoint_dir = download_artifact(data_dir, run_id)
    out_dir = Path(f"{data_dir}/embeddings/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(flatten_params("embeddings", config.model_dump()))
        for fold, loaders in enumerate(dataloaders):
            best_path = checkpoint_dir / f"fold{fold}_best.pt"
            checkpoint_path = best_path if best_path.exists() else checkpoint_dir / f"fold{fold}_last.pt"
            source = HALLOEmbeddingModel(str(checkpoint_path))
            embeddings = {split: source.extract(loader) for split, loader in loaders.items()}
            out_path = out_dir / f"fold{fold}.h5"
            with h5py.File(out_path, "w") as h5:
                for split, (split_embeddings, split_labels) in embeddings.items():
                    h5.create_dataset(f"{split}_embeddings", data=split_embeddings)
                    h5.create_dataset(f"{split}_labels", data=split_labels)

            with mlflow.start_run(nested=True, run_name=f"embeddings_fold{fold}"):
                mlflow.log_param("checkpoint_path", str(checkpoint_path))
                mlflow.log_param("embeddings_path", str(out_path))
                mlflow.log_param("embedding_dim", next(iter(embeddings.values()))[0].shape[1])
                for split, (split_embeddings, _) in embeddings.items():
                    mlflow.log_param(f"{split}_n_samples", split_embeddings.shape[0])
                mlflow.log_artifact(str(out_path))

                if "train" in embeddings:
                    # run linear probing and store metric curves in mlflow post training linear layer
                    linear_probe = LinearProbe(epochs=config.linear_probe_epochs, lr=config.linear_probe_lr)
                    linear_probe_metrics, linear_probe_loss_curves = linear_probe.evaluate(embeddings)
                    for curve_key, curve_values in linear_probe_loss_curves.items():
                        for epoch, value in enumerate(curve_values):
                            mlflow.log_metric(f"linear_probe_{curve_key}", value, step=epoch)
                    for key, value in linear_probe_metrics.items():
                        mlflow.log_metric(f"linear_probe_{key}", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset produced by preprocessor.")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="model_trainer MLflow run that wrote the checkpoints to embed. Defaults to embeddings.mlflow_id in --config.",
    )
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)
    run_id = args.run_id or pipeline_config.embeddings.mlflow_id
    if run_id is None:
        parser.error("--run-id is required unless embeddings.mlflow_id is set in --config.")

    configure_mlflow(pipeline_config)
    with mlflow.start_run(run_id=run_id):
        dataloaders = build_dataloaders(args.dataset_path, pipeline_config.model_trainer.dataloader, pipeline_config.data_dir)
    generate_embeddings(pipeline_config.embeddings, dataloaders, run_id, pipeline_config.data_dir)


if __name__ == "__main__":
    main()
