import argparse
import pickle
from pathlib import Path

import mlflow

from encoder_pipeline.common.config_utils import load_pipeline_config
from encoder_pipeline.common.mlflow_utils import configure_mlflow, flatten_params
from encoder_pipeline.preprocessor.config import PreprocessorConfig
from encoder_pipeline.preprocessor.dataset import Dataset


def run_preprocessor(config: PreprocessorConfig, data_dir: str) -> str:
    dataset = Dataset(config.spectrogram, config.audio_file, config.annotation, config.dataset, data_dir, config.run_name)
    dataset.build_hdf5()
    # search if the hdf5 file has been logged to mlflow, by content hash (ignoring the timestamp)
    content_hash = Path(dataset.out_file).stem.rsplit("_", 1)[0]
    existing = mlflow.search_runs(
        filter_string=f"params.dataset_path LIKE '%{content_hash}%'", order_by=["start_time ASC"], max_results=1,
    )
    if not existing.empty:
        print(f"reusing run_id={existing.iloc[0]['run_id']} dataset_path={dataset.out_file}")
    else:
        with mlflow.start_run(run_name=config.run_name):
            mlflow.log_params(flatten_params("preprocessor", config.model_dump()))
            mlflow.log_param("dataset_path", dataset.out_file)
            spec_config_path = Path(f"{data_dir}/preprocessor/{mlflow.active_run().info.run_id}/spectrogram_config.pkl")
            spec_config_path.parent.mkdir(parents=True, exist_ok=True)
            with spec_config_path.open("wb") as f:
                pickle.dump(config.spectrogram, f)
            mlflow.log_artifact(str(spec_config_path))
            print(f"run_id={mlflow.active_run().info.run_id} dataset_path={dataset.out_file}")
    return dataset.out_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Base config, e.g. configs/base.yaml.")
    parser.add_argument("--override", type=Path, default=None, help="Override config, deep-merged onto --config.")
    args = parser.parse_args()
    pipeline_config = load_pipeline_config(args.config, args.override)

    configure_mlflow(pipeline_config)
    run_preprocessor(pipeline_config.preprocessor, pipeline_config.data_dir)


if __name__ == "__main__":
    main()
