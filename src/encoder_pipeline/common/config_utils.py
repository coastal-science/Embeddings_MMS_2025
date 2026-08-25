from pathlib import Path
from typing import Optional, TypeVar

import yaml
from pydantic import BaseModel

from encoder_pipeline.common.base import StrictBaseModel
from encoder_pipeline.embeddings.config import EmbeddingsConfig
from encoder_pipeline.model_trainer.config import ModelTrainerConfig
from encoder_pipeline.preprocessor.config import PreprocessorConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _read_yaml_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Load a YAML file and validate it with a Pydantic model."""
    return model.model_validate(_read_yaml_dict(path))


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override onto base; override's values win on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(base_path: Path, override_path: Optional[Path] = None) -> "PipelineConfig":
    """Loads override_path (if given) deep-merged onto base_path, so every
    default lives in one place and an override only needs to specify what
    it's overriding."""
    base = _read_yaml_dict(base_path)
    merged = deep_merge(base, _read_yaml_dict(override_path)) if override_path else base
    return PipelineConfig.model_validate(merged)


class PipelineConfig(StrictBaseModel):
    """Top-level config shared across every pipeline stage; each stage owns
    its own section, filled in as that stage gets built out."""

    experiment_name: str
    """MLflow experiment name (level 1) every run/sub-run in this config gets logged under."""

    mlflow_tracking_uri: Optional[str] = None
    """MLflow tracking server URL. If unset, MLflow's own default resolution is used."""

    data_dir: str = "data"
    """Base directory every stage writes local artifacts under, each in its
    own subdirectory (e.g. data_dir/preprocessor, data_dir/model_trainer)."""

    preprocessor: PreprocessorConfig
    model_trainer: ModelTrainerConfig = ModelTrainerConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
