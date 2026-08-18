from typing import Optional

from pydantic import BaseModel

from encoder_pipeline.preprocessor.annotation import AnnotationConfig
from encoder_pipeline.preprocessor.dataset import DatasetConfig
from encoder_pipeline.preprocessor.spectrogram import SpectrogramConfig


class PreprocessorConfig(BaseModel):
    spectrogram: SpectrogramConfig = SpectrogramConfig()
    annotation: AnnotationConfig = AnnotationConfig()
    dataset: DatasetConfig
    run_name: Optional[str] = None # MLflow run name. If unset, MLflow auto-generates one.