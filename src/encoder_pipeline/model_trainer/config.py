from typing import Optional

from pydantic import BaseModel

from encoder_pipeline.model_trainer.data_loader import DataLoaderConfig


class ModelTrainerConfig(BaseModel):
    """Dummy stand-in for real training config -- for testing the pipeline's
    wiring, not for actually training a model."""

    architecture: str = "dummy"
    run_name: Optional[str] = None
    """MLflow sub-run name. If unset, MLflow auto-generates one."""
    dataloader: DataLoaderConfig = DataLoaderConfig()
