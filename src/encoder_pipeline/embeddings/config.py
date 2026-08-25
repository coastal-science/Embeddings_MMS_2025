from typing import Literal, Optional

from encoder_pipeline.common.base import StrictBaseModel


class EmbeddingsConfig(StrictBaseModel):
    source: Literal["HALLO_encoder_collection", "perch_hoplite"] = "HALLO_encoder_collection"
    """"HALLO_encoder_collection": run_id's model_trainer checkpoint, from
    local disk if present, else downloaded from the MLflow tracking server.
    "perch_hoplite": not yet implemented."""
    mlflow_id: Optional[str] = None
    """run_id to embed. null = the run this pipeline invocation's own
    model_trainer stage just produced -- see pipeline.run_pipeline."""
    linear_probe_epochs: int = 1000
    linear_probe_lr: float = 3e-4
