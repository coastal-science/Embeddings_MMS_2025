from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from encoder_pipeline.preprocessor.config import SpectrogramConfig



class EmbeddingModel(ABC):
    """Shared interface for embeddings - implementations will perform forward-only pass over a
    DataLoader, returning stacked (embeddings, labels)."""

    @abstractmethod
    def extract(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Runs forward-only over every batch in loader, returning
        (embeddings, labels) stacked across the whole loader."""

class HALLOEmbeddingModel(EmbeddingModel):
    """Embeddings from a checkpoint written by
    encoder_pipeline.model_trainer.train.Trainer.fit"""

    def __init__(self, checkpoint_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = bundle["model"].to(self.device)
        self.spectrogram_config: SpectrogramConfig = bundle["spectrogram_config"]

    def extract(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        embeddings, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for specs, batch_labels in loader:
                specs = specs.to(self.device).unsqueeze(1)
                embeddings.append(self.model.backbone(specs).cpu().numpy())
                labels.append(batch_labels.numpy())
        return np.concatenate(embeddings), np.concatenate(labels)
