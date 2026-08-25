from typing import Literal, Optional, TypeAlias, Union

import torch
from pydantic import Field

from encoder_pipeline.common.base import StrictBaseModel


class DataLoaderConfig(StrictBaseModel):
    """torch DataLoader + split params for SpectrogramDataset."""

    batch_size: int = 32
    shuffle: bool = True
    n_folds: int = 1
    test_size: float = 0.1
    val_size: float = 0.1
    split_seed: int = 10
    num_workers: int = 0
    col_to_group_by: Optional[str] = None
    """Metadata column to split on, e.g. 'LocalPath' (file-level) or a
    deployment id column"""
class SpectrogramSSLAugmentConfig(StrictBaseModel):
    time_mask_frac: float = 0.15
    freq_mask_frac: float = 0.15
    shift_frac: float = 0.1
    noise_std: float = 0.05


ResNetVariant = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
EfficientNetVariant = Literal[
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
]
backbone: TypeAlias = Union[ResNetVariant, EfficientNetVariant]


class SimCLRConfig(StrictBaseModel):
    """Backbone + Lightly's SimCLR projection head"""

    backbone_name: backbone = "resnet18"
    augment: SpectrogramSSLAugmentConfig = SpectrogramSSLAugmentConfig()
    projection_hidden_dim: int = 512
    projection_out_dim: int = 128
    temperature: float = 0.5
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class ClassifierConfig(StrictBaseModel):
    """Backbone + linear classification head"""

    backbone_name: backbone = "resnet18"
    num_classes: int
    """Must match the training dataset's label cardinality (SpectrogramDataset.classes)."""
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-6
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class ModelTrainerConfig(StrictBaseModel):
    run_name: Optional[str] = None
    """MLflow sub-run name. If unset, MLflow auto-generates one."""
    dataloader: DataLoaderConfig = DataLoaderConfig()
    paradigm: Literal["simclr", "classifier"] = "simclr"
    """Which Trainer train_model runs."""
    simclr: SimCLRConfig = SimCLRConfig()
    classifier: Optional[ClassifierConfig] = None
    """Required when paradigm == 'classifier'."""
