import torch
import torch.nn as nn
import torchvision.models as tv_models
from lightly.models.modules import SimCLRProjectionHead

from encoder_pipeline.model_trainer.config import ClassifierConfig, EfficientNetVariant, ResNetVariant, SimCLRConfig, backbone


class ResNetBackbone(nn.Module):
    """torchvision ResNet family"""

    def __init__(self, variant: ResNetVariant) -> None:
        super().__init__()
        resnet = getattr(tv_models, variant)()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_features = resnet.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(start_dim=1)


class EfficientNetBackbone(nn.Module):
    """torchvision EfficientNet family"""

    def __init__(self, variant: EfficientNetVariant) -> None:
        super().__init__()
        effnet = getattr(tv_models, variant)()
        first_conv = effnet.features[0][0]
        effnet.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
            stride=first_conv.stride, padding=first_conv.padding, bias=False,
        )
        self.features = effnet.features
        self.out_features = effnet.classifier[1].in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(start_dim=1)


def build_backbone(name: backbone) -> nn.Module:
    if name.startswith("efficientnet"):
        return EfficientNetBackbone(name)
    return ResNetBackbone(name)


class SimCLRModel(nn.Module):
    def __init__(self, config: SimCLRConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone_name)
        self.projection_head = SimCLRProjectionHead(
            self.backbone.out_features, config.projection_hidden_dim, config.projection_out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.backbone(x))


class ClassifierModel(nn.Module):
    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self.backbone = build_backbone(config.backbone_name)
        self.classifier = nn.Linear(self.backbone.out_features, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))
