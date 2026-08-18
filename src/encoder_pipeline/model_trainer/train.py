from torch.utils.data import DataLoader

from encoder_pipeline.model_trainer.config import ModelTrainerConfig


def train_model(config: ModelTrainerConfig, dataloaders: list[dict[str, DataLoader]]) -> None:
    """Dummy stand-in for real training -- for testing the pipeline's
    wiring, not for actually training a model. One {"train","val"[,"test"]}
    dict of DataLoaders per fold (length 1 unless config.dataloader.n_folds
    > 1); trains config.architecture once per fold."""
    for fold, loaders in enumerate(dataloaders):
        print(f"training {config.architecture} on fold {fold}: {len(loaders['train'].dataset)} train samples")
