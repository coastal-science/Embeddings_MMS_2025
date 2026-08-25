from pathlib import Path
from typing import Optional

import h5py
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from encoder_pipeline.model_trainer.config import DataLoaderConfig


class SpectrogramDataset(Dataset):
    def __init__(self, hdf5_path: str, label_col: str = "Labels") -> None:
        self.hdf5_path = hdf5_path
        self._h5: Optional[h5py.File] = None

        with h5py.File(hdf5_path, "r") as h5:
            self.length = h5["spec"].shape[0]
            labels = h5[label_col].asstr()[:]
        self.classes = sorted(set(labels))
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels = [self.label_to_idx[label] for label in labels]

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec = self._file()["spec"][idx]
        return torch.from_numpy(spec).float(), self.labels[idx]


def compute_splits(hdf5_path: str, config: DataLoaderConfig) -> list[dict[str, np.ndarray]]:
    """
    Helper function to compute the train / test / val splits, and do k-fold splitting
    """
    with h5py.File(hdf5_path, "r") as h5:
        n = h5["spec"].shape[0]
        groups = h5[config.col_to_group_by].asstr()[:] if config.col_to_group_by else np.arange(n)
    unique_groups = np.unique(groups)

    def row_idx(group_subset: np.ndarray) -> np.ndarray:
        return np.where(np.isin(groups, group_subset))[0]

    if config.n_folds > 1:
        kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.split_seed)
        return [
            {"train": row_idx(unique_groups[train_pos]), "val": row_idx(unique_groups[val_pos])}
            for train_pos, val_pos in kfold.split(unique_groups)
        ]

    remaining_groups, test_groups = unique_groups, np.array([])
    if config.test_size > 0:
        remaining_groups, test_groups = train_test_split(
            remaining_groups, test_size=config.test_size, random_state=config.split_seed,
        )

    train_groups, val_groups = remaining_groups, np.array([])
    if config.val_size > 0:
        val_frac = config.val_size / (1 - config.test_size)
        train_groups, val_groups = train_test_split(
            remaining_groups, test_size=val_frac, random_state=config.split_seed,
        )

    return [{"train": row_idx(train_groups), "val": row_idx(val_groups), "test": row_idx(test_groups)}]


def save_splits(hdf5_path: str, splits: list[dict[str, np.ndarray]], out_dir: str, uid_col: str = "uid") -> str:
    with h5py.File(hdf5_path, "r") as h5:
        uids = h5[uid_col].asstr()[:]

    df = pd.DataFrame({"uid": uids})
    for fold, split in enumerate(splits):
        col = np.empty(len(uids), dtype=object)
        for split_name, idx in split.items():
            col[idx] = split_name
        df[f"fold_{fold}"] = col

    out_path = Path(out_dir) / "splits.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def build_dataloaders(hdf5_path: str, config: DataLoaderConfig, data_dir: str) -> list[dict[str, DataLoader]]:
    splits = compute_splits(hdf5_path, config)
    splits_path = save_splits(hdf5_path, splits, out_dir=f"{data_dir}/model_trainer/{mlflow.active_run().info.run_id}")
    mlflow.log_artifact(splits_path)

    dataset = SpectrogramDataset(hdf5_path)
    return [
        {
            name: DataLoader(Subset(dataset, idx), batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
            for name, idx in split.items()
        }
        for split in splits
    ]