from pathlib import Path
from typing import Optional

import h5py
import mlflow
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class DataLoaderConfig(BaseModel):
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
    deployment id column -- every row sharing a value stays in the same
    split. None splits row by row."""


class SpectrogramDataset(Dataset):
    """Reads spec + a label column from an HDF5 file built by
    encoder_pipeline.preprocessor.Dataset.build_hdf5. Opens the file lazily,
    once per worker process, since an h5py.File isn't safe to share across
    DataLoader worker processes."""

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
    """Row indices into the HDF5's spec/metadata arrays, split per
    config.col_to_group_by (every row sharing a value stays on the same
    side, e.g. every row from one file or one deployment; None splits row
    by row). n_folds>1 gives one {"train", "val"} dict per fold (KFold over
    the unique group values); n_folds=1 gives a single {"train", "val",
    "test"} dict, val carved out of what's left after test (val_size scaled
    by 1 - test_size)."""
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
    """One row per uid, one column per fold ("fold_0", "fold_1", ...)
    holding that uid's split ("train"/"val"/"test") for that fold -- uid
    resolved from the HDF5's uid_col so the log is readable without the
    row-index/HDF5 mapping. Writes to out_dir/splits.csv, returning the
    path written to."""
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


def build_dataloaders(hdf5_path: str, config: DataLoaderConfig) -> list[dict[str, DataLoader]]:
    """One {"train","val","test"} dict of DataLoaders for a plain split, or
    one {"train","val"} dict per fold for k-fold. Also saves + logs the
    split as an MLflow artifact (data/model_trainer/{run_id}/splits.csv) --
    call this from within an active mlflow run so it lands in the right
    place."""
    splits = compute_splits(hdf5_path, config)
    splits_path = save_splits(hdf5_path, splits, out_dir=f"data/model_trainer/{mlflow.active_run().info.run_id}")
    mlflow.log_artifact(splits_path)

    dataset = SpectrogramDataset(hdf5_path)
    return [
        {
            name: DataLoader(Subset(dataset, idx), batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers)
            for name, idx in split.items()
        }
        for split in splits
    ]