# Turns K. Palmer et al.'s reconstructed split CSVs into per-model splits files for the
# freshly built annotations. For each reconstructed_splits/train_birdnet0N.csv (and
# full_train.csv), splits_<name>.csv (uid, fold_0) marks an annotation "train" if that
# exact row is in the train CSV, "test" if it is in holdout_eval.csv, else blank.
# Rows are matched directly on (Soundfile stem, FileBeginSec, FileEndSec) -- the columns
# every one of these CSVs carries -- and uid comes from all_anno.
from pathlib import Path

import pandas as pd

FULL_TRAIN = "full_train.csv"
HOLDOUT_EVAL = "holdout_eval.csv"


def _stem(soundfile: object) -> str:
    return Path(str(soundfile)).stem.lower()


def _row_keys(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(zip(df["Soundfile"].map(_stem), df["FileBeginSec"].round(3), df["FileEndSec"].round(3))),
        index=df.index,
    )


_KEY_COLS = {"Soundfile", "FileBeginSec", "FileEndSec", "Augmented", "ShiftSec"}


def _keys_in(split_csv: Path) -> set:
    """(stem, begin, end) row keys in a split CSV, excluding K. Palmer's pre-baked
    time-shifted / augmented rows -- the pipeline does time shifting itself."""
    df = pd.read_csv(split_csv, low_memory=False, usecols=lambda c: c in _KEY_COLS)
    if "Augmented" in df.columns:
        df = df[~df["Augmented"].astype(bool)]
    if "ShiftSec" in df.columns:
        df = df[df["ShiftSec"].fillna(0) == 0]
    return set(_row_keys(df))


def generate_splits_files(all_anno: pd.DataFrame, out_dir: Path, splits_src_dir: Path) -> None:
    """Write one splits_<name>.csv per train_*.csv / full_train.csv in splits_src_dir, all testing on holdout_eval.csv."""
    anno_key = _row_keys(all_anno)
    test_keys = _keys_in(splits_src_dir / HOLDOUT_EVAL)

    out_dir.mkdir(parents=True, exist_ok=True)
    for train_csv in sorted(splits_src_dir.glob("train_*.csv")) + [splits_src_dir / FULL_TRAIN]:
        train_keys = _keys_in(train_csv) - test_keys
        assert not (train_keys & test_keys), f"{train_csv.name}: row in both train and test"

        fold = pd.Series(pd.NA, index=all_anno.index, dtype="object")
        fold[anno_key.isin(train_keys)] = "train"
        fold[anno_key.isin(test_keys)] = "test"

        name = train_csv.stem.removeprefix("train_")
        pd.DataFrame({"uid": all_anno["uid"], "fold_0": fold}).to_csv(out_dir / f"splits_{name}.csv", index=False)
        print(f"splits_{name}.csv: {fold.value_counts(dropna=False).to_dict()}")
