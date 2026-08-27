# Turns K. Palmer et al.'s reconstructed split CSVs into per-model splits files for the
# freshly built annotations. For each reconstructed_splits/train_birdnet0N.csv (and
# full_train.csv), splits_<name>.csv (uid, fold_0) marks an annotation "train" if its
# recording is in that train CSV and "test" if its recording is in holdout_eval.csv.
#
# Matching is at the recording level -- every annotation whose LocalPath stem appears in
# a split CSV's Soundfile column is assigned, so the pipeline's algorithmically-tiled
# "Background" rows (which sit at their own timestamps and would never match Palmer's
# rows exactly) land in whichever split their file belongs to. holdout_eval.csv wins any
# tie: a recording it shares with a per-model train CSV goes entirely to test, so no
# frame of a held-out recording can leak into training.
#
# Rows that land in neither split are dropped from the file, not written as blanks: the
# per-model CSVs deliberately train on a subset of recordings, and a row whose audio
# isn't present locally has no LocalPath to match on. uid comes from all_anno.
from pathlib import Path

import pandas as pd

FULL_TRAIN = "full_train.csv"
HOLDOUT_EVAL = "holdout_eval.csv"

_KEY_COLS = {"Soundfile", "Augmented", "ShiftSec"}


def _stem(path: object) -> str:
    return Path(str(path)).stem.lower()


def _recording_stems(split_csv: Path) -> set:
    """Recording stems (lowercased Soundfile stem) in a split CSV, excluding
    K. Palmer's pre-baked time-shifted / augmented rows -- the pipeline does
    time shifting itself."""
    df = pd.read_csv(split_csv, low_memory=False, usecols=lambda c: c in _KEY_COLS)
    if "Augmented" in df.columns:
        df = df[~df["Augmented"].astype(bool)]
    if "ShiftSec" in df.columns:
        df = df[df["ShiftSec"].fillna(0) == 0]
    return set(df["Soundfile"].map(_stem))


def generate_splits_files(all_anno: pd.DataFrame, out_dir: Path, splits_src_dir: Path) -> None:
    """Write one splits_<name>.csv per train_*.csv / full_train.csv in splits_src_dir, all testing on holdout_eval.csv."""
    anno_stem = all_anno["LocalPath"].map(lambda p: _stem(p) if pd.notna(p) else None)
    test_stems = _recording_stems(splits_src_dir / HOLDOUT_EVAL)

    out_dir.mkdir(parents=True, exist_ok=True)
    for train_csv in sorted(splits_src_dir.glob("train_*.csv")) + [splits_src_dir / FULL_TRAIN]:
        train_stems = _recording_stems(train_csv) - test_stems

        fold = pd.Series(pd.NA, index=all_anno.index, dtype="object")
        fold[anno_stem.isin(train_stems)] = "train"
        fold[anno_stem.isin(test_stems)] = "test"
        assigned = fold.notna()

        name = train_csv.stem.removeprefix("train_")
        out = pd.DataFrame({"uid": all_anno.loc[assigned, "uid"], "fold_0": fold[assigned]})
        out.to_csv(out_dir / f"splits_{name}.csv", index=False)
        print(f"splits_{name}.csv: {out['fold_0'].value_counts().to_dict()} ({len(all_anno) - len(out)} rows dropped)")
