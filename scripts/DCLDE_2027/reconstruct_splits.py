# Reconstructs the train/holdout split from K. Palmer et al., "Population-Level Acoustic
# Classification of Salish Sea Killer Whales: Integrating Biologically Informed Call Type
# Balancing to Build Robust Models for Conservation Monitoring". Reconstruction logic copied
# from exp_1_reproduce_k_palmer_2026_results_corrected/reconstruct_splits.py (itself from
# exp_0's original_train_test_split_analysis.ipynb + verify_data_leak_in_original_exps.ipynb),
import re
from pathlib import Path

import pandas as pd

PROVIDERS = ['OrcaSound', 'DFO_WDA', 'JASCO_VFPA', 'DFO_CRP', 'SMRUConsulting', 'ONC', 'SIO', 'SIMRES', 'UAF']
PAT = re.compile(r'^(.+?\.(?:wav|flac))_(' + '|'.join(PROVIDERS) + r')_', re.IGNORECASE)


def reconstruct_splits(all_anno: pd.DataFrame, out_dir: Path, eval_csv: Path, birdnet_models_dir: Path) -> None:
    """Re-derives K. Palmer's train/holdout split, writing full_train.csv /
    holdout_eval.csv (same shape as exp_1's own output) plus a uid/fold_0
    splits.csv for model_trainer.dataloader.splits_path."""
    birdnet01_DCLDE_eval = pd.read_csv(eval_csv)

    # --- Eval set: match on the real original filename, not the meaningless trailing index ---
    eval_stems = set(
        birdnet01_DCLDE_eval["File"]
        .apply(lambda f: Path(m.group(1)).stem.lower() if (m := PAT.match(f)) else None)
        .dropna()
    )
    anno_stems = all_anno["Soundfile"].apply(lambda s: Path(s).stem.lower())

    # --- train_stems: files used to train any of the paper's own models, for leakage removal ---
    train_stems = set()
    for i in range(1, 10):
        try:
            original = pd.read_csv(birdnet_models_dir / f"birdnet{i:02d}" / f"birdnet{i:02d}.csv", low_memory=False, usecols=["Soundfile"])
        except FileNotFoundError:
            continue
        train_stems.update(original["Soundfile"].apply(lambda s: Path(s).stem.lower()))

    # --- fold: test = in eval set and not leaked; excluded (NaN) = in eval set but leaked; else train ---
    is_leaked = anno_stems.isin(eval_stems) & anno_stems.isin(train_stems)
    print(f"excluding {anno_stems[is_leaked].nunique()} leaked files ({is_leaked.sum()} rows) from test")

    fold = pd.Series("train", index=all_anno.index)
    fold[anno_stems.isin(eval_stems)] = "test"
    fold[is_leaked] = None
    print(fold.value_counts(dropna=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    all_anno[fold == "train"].to_csv(out_dir / "full_train.csv", index=False)
    all_anno[fold == "test"].to_csv(out_dir / "holdout_eval.csv", index=False)
    pd.DataFrame({"uid": all_anno["uid"], "fold_0": fold}).to_csv(out_dir / "splits.csv", index=False)
