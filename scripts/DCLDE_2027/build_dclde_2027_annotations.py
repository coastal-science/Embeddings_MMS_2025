"""Merge DCLDE 2027 root annotations with K. Palmer's per-model call-type tables.

Reproduces the join done in exp_0's ``call_type_aggregation.ipynb``: the root
``Annotations.csv`` inventory carries every annotation, but ``CallType`` only
lives in the BirdNET "parent" CSVs produced by ``CreateParentAnnotations.R``.
This script merges the two back together, then appends algorithmically-selected
"Background" rows (see background_method), into one ``annotations_w_calltype.csv``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from encoder_pipeline.common.mlflow_utils import configure_datasets_mlflow
from reproduce_k_palmer_2026_population_level.generate_splits_files import (
    generate_splits_files,
    stratify_for_split_coverage,
)

DROP_COLS = {"Unnamed: 0"}
MERGE_COLS = ["Soundfile", "FileBeginSec", "FileEndSec", "CallType", "CalltypeCategory", "CalltypeHasQ", "HasQ"]
PARENT_CSV_NAMES = [
    "DCLDE_train_parent_birdnetGrid2.csv",
    "DCLDE_Holdout_parent_birdnetGrid2.csv",
    "Malahat_Holdout_parent_redo_birdnetGrid2.csv",
]
AUDIO_EXTENSIONS = (".wav", ".flac")
BACKGROUND_LABEL = "Background"
DATASET_NAME = "DCLDE_2027"
MLFLOW_EXPERIMENT_NAME = f"Datasets/{DATASET_NAME}"
TEST_N_PER_LABEL = 10


def build_local_audio_index(audio_root: Path) -> dict[str, str]:
    """Map lowercased audio filename stems to their local path under audio_root."""

    return {
        f.stem.lower(): str(f)
        for f in audio_root.rglob("*")
        if f.suffix.lower() in AUDIO_EXTENSIONS
    }


def load_parent_calltypes(raw_data_dir: Path) -> pd.DataFrame:
    """Concat the BirdNET parent CSVs' call-type columns, deduped by event key."""

    frames = []
    for fname in PARENT_CSV_NAMES:
        path = raw_data_dir / fname
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False, usecols=MERGE_COLS))
    parent = pd.concat(frames, ignore_index=True)
    return parent.drop_duplicates(subset=["Soundfile", "FileBeginSec", "FileEndSec"])


def derive_labels(df: pd.DataFrame) -> pd.Series:
    labels = df["ClassSpecies"]
    labels[df["Ecotype"].notna()] = df["Ecotype"][df["Ecotype"].notna()]
    labels[(df["KW"] == 1) & df["Ecotype"].isna()] = "KW_und"
    return labels


def select_naive_background_windows(annotations: pd.DataFrame, window_duration: float, hop_duration: float, event_buffer: float) -> pd.DataFrame:
    """The "naive" background_method: slides a window across every annotated
    recording, keeping only starts that don't overlap a labeled event
    (+/- event_buffer). See notebooks/background_windows.ipynb."""
    valid = annotations[annotations["LocalFileOk"]].copy()

    candidates = []
    for local_path, file_rows in tqdm(valid.groupby("LocalPath"), desc="selecting background windows", unit="file"):
        file_duration = librosa.get_duration(path=local_path)
        padded = list(zip(file_rows["FileBeginSec"] - event_buffer, file_rows["FileEndSec"] + event_buffer))

        for start in np.arange(0, file_duration - window_duration, hop_duration):
            end = start + window_duration
            if any(start < pe and end > ps for ps, pe in padded):
                continue
            candidates.append({
                "Soundfile": file_rows["Soundfile"].iloc[0],
                "LocalPath": local_path,
                "Dataset": file_rows["Dataset"].iloc[0],
                "Provider": file_rows["Provider"].iloc[0],
                "FileBeginSec": float(start),
                "FileEndSec": float(end),
            })
    return pd.DataFrame(candidates)


def to_annotations_schema(candidates: pd.DataFrame, columns: list[str], method: str) -> pd.DataFrame:
    """Expands a background_method's (Soundfile, LocalPath, Dataset, Provider,
    FileBeginSec, FileEndSec) output to match `columns` (the real annotations'
    own column set) so it can be concatenated straight onto it. Columns that
    don't apply to an algorithmically-selected background window (frequency
    band, species/call-type fields, UTC -- see
    notebooks/background_windows.ipynb for why UTC isn't fabricated here) are
    left null. BackgroundMethod records which method produced the row."""
    rows = candidates.copy()
    rows["Duration"] = rows["FileEndSec"] - rows["FileBeginSec"]
    rows["CenterTime"] = (rows["FileBeginSec"] + rows["FileEndSec"]) / 2
    rows["Labels"] = BACKGROUND_LABEL
    rows["BackgroundMethod"] = method
    rows["KW"] = 0
    rows["FileOk"] = True
    rows["LocalFileOk"] = True
    rows["HasQ"] = False
    rows["CalltypeHasQ"] = False
    rows["EcotypeCertain"] = False
    for col in columns:
        if col not in rows.columns:
            rows[col] = np.nan
    return rows[columns]


def select_background_windows(
    annotations: pd.DataFrame, method: str, columns: list[str], window_duration: float, hop_duration: float, event_buffer: float,
) -> pd.DataFrame:
    """Dispatches to the requested background_method, then expands the result
    to annotations' own column schema. "naive" is the only method implemented
    so far -- embedding- and spectral-based selection are planned."""
    if method == "naive":
        candidates = select_naive_background_windows(annotations, window_duration, hop_duration, event_buffer)
    else:
        raise NotImplementedError(f"background_method={method!r} not implemented yet -- only 'naive' exists so far.")
    return to_annotations_schema(candidates, columns, method)


def subsample_by_label(df: pd.DataFrame, n_per_label: int, random_state: int = 0) -> pd.DataFrame:
    """Keeps at most n_per_label rows per Labels value."""
    groups = [group.sample(n=min(n_per_label, len(group)), random_state=random_state) for _, group in df.groupby("Labels")]
    return pd.concat(groups, ignore_index=True)


def build_annotations_with_calltype(
    annotations_path: Path, audio_root: Path, raw_data_dir: Path, background_method: str,
    background_window_duration: float, background_hop_duration: float, background_event_buffer: float,
    test: bool = False, splits_src_dir: Path | None = None,
) -> pd.DataFrame:
    """Join the root annotations with parent-CSV call types and derived
    columns, then append background_method-selected "Background" rows. If
    test, subsamples for fast iteration before (and after) background
    selection: to TEST_N_PER_LABEL rows per (label, split cell) when
    splits_src_dir is given -- so every splits_*.csv keeps every class it
    could have -- else TEST_N_PER_LABEL rows per label."""

    def _test_subsample(df: pd.DataFrame) -> pd.DataFrame:
        if splits_src_dir and splits_src_dir.exists():
            return stratify_for_split_coverage(df, splits_src_dir, TEST_N_PER_LABEL)
        return subsample_by_label(df, TEST_N_PER_LABEL)

    audio_index = build_local_audio_index(audio_root)

    all_anno = pd.read_csv(annotations_path, low_memory=False, index_col=0)
    all_anno = all_anno.drop(columns=[c for c in DROP_COLS if c in all_anno.columns])

    parent = load_parent_calltypes(raw_data_dir)
    all_anno = all_anno.merge(parent, on=["Soundfile", "FileBeginSec", "FileEndSec"], how="left")

    all_anno["LocalPath"] = all_anno["Soundfile"].apply(lambda s: audio_index.get(Path(s).stem.lower()))
    all_anno["LocalFileOk"] = all_anno["LocalPath"].notna()

    all_anno["CenterTime"] = (all_anno["FileBeginSec"] + all_anno["FileEndSec"]) / 2
    all_anno["Duration"] = all_anno["FileEndSec"] - all_anno["FileBeginSec"]
    all_anno["EcotypeCertain"] = all_anno["Ecotype"].isin(["NRKW", "OKW", "SRKW", "TKW", "SAR"])
    all_anno["Labels"] = derive_labels(all_anno)

    missing_ctq = all_anno["CalltypeHasQ"].isna()
    all_anno.loc[missing_ctq, "CalltypeHasQ"] = all_anno.loc[missing_ctq, "CallType"].fillna("").str.contains("?", regex=False)
    all_anno["HasQ"] = all_anno["HasQ"].fillna(False)
    all_anno["BackgroundMethod"] = np.nan  # real annotations weren't algorithmically generated

    if test:
        all_anno = _test_subsample(all_anno)

    background = select_background_windows(
        all_anno, background_method, all_anno.columns.tolist(),
        background_window_duration, background_hop_duration, background_event_buffer,
    )
    if test:
        background = _test_subsample(background)

    result = pd.concat([all_anno, background], ignore_index=True)
    result = result.drop_duplicates(ignore_index=True)  # source Annotations.csv + parent merge emit some exact-dup rows
    result["uid"] = result.index  # row-level id the model_trainer pipeline requires (DatasetConfig.uid_col)
    return result


def run_create_parent_annotations(script_path: Path, output_dir: Path) -> None:
    """Runs CreateParentAnnotations.R, writing its 3 parent call-type CSVs into
    output_dir (via the DCLDE_OUTPUT_DIR env var the script reads)."""
    subprocess.run(["Rscript", str(script_path)], env={**os.environ, "DCLDE_OUTPUT_DIR": str(output_dir)}, check=True)


def git_commit(repo_dir: Path) -> str:
    """Current commit hash of the repo this script lives in."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout.strip()


def check_uncommitted_changes(repo_dir: Path) -> None:
    """Raises if the working tree has uncommitted changes -- dataset generation
    requires a clean tree so git_commit fully captures what actually ran."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout
    if status.strip():
        raise RuntimeError(f"uncommitted changes in {repo_dir} -- commit before running:\n{status}")


def reconstruct_command(script_path: Path, args: argparse.Namespace) -> str:
    """Rebuilds a copy-pasteable command that reproduces this run, using the
    actual resolved argument values -- including flags left at their default,
    not just the ones explicitly passed -- so it's reproducible on its own.
    Flags whose value is None are omitted so argparse's own default (also
    None) is what's restored, rather than passing the literal string "None"."""
    flags = " ".join(
        f"--{key.replace('_', '-')} {value}" for key, value in vars(args).items() if value is not None
    )
    return f"{sys.executable} {script_path} {flags}"


def main() -> None:
    """Builds annotations_w_calltype.csv from the CLI: runs CreateParentAnnotations.R,
    then merges + appends background windows, both written into one timestamped
    run directory and logged to MLflow's shared 'Datasets' experiment."""

    repo_root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dclde-root", type=Path, default=Path("/data/DCLDE_2027/dclde_2027_killer_whales"))
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data_raw" / DATASET_NAME)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument(
        "--background-method", type=str, default="naive", choices=["naive"],
        help="How to select 'Background' windows; only 'naive' is implemented so far.",
    )
    parser.add_argument("--background-window-duration", type=float, default=5.0)
    parser.add_argument("--background-hop-duration", type=float, default=2.5)
    parser.add_argument("--background-event-buffer", type=float, default=10.0)
    parser.add_argument(
        "--test", action="store_true",
        help=f"Subsample to {TEST_N_PER_LABEL} rows per label, for fast iteration.",
    )
    parser.add_argument(
        "--reconstruct-splits", action="store_true",
        help="Also write per-model splits_<name>.csv files into <run_dir>/reconstructed_splits/.",
    )
    parser.add_argument(
        "--reconstructed-splits-dir", type=Path,
        default=script_dir / "reproduce_k_palmer_2026_population_level" / "reconstructed_splits",
        help="Dir with K. Palmer's reconstructed train_*.csv / full_train.csv / holdout_eval.csv, used only when --reconstruct-splits is set.",
    )
    args = parser.parse_args()

    check_uncommitted_changes(script_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S") + ("_test" if args.test else "")
    run_dir = args.data_dir / timestamp
    run_dir.mkdir(parents=True)

    configure_datasets_mlflow(MLFLOW_EXPERIMENT_NAME, args.mlflow_tracking_uri)
    with mlflow.start_run(run_name=run_dir.name):
        mlflow.log_params({
            "dclde_root": str(args.dclde_root),
            "run_dir": str(run_dir),
            "background_method": args.background_method,
            "background_window_duration": args.background_window_duration,
            "background_hop_duration": args.background_hop_duration,
            "background_event_buffer": args.background_event_buffer,
            "test": args.test,
            "git_commit": git_commit(script_dir),
            "command": reconstruct_command(Path(__file__).resolve(), args),
        })

        run_create_parent_annotations(script_dir / "CreateParentAnnotations.R", run_dir)

        annotations_path = args.dclde_root / "Annotations.csv"
        all_anno = build_annotations_with_calltype(
            annotations_path, args.dclde_root, run_dir, args.background_method,
            args.background_window_duration, args.background_hop_duration, args.background_event_buffer,
            args.test, args.reconstructed_splits_dir,
        )

        out_path = run_dir / "annotations.csv"
        all_anno.to_csv(out_path, index=False)
        mlflow.log_param("annotations_path", str(out_path))
        mlflow.log_metric("n_rows", len(all_anno))
        mlflow.log_metric("n_background_rows", int((all_anno["Labels"] == BACKGROUND_LABEL).sum()))
        print(f"Saved {len(all_anno):,} rows, {len(all_anno.columns)} columns -> {out_path}")

        if args.reconstruct_splits:
            splits_dir = run_dir / "reconstructed_splits"
            generate_splits_files(all_anno, splits_dir, args.reconstructed_splits_dir)
            mlflow.log_param("splits_path", str(splits_dir))


if __name__ == "__main__":
    main()
