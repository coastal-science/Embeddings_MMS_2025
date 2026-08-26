"""Compute RMS energy and spectral flatness per (deployment, label) for DCLDE
2027 annotations, including a synthetic "candidate_background" class, and log
each sampled window's mean + std of both to a CSV for downstream analysis --
e.g. is a class's flatness score reflecting real acoustic content, or mostly
which site/equipment recorded it (see e.g. NRKW vs SRKW scoring wildly
differently despite being related killer whale ecotypes).

Operates on annotations_w_calltype.csv (build_dclde_2027_annotations.py's output).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from encoder_pipeline.preprocessor.annotation import AudioFile

CANDIDATE_BACKGROUND_LABEL = "candidate_background"


def compute_stats(clip: np.ndarray) -> tuple[float, float, float, float]:
    """RMS energy and spectral flatness -- mean + std across the clip's STFT
    frames, both derived from one shared STFT magnitude for consistent
    framing. Returns (rms_mean, rms_std, flatness_mean, flatness_std)."""
    S = np.abs(librosa.stft(clip))
    rms = librosa.feature.rms(S=S)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    return rms.mean(), rms.std(), flat.mean(), flat.std()


def sample_labeled_windows(annotations: pd.DataFrame, window_duration: float, n_samples: int, random_state: int) -> pd.DataFrame:
    """Up to n_samples rows per (Dataset, Labels) pair, windowed to
    window_duration around each row's CenterTime."""
    valid = annotations[annotations["LocalFileOk"]].copy()
    valid["FileBeginSec"] = valid["CenterTime"] - window_duration / 2
    valid["FileEndSec"] = valid["CenterTime"] + window_duration / 2
    samples = [
        group.sample(n=min(n_samples, len(group)), random_state=random_state)
        for _, group in valid.groupby(["Dataset", "Labels"])
    ]
    return pd.concat(samples, ignore_index=True)[["LocalPath", "Dataset", "Labels", "FileBeginSec", "FileEndSec"]]


def candidate_background_windows(annotations: pd.DataFrame, window_duration: float, event_buffer: float) -> pd.DataFrame:
    """Every non-overlapping window per recording that doesn't overlap a
    labeled event (+/- event_buffer)."""
    valid = annotations[annotations["LocalFileOk"]]
    candidates = []
    for local_path, file_rows in tqdm(valid.groupby("LocalPath"), desc="carving candidate windows", unit="file"):
        file_duration = librosa.get_duration(path=local_path)
        padded = list(zip(file_rows["FileBeginSec"] - event_buffer, file_rows["FileEndSec"] + event_buffer))
        dataset = file_rows["Dataset"].iloc[0]
        for start in np.arange(0, file_duration - window_duration, window_duration):
            end = start + window_duration
            if any(start < pe and end > ps for ps, pe in padded):
                continue
            candidates.append({
                "LocalPath": local_path, "Dataset": dataset, "Labels": CANDIDATE_BACKGROUND_LABEL,
                "FileBeginSec": float(start), "FileEndSec": float(end),
            })
    return pd.DataFrame(candidates)


def sample_candidate_background_windows(
    annotations: pd.DataFrame, window_duration: float, event_buffer: float, n_samples: int, random_state: int,
) -> pd.DataFrame:
    """Up to n_samples candidate background windows per deployment."""
    candidates = candidate_background_windows(annotations, window_duration, event_buffer)
    samples = [
        group.sample(n=min(n_samples, len(group)), random_state=random_state)
        for _, group in candidates.groupby("Dataset")
    ]
    return pd.concat(samples, ignore_index=True)


STAT_COLUMNS = ["rms_mean", "rms_std", "flatness_mean", "flatness_std"]


def _score_file(local_path: str, rows: list[dict]) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Picklable multiprocessing worker: decodes one file once, scores every
    window in it. Runs in a separate process, so must be a top-level function."""
    audio_file = AudioFile(local_path)
    return [
        (row["_row_index"], compute_stats(audio_file.slice(row["FileBeginSec"], row["FileEndSec"] - row["FileBeginSec"])))
        for row in rows
    ]


def score(windows: pd.DataFrame, max_workers: int | None = None) -> pd.DataFrame:
    """Adds rms_mean/rms_std/flatness_mean/flatness_std columns, one process
    per file in parallel (max_workers=None uses every available core). Skips
    files librosa can't decode."""
    windows = windows.reset_index(drop=True)
    stats = np.full((len(windows), len(STAT_COLUMNS)), np.nan)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_file, local_path, [{"_row_index": idx, **row} for idx, row in zip(rows.index, rows.to_dict("records"))]): len(rows)
            for local_path, rows in windows.groupby("LocalPath")
        }
        with tqdm(total=len(windows), desc="scoring clips", unit="clip") as pbar:
            for future in as_completed(futures):
                try:
                    for row_index, row_stats in future.result():
                        stats[row_index] = row_stats
                except Exception:
                    pass  # file couldn't be decoded -- its rows stay NaN, dropped below
                pbar.update(futures[future])

    windows[STAT_COLUMNS] = stats
    return windows.dropna(subset=STAT_COLUMNS)


def aggregate(per_clip: pd.DataFrame) -> pd.DataFrame:
    """Compresses per-clip rows down to one row per (deployment, label): each
    stat's mean/std across clips, plus how many clips went into it."""
    grouped = per_clip.groupby(["deployment", "label"])
    agg = grouped[STAT_COLUMNS].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_clips"] = grouped.size()
    return agg.reset_index()


def main(
    annotations_path: Path, out_path: Path, aggregated_out_path: Path, window_duration: float, event_buffer: float,
    n_samples: int, random_state: int, max_workers: int | None,
) -> None:
    """Log per-clip RMS/flatness stats, plus a deployment/label aggregate, to CSVs."""

    annotations = pd.read_csv(annotations_path, low_memory=False)

    labeled = sample_labeled_windows(annotations, window_duration, n_samples, random_state)
    background = sample_candidate_background_windows(annotations, window_duration, event_buffer, n_samples, random_state)
    # score both together in one pool -- files common to both sets (e.g. a labeled
    # file that also contributed background candidates) only get decoded once
    per_clip = score(pd.concat([labeled, background], ignore_index=True), max_workers)
    per_clip = per_clip[["Dataset", "Labels", "LocalPath", "FileBeginSec", "FileEndSec", *STAT_COLUMNS]].rename(columns={
        "Dataset": "deployment", "Labels": "label", "LocalPath": "local_path",
        "FileBeginSec": "file_begin_sec", "FileEndSec": "file_end_sec",
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_clip.to_csv(out_path, index=False)
    print(f"Saved {len(per_clip):,} per-clip rows -> {out_path}")

    aggregated = aggregate(per_clip)
    aggregated_out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(aggregated_out_path, index=False)
    print(f"Saved {len(aggregated):,} deployment/label rows -> {aggregated_out_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data_raw" / "DCLDE_2027"

    main(
        annotations_path=data_dir / "annotations_w_calltype.csv",
        out_path=data_dir / "flatness_by_clip.csv",
        aggregated_out_path=data_dir / "flatness_by_deployment.csv",
        window_duration=5.0,
        event_buffer=10.0,
        n_samples=50,
        random_state=0,
        max_workers=None,
    )
