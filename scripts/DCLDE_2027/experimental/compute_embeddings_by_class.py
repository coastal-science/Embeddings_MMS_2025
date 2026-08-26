"""Compute Perch 2.0 (perch_hoplite) embeddings per (deployment, label) window for
DCLDE 2027 annotations, including a synthetic "candidate_background" class -- the
embedding-based analog of compute_flatness_by_class.py, to check whether a frozen
encoder separates background from real calls better than spectral flatness does.

For now this uses perch_hoplite's own Perch 2.0 model at its default config (5s
window, 32kHz, hop=5s, target_peak=0.25) rather than encoder_pipeline's
SpectrogramConfig/AudioFileConfig plumbing. GPU vs CPU is chosen automatically by
perch_hoplite based on GPU visibility (see _ensure_gpu_ld_library_path below).

Window sampling (window_duration/event_buffer/n_samples/random_state) is kept
identical to compute_flatness_by_class.py's defaults on purpose -- with the same
seed, sample_labeled_windows/sample_candidate_background_windows draw the exact
same clips, so this script's per-clip output is directly joinable against
flatness_by_clip.csv on (deployment, label, local_path, file_begin_sec, file_end_sec).

Operates on annotations_w_calltype.csv (build_dclde_2027_annotations.py's output).
Embeddings are saved as .npy, not CSV columns -- a 1536-dim vector per row doesn't
belong in text-formatted CSV columns; the metadata CSVs are row-aligned with them.
"""

from __future__ import annotations

import glob
import os
import sys


def _ensure_gpu_ld_library_path() -> None:
    """Re-execs this process with the pip-installed nvidia/*/lib dirs on
    LD_LIBRARY_PATH. TF's dynamic linker only resolves libcudart/libcudnn/etc.
    against LD_LIBRARY_PATH as it was at interpreter startup, so setting
    os.environ after tensorflow is imported (or even just before, without a
    re-exec) has no effect -- this must run first, before any perch_hoplite/TF
    import below."""
    site_packages = next(p for p in sys.path if p.endswith("site-packages"))
    lib_dirs = glob.glob(os.path.join(site_packages, "nvidia", "*", "lib"))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if not lib_dirs or all(d in current for d in lib_dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_gpu_ld_library_path()

from pathlib import Path

import numpy as np
import pandas as pd
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo.zoo_interface import EmbeddingModel as PerchModel
from tqdm import tqdm

from compute_flatness_by_class import sample_candidate_background_windows, sample_labeled_windows
from encoder_pipeline.preprocessor.annotation import AudioFile


def load_clips(windows: pd.DataFrame, sample_rate: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Decodes each file once, slicing all its sampled windows from it. Drops
    windows whose slice falls short of a full window (near a file's edge)."""
    clips, keep = [], []
    for local_path, rows in tqdm(windows.groupby("LocalPath"), desc="decoding clips", unit="file"):
        audio_file = AudioFile(local_path, resample_sr=sample_rate)
        for idx, row in rows.iterrows():
            duration = row["FileEndSec"] - row["FileBeginSec"]
            clip = audio_file.slice(row["FileBeginSec"], duration)
            if len(clip) == int(duration * sample_rate):
                clips.append(clip)
                keep.append(idx)
    kept = windows.loc[keep].reset_index(drop=True)
    return kept, np.stack(clips)


def embed(clips: np.ndarray, model: PerchModel, batch_size: int) -> np.ndarray:
    """One pooled (mean over time + channel) embedding vector per clip, batched through model."""
    pooled = [
        model.batch_embed(clips[start:start + batch_size]).pooled_embeddings("mean", "mean")
        for start in tqdm(range(0, len(clips), batch_size), desc="embedding clips", unit="batch")
    ]
    return np.concatenate(pooled).astype(np.float32)


def aggregate(windows: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean embedding per (deployment, label) -- the embedding analog of
    compute_flatness_by_class.aggregate's mean/std table."""
    df = windows.rename(columns={"Dataset": "deployment", "Labels": "label"})
    rows, centroids = [], []
    for (deployment, label), group in df.groupby(["deployment", "label"]):
        rows.append({"deployment": deployment, "label": label, "n_clips": len(group)})
        centroids.append(embeddings[group.index].mean(axis=0))
    return pd.DataFrame(rows), np.stack(centroids).astype(np.float32)


def main(
    annotations_path: Path, out_dir: Path, window_duration: float, event_buffer: float,
    n_samples: int, random_state: int, batch_size: int,
) -> None:
    """Embed sampled labeled + candidate-background windows with Perch 2.0, save
    per-clip and per-(deployment,label) outputs."""

    annotations = pd.read_csv(annotations_path, low_memory=False)

    labeled = sample_labeled_windows(annotations, window_duration, n_samples, random_state)
    background = sample_candidate_background_windows(annotations, window_duration, event_buffer, n_samples, random_state)
    windows = pd.concat([labeled, background], ignore_index=True)

    model = model_configs.load_model_by_name("perch_v2")
    print(f"Loaded {type(model).__name__} ({model.tfhub_path}), embedding_dim inferred at runtime")
    windows, clips = load_clips(windows, model.sample_rate)
    embeddings = embed(clips, model, batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = windows.rename(columns={
        "Dataset": "deployment", "Labels": "label", "LocalPath": "local_path",
        "FileBeginSec": "file_begin_sec", "FileEndSec": "file_end_sec",
    })
    metadata.to_csv(out_dir / "embeddings_by_clip_metadata.csv", index=False)
    np.save(out_dir / "embeddings_by_clip.npy", embeddings)
    print(f"Saved {len(metadata):,} clip embeddings ({embeddings.shape[1]}-d) -> {out_dir / 'embeddings_by_clip.npy'}")

    centroid_metadata, centroids = aggregate(windows, embeddings)
    centroid_metadata.to_csv(out_dir / "embeddings_by_deployment_metadata.csv", index=False)
    np.save(out_dir / "embeddings_by_deployment.npy", centroids)
    print(f"Saved {len(centroid_metadata):,} deployment/label centroids -> {out_dir / 'embeddings_by_deployment.npy'}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data_raw" / "DCLDE_2027"

    main(
        annotations_path=data_dir / "annotations_w_calltype.csv",
        out_dir=data_dir,
        window_duration=5.0,
        event_buffer=10.0,
        n_samples=50,
        random_state=0,
        batch_size=16,
    )
