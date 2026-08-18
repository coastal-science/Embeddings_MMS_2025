import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from pydantic import BaseModel

from encoder_pipeline.common.paths import get_or_create_hashed_file
from encoder_pipeline.preprocessor.annotation import AudioFile, Annotation, AnnotationConfig
from encoder_pipeline.preprocessor.spectrogram import Spectrogram, SpectrogramConfig


class DatasetConfig(BaseModel):
    """Where Dataset reads annotations from, and how/where parallel
    Dataset.build_hdf5 writes its output."""

    annotations_csv: str
    out_dir: str = "data/preprocessor"
    """Filename within out_dir. Auto-timestamped if unset."""
    max_workers: Optional[int] = None
    """None or a positive int, same as ProcessPoolExecutor; also accepts
    joblib-style negative values (-1 = all cores, -2 = all but one, ...)."""
    metadata_columns: Optional[list[str]] = None # specify metadata columns which should be included.
    local_file_col: Optional[str] = "LocalPath" # column in annotations csv where local audio file paths are specified
    uid_col: Optional[str] = "uid" # column in annotations csv whuich specifies the row-level uid
    def resolve_max_workers(self) -> Optional[int]:
        if self.max_workers is None or self.max_workers > 0:
            return self.max_workers
        return max(1, (os.cpu_count() or 1) + self.max_workers + 1)


class Dataset:
    """Ties a DataFrame of annotation rows to a SpectrogramConfig and drives
    the read-file-once -> spectrogram -> single-writer HDF5 pipeline."""

    def __init__(
        self, spec_config: SpectrogramConfig, annotation_config: AnnotationConfig, dataset_config: DatasetConfig,
        run_name: Optional[str] = None,
    ) -> None:
        self.spec_config = spec_config
        self.annotation_config = annotation_config
        self.dataset_config = dataset_config
        self.run_name = run_name
        self.out_file, self.content_hash = get_or_create_hashed_file(self.dataset_config.out_dir, ".h5", {
                    "spectrogram": self.spec_config.model_dump(),
                    "annotation": self.annotation_config.model_dump(),
                    "annotations_csv": self.dataset_config.annotations_csv,
                    "run_name": self.run_name,
        })
        self.is_materialized = Path(self.out_file).exists()

    def _load_annotations(self) -> pd.DataFrame:
        """Reads annotations_csv and coerces the numeric columns Dataset
        relies on. computed once here (not per-row in a worker) since split
        assignment needs every uid up front, before dispatching to workers."""
        df = pd.read_csv(self.dataset_config.annotations_csv)
        for col in ["FileBeginSec", "FileEndSec", "Duration", "CenterTime"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # check required columns
        if self.dataset_config.uid_col not in df.columns:
            raise ValueError(f"specified uid_col: {self.dataset_config.uid_col} not in {self.dataset_config.annotations_csv}")
        if self.dataset_config.local_file_col not in df.columns:
            raise ValueError(f"specified local_file_col: {self.dataset_config.local_file_col} not in {self.dataset_config.annotations_csv}")
        return df

    @staticmethod
    def _process_file(
        file_path: str, rows: list[dict], spec_config: SpectrogramConfig, annotation_config: AnnotationConfig,
    ) -> list[dict]:
        """helper for processing file (for parralelization of reading files / generating specs.)"""
        audio_file = AudioFile(file_path, annotation_config.resample_sr)
        spec = Spectrogram(spec_config)
        results = []
        for row in rows:
            annotation = Annotation(audio_file, row["Labels"], row["FileBeginSec"], row["Duration"], annotation_config)
            raw = spec.compute_mel(annotation) if spec_config.freq_scale == "mel" else spec.compute_magnitude(annotation)
            # append all cols with the specified row
            results.append({**row, "spec": spec.apply_dynamic(annotation, raw)})
        return results

    def build_hdf5(self, force_rebuild: bool = False) -> str:
        """Groups self.df by local_file_col so each file is decoded once (in
        a worker process), then writes every resulting spectrogram into one
        pre-allocated (n_rows, *sample_shape) dataset -- row order fixed up
        front via _row_index, since as_completed yields workers in whatever
        order they finish -- plus a parallel array per metadata column, to
        self.dataset_config's resolved output path, from this process only,
        since h5py isn't safe for concurrent writers to one file.

        sample_shape itself is learned from the first result rather than
        derived from spec_config/annotation_config, so it stays correct even
        if that math ever changes -- relies on every row producing the same
        shape, which resample_sr + a fixed annotation window guarantee."""
        # exit early if already materialized
        if self.is_materialized and not force_rebuild:
            return
        resolved = self.out_file
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        df = self._load_annotations().reset_index(drop=True)
        grouped = df.groupby(self.dataset_config.local_file_col)
        metadata_columns = self.dataset_config.metadata_columns or list(df.columns)
        metadata: dict[str, list] = {col: [None] * len(df) for col in metadata_columns}
        with h5py.File(resolved, "w") as h5, ProcessPoolExecutor(max_workers=self.dataset_config.resolve_max_workers()) as pool:
            futures = {
                pool.submit(
                    Dataset._process_file,
                    file_path,
                    [{"_row_index": idx, **row} for idx, row in zip(rows.index, rows.to_dict("records"))],
                    self.spec_config, self.annotation_config,
                ): file_path
                for file_path, rows in grouped
            }
            specs = None
            for future in as_completed(futures):
                for result in future.result():
                    # create hdf5 dataset to fill in with the rest of the spectrogram data
                    if specs is None:
                        specs = h5.create_dataset("spec", shape=(len(df), *result["spec"].shape), dtype=result["spec"].dtype)
                    # insert row in correct index (as_completed(futures) may not be in same order as df)
                    row_index = result["_row_index"]
                    specs[row_index] = result["spec"]
                    for col in metadata_columns:
                        metadata[col][row_index] = result[col]

            for col, values in metadata.items():
                if pd.api.types.is_numeric_dtype(df[col]):
                    h5.create_dataset(col, data=values)
                else:
                    h5.create_dataset(col, data=[str(v) for v in values], dtype=h5py.string_dtype())
        self.is_materialized = True
