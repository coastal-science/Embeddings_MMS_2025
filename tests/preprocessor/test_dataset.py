import pandas as pd

from encoder_pipeline.preprocessor.config import AnnotationConfig, AudioFileConfig, DatasetConfig, SpectrogramConfig
from encoder_pipeline.preprocessor.dataset import Dataset


def _write_csv(path, labels):
    n = len(labels)
    pd.DataFrame({
        "uid": range(n),
        "LocalPath": [f"/audio/f{i}.wav" for i in range(n)],
        "FileBeginSec": [0.0] * n,
        "FileEndSec": [1.0] * n,
        "Duration": [1.0] * n,
        "CenterTime": [0.5] * n,
        "Labels": labels,
    }).to_csv(path, index=False)
    return str(path)


def _dataset(csv, tmp_path, **ds_kwargs):
    return Dataset(
        SpectrogramConfig(), AudioFileConfig(), AnnotationConfig(),
        DatasetConfig(annotations_csv=csv, **ds_kwargs), data_dir=str(tmp_path),
    )


def test_load_annotations_drops_specified_classes(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "SRKW", "Background", "HW", "SAR"])
    df = _dataset(csv, tmp_path, classes_to_drop=["Background", "SAR"])._load_annotations()

    assert sorted(df["Labels"]) == ["HW", "HW", "SRKW"]


def test_load_annotations_keeps_everything_when_classes_to_drop_unset(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "Background"])
    assert len(_dataset(csv, tmp_path)._load_annotations()) == 2


def test_classes_to_drop_changes_the_output_hash(tmp_path):
    csv = _write_csv(tmp_path / "a.csv", ["HW", "Background"])
    plain = _dataset(csv, tmp_path)
    dropped = _dataset(csv, tmp_path, classes_to_drop=["Background"])

    assert plain.content_hash != dropped.content_hash
