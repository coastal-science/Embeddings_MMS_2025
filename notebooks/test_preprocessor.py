from pathlib import Path

import h5py

from encoder_pipeline.common.config import load_yaml_config, PipelineConfig
from encoder_pipeline.preprocessor.dataset import Dataset

config = load_yaml_config(Path("../configs/sample.yaml"), PipelineConfig)

dataset = Dataset(config.preprocessor.spectrogram, config.preprocessor.annotation, config.preprocessor.dataset)
# dataset.df = dataset.df[dataset.df["Labels"] == "SRKW"]

# Take just the first 10 unique files' worth of annotations, to exercise
# Dataset.build_hdf5's read-file-once -> spectrogram -> single-writer HDF5
# pipeline without processing the whole df.
# sample_files = dataset.df["LocalPath"].unique() # only 2 in this case
# dataset.df = dataset.df[dataset.df["LocalPath"].isin(sample_files)].copy()

dataset.build_hdf5()

with h5py.File(dataset.out_file, "r") as h5:
    specs = h5["spec"][:]
    uid = h5["uid"].asstr()[:]
    labels = h5["Labels"].asstr()[:]

    # breakpoint()