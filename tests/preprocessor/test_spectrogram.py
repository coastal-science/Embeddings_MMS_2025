"""Guards STFTConfig against drift from scipy.signal.ShortTimeFFT's signature."""

import inspect

from scipy.signal import ShortTimeFFT

from encoder_pipeline.preprocessor.spectrogram import STFTConfig


def test_stft_config_fields_match_short_time_fft_params():
    stft_params = set(inspect.signature(ShortTimeFFT.__init__).parameters) - {"self"}
    config_fields = set(STFTConfig.model_fields)

    missing = config_fields - stft_params
    assert not missing, f"STFTConfig fields not present on ShortTimeFFT: {missing}"
