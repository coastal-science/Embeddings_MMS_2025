"""librosa-backed equivalent of spectrogram.py, for side-by-side comparison.

Same STFTConfig/Spectrogram split, same idea (config mirrors the underlying
transform's constructor field-for-field, Spectrogram ties an Annotation to
it), just backed by librosa.stft instead of scipy.signal.ShortTimeFFT.
"""

from typing import Literal, Optional

import librosa
import numpy as np
from pydantic import BaseModel

from encoder_pipeline.preprocessor.annotation import Annotation

# Features to select from:
# spectrogram kinds - linear, mel
#   for linear spec, frequency axis - linear / log scaling
# magnitude scales - power, amplitude
# dynamic scale - db, pcen



class STFTConfig(BaseModel):
    """Mirrors librosa.stft's constructor field-for-field."""

    n_fft: int = 2048
    """Number of samples i.e. FFT length; also the window length if win_length is None."""

    hop_length: Optional[int] = None
    """How many samples to move the window by i.e. samples between successive frames. Defaults to n_fft // 4 if None."""

    win_length: Optional[int] = None
    """Window length in samples. Defaults to n_fft if None."""

    window: str = "hann"
    """Window function name -- see scipy.signal.get_window for accepted specs."""

    center: bool = True
    """If True, zero-pads by n_fft // 2 on each side so frame t is centered
    at y[t * hop_length]. If False, frame t starts at y[t * hop_length] with
    no padding."""

    pad_mode: str = "constant"
    """Padding mode used when center=True; 'constant' zero-pads."""

    def stft_kwargs(self) -> dict:
        return self.model_dump()


class MagConfig(BaseModel):
    """Linear-frequency spectrogram, no mel filterbank."""

    power: float = 1.0
    """1.0 for amplitude, 2.0 for power."""


class MelConfig(BaseModel):

    """Mirrors librosa.filters.mel's constructor field-for-field, minus sr and
    n_fft -- those come from the Annotation/SpectrogramConfig already in use
    (via compute_mel), rather than being duplicated here where they could
    drift out of sync."""

    n_mels: int = 128
    """Number of mel bands to generate. """

    fmin: float = 0.0
    """Lowest frequency, in Hz."""

    fmax: Optional[float] = None
    """Highest frequency, in Hz. Defaults to sr / 2 if None."""

    htk: bool = False
    """Use the HTK formula instead of Slaney's."""

    norm: Optional[str] = "slaney"
    """Filter normalization; 'slaney' divides each filter by its bandwidth."""

    power: float = 2.0
    """1.0 for amplitude, 2.0 for power."""

    def mel_kwargs(self) -> dict:
        return self.model_dump(exclude={"power"})


class DbConfig(BaseModel):
    """Mirrors librosa.power_to_db / amplitude_to_db's constructor field-for-field."""

    ref: float = 1.0
    amin: float = 1e-10
    top_db: Optional[float] = 80.0


class PcenConfig(BaseModel):
    """Mirrors librosa.pcen's constructor field-for-field."""

    gain: float = 0.98
    bias: float = 2.0
    power: float = 0.5
    time_constant: float = 0.4
    eps: float = 1e-6


class SpectrogramConfig(BaseModel):
    """Overarching, YAML-shaped config: one STFTConfig, plus an explicit
    choice of frequency scale (mag/mel) and dynamic-range compression
    (db/pcen). The `freq_scale`/`dynamic` fields say which of the two
    sibling sub-configs is active -- no type inference needed, since each
    sub-config's model is already known from its field name."""

    stft: STFTConfig = STFTConfig()

    freq_scale: Literal["mag", "mel"] = "mag"
    mag: MagConfig = MagConfig()
    mel: MelConfig = MelConfig()

    dynamic: Literal["db", "pcen"] = "db"
    db: DbConfig = DbConfig()
    pcen: PcenConfig = PcenConfig()


class Spectrogram:
    """Ties an Annotation (which audio) to a SpectrogramConfig (how to transform it).

    compute() is the single source of truth (the raw complex STFT).
    compute_magnitude() derives a real-valued magnitude/power view straight
    from it via librosa.magphase (no second STFT). compute_mel() derives a
    mel-scaled version instead -- from its own re-derived STFT rather than
    compute()'s, see its docstring -- so linear and mel are two views of the
    same underlying transform, each real-valued via the same power convention.
    """

    def __init__(self, config: SpectrogramConfig):
        self.config = config


    def compute_magnitude(self, annotation: Annotation) -> np.ndarray:
        """Real-valued magnitude (config.mag.power=1) or power
        (config.mag.power=2) spectrogram, derived from compute()'s complex
        STFT via librosa.magphase -- librosa's own complex -> (magnitude,
        phase) split -- rather than a hand-rolled np.abs(S) ** power.
        Discards the phase half of magphase's return; use compute() directly
        if phase is needed too."""
        raw_stft = librosa.stft(annotation.audio, **self.config.stft.stft_kwargs())
        magnitude, _phase = librosa.magphase(raw_stft, power=self.config.mag.power)
        return magnitude

    def compute_mel(self, annotation: Annotation) -> np.ndarray:
        """Computes the mel spectrogram"""
        return librosa.feature.melspectrogram(
            y=annotation.audio, sr=annotation.sr, power=self.config.mel.power,
            **self.config.stft.stft_kwargs(), **self.config.mel.mel_kwargs(),
        )

    def apply_dynamic(self, annotation: Annotation, spec: np.ndarray) -> np.ndarray:
        """Applies whichever dynamic-range compression config.dynamic
        selects to an already-computed spectrogram.

        db: librosa.power_to_db or amplitude_to_db, whichever matches the
        active freq_scale's power convention (config.mag.power /
        config.mel.power: 1.0 = amplitude, 2.0 = power).

        pcen: librosa.pcen (Per-Channel Energy Normalization). Expects a
        power spectrogram (config.mel.power=2.0) -- PCEN's own AGC assumes
        energy, not amplitude.
        """
        if self.config.dynamic == "pcen":
            pcen_config = self.config.pcen
            hop_length = self.config.stft.hop_length or self.config.stft.n_fft // 4
            return librosa.pcen(
                spec,
                sr=annotation.sr,
                hop_length=hop_length,
                gain=pcen_config.gain,
                bias=pcen_config.bias,
                power=pcen_config.power,
                time_constant=pcen_config.time_constant,
                eps=pcen_config.eps,
            )
        elif self.config.dynamic == "db":
            db_config = self.config.db
            active_power = self.config.mel.power if self.config.freq_scale == "mel" else self.config.mag.power
            to_db = librosa.power_to_db if active_power == 2.0 else librosa.amplitude_to_db
            return to_db(spec, ref=db_config.ref, amin=db_config.amin, top_db=db_config.top_db)
        else:
            raise ValueError(f"self.config.dynamic must either be 'db' or 'pcen', {self.config.dynamic} was passed")