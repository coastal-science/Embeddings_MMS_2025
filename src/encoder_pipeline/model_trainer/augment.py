import torch

from encoder_pipeline.model_trainer.config import SpectrogramClassifierAugmentConfig, SpectrogramSSLAugmentConfig


def random_time_shift(specs: torch.Tensor, shift_frac: float) -> torch.Tensor:
    """Circularly rolls each sample of a (B, F, T) batch along the time axis
    by a random offset up to shift_frac * T, independently per sample.
    Mutates specs in place."""
    max_shift = int(specs.shape[-1] * shift_frac)
    if max_shift == 0:
        return specs
    for i in range(specs.shape[0]):
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        specs[i] = torch.roll(specs[i], shifts=shift, dims=-1)
    return specs


class SpectrogramClassifierAugment:
    """Train-time augmentation pipeline for (B, F, T) spectrogram batches fed
    to a classifier: random circular time-window shift, independently per
    sample."""

    def __init__(self, config: SpectrogramClassifierAugmentConfig) -> None:
        self.config = config

    def __call__(self, specs: torch.Tensor) -> torch.Tensor:
        if self.config.shift_frac == 0:
            return specs
        return random_time_shift(specs.clone(), self.config.shift_frac)


class SpectrogramSSLAugment:
    """SimCLR-style view augmentation for (B, F, T) spectrogram batches:
    random time mask, random frequency mask, random circular time shift,
    and additive Gaussian noise -- each independently per sample. Call
    twice on the same batch to get two independent views."""

    def __init__(self, config: SpectrogramSSLAugmentConfig) -> None:
        self.config = config

    def __call__(self, specs: torch.Tensor) -> torch.Tensor:
        specs = specs.clone()
        specs = self._mask_axis(specs, axis=2, frac=self.config.time_mask_frac)
        specs = self._mask_axis(specs, axis=1, frac=self.config.freq_mask_frac)
        specs = random_time_shift(specs, self.config.shift_frac)
        specs = specs + torch.randn_like(specs) * self.config.noise_std
        return specs

    def _mask_axis(self, specs: torch.Tensor, axis: int, frac: float) -> torch.Tensor:
        """Zeroes a random contiguous span along axis, sized frac * that
        axis's length, independently for each sample in the batch."""
        length = specs.shape[axis]
        span = int(length * frac)
        if span == 0:
            return specs
        for i in range(specs.shape[0]):
            start = torch.randint(0, length - span + 1, (1,)).item()
            index = [slice(None)] * specs.dim()
            index[0] = i
            index[axis] = slice(start, start + span)
            specs[tuple(index)] = 0.0
        return specs
