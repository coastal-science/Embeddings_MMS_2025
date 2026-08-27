import torch

from encoder_pipeline.model_trainer.augment import SpectrogramClassifierAugment, random_time_shift
from encoder_pipeline.model_trainer.config import SpectrogramClassifierAugmentConfig


def test_random_time_shift_rolls_each_sample_circularly_within_bound():
    specs = torch.arange(80.0).reshape(2, 4, 10)
    shifted = random_time_shift(specs.clone(), shift_frac=0.3)

    assert shifted.shape == specs.shape
    for b in range(specs.shape[0]):
        assert torch.allclose(shifted[b].sort(-1).values, specs[b].sort(-1).values)  # circular roll keeps every value
        assert any(torch.equal(torch.roll(specs[b], s, dims=-1), shifted[b]) for s in range(-3, 4))


def test_random_time_shift_is_a_noop_when_frac_rounds_to_zero():
    specs = torch.randn(3, 4, 6)
    assert torch.equal(random_time_shift(specs.clone(), shift_frac=0.1), specs)


def test_classifier_augment_does_not_mutate_its_input():
    specs = torch.randn(3, 4, 20)
    original = specs.clone()
    SpectrogramClassifierAugment(SpectrogramClassifierAugmentConfig(shift_frac=0.5))(specs)

    assert torch.equal(specs, original)
