import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from encoder_pipeline.model_trainer.config import ModelTrainerConfig, MoCoConfig, MoCoV3Config
from encoder_pipeline.model_trainer.models import MoCoModel, MoCoV3Model
from encoder_pipeline.model_trainer.train import MoCoTrainer, MoCoV3Trainer


@pytest.fixture
def moco_config():
    return MoCoConfig(
        device="cpu", memory_bank_size=32, projection_hidden_dim=32, projection_out_dim=16, epochs=1,
    )


def test_moco_model_momentum_encoder_is_frozen_and_forward_shapes_match(moco_config):
    model = MoCoModel(moco_config)

    assert all(not p.requires_grad for p in model.backbone_momentum.parameters())
    assert all(not p.requires_grad for p in model.projection_head_momentum.parameters())
    assert all(p.requires_grad for p in model.backbone.parameters())

    x = torch.randn(4, 1, 12, 20)
    query, key = model(x), model.forward_momentum(x)
    assert query.shape == key.shape == (4, moco_config.projection_out_dim)
    assert key.grad_fn is None  # keys never carry gradient


def test_moco_trainer_epoch_updates_query_by_grad_and_momentum_by_ema(moco_config):
    trainer = MoCoTrainer(moco_config)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 12, 20), torch.zeros(8, dtype=torch.long)), batch_size=4,
    )

    query_before = next(trainer.model.backbone.parameters()).clone()
    mom_before = next(trainer.model.backbone_momentum.parameters()).clone()
    loss = trainer._run_epoch(loader, train=True)

    assert loss > 0
    assert not torch.equal(query_before, next(trainer.model.backbone.parameters()))
    assert not torch.equal(mom_before, next(trainer.model.backbone_momentum.parameters()))


def test_moco_paradigm_requires_a_moco_config():
    config = ModelTrainerConfig(paradigm="moco")
    assert config.moco is None  # asserted at train_model time, not construction

    config = ModelTrainerConfig(paradigm="moco", moco=MoCoConfig(device="cpu"))
    assert config.moco.momentum == 0.999


@pytest.fixture
def moco_v3_config():
    return MoCoV3Config(
        device="cpu", projection_hidden_dim=64, projection_out_dim=32, prediction_hidden_dim=64, epochs=2,
    )


def test_moco_v3_model_has_predict_head_only_on_query_side(moco_v3_config):
    model = MoCoV3Model(moco_v3_config)

    assert not hasattr(model, "prediction_head_momentum")  # keys skip the prediction head
    assert all(not p.requires_grad for p in model.backbone_momentum.parameters())
    assert all(not p.requires_grad for p in model.projection_head_momentum.parameters())

    x = torch.randn(4, 1, 12, 20)
    query, key = model(x), model.forward_momentum(x)
    assert query.shape == key.shape == (4, moco_v3_config.projection_out_dim)
    assert key.grad_fn is None


def test_moco_v3_trainer_cosine_anneals_momentum_from_base_to_one(moco_v3_config):
    trainer = MoCoV3Trainer(moco_v3_config)
    loader = DataLoader(
        TensorDataset(torch.randn(12, 12, 20), torch.zeros(12, dtype=torch.long)), batch_size=4,
    )

    mom_before = next(trainer.model.backbone_momentum.parameters()).clone()
    trainer._run_epoch(loader, train=True)

    assert trainer._total_steps == moco_v3_config.epochs * len(loader)
    assert trainer._step == len(loader)  # advanced only on the train pass
    assert not torch.equal(mom_before, next(trainer.model.backbone_momentum.parameters()))

    trainer._run_epoch(loader, train=False)
    assert trainer._step == len(loader)  # val pass does not touch the schedule
