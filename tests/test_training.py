from __future__ import annotations

import torch

from o1fm.metrics import score_prediction
from o1fm.model import O1FMConfig, O1FMForecaster
from o1fm.synthetic import SyntheticConfig, generate_synthetic_dataset
from o1fm.training import model_identity, predict, train_forecaster


def test_training_reduces_loss_and_returns_the_strict_prediction_contract() -> None:
    data = generate_synthetic_dataset(
        SyntheticConfig(
            total_steps=160,
            history_length=32,
            horizon=8,
            train_stride=8,
            online_stride=8,
            missing_rate=0.0,
        )
    )
    torch.manual_seed(11)
    model = O1FMForecaster(
        O1FMConfig(
            patch_len=4,
            d_model=24,
            num_heads=4,
            num_layers=1,
        )
    )
    before = model_identity(model)
    report = train_forecaster(
        model,
        data.train,
        epochs=8,
        batch_size=8,
        learning_rate=1e-3,
        seed=11,
    )
    after = model_identity(model)

    assert report.final_loss < report.first_loss
    assert before != after
    example = data.online[0]
    prediction = predict(model, example.inputs)
    prediction.validate_for(example.inputs)
    assert score_prediction(prediction, example.require_target()).mae >= 0.0
