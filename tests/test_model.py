import unittest

import torch

from o1fm.losses import forecast_loss
from o1fm.model import ForecastBatch, O1FMConfig, O1FMForecaster


def make_batch(*, future_target_value: float, future_observed: bool = False) -> ForecastBatch:
  batch = 2
  context = 6
  horizon = 4
  total = context + horizon
  values = torch.zeros(batch, 3, total)
  observed_mask = torch.zeros_like(values, dtype=torch.bool)

  values[:, 0, :context] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  observed_mask[:, 0, :context] = True
  values[:, 0, context:] = future_target_value
  observed_mask[:, 0, context:] = future_observed

  values[:, 1, :context] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
  observed_mask[:, 1, :context] = True

  values[:, 2, :] = torch.arange(total, dtype=torch.float32)
  observed_mask[:, 2, :] = True

  role_ids = torch.tensor([0, 1, 2], dtype=torch.long)
  return ForecastBatch(
    values=values,
    observed_mask=observed_mask,
    role_ids=role_ids,
    horizon=horizon,
  )


class O1FMModelTest(unittest.TestCase):
  def setUp(self) -> None:
    torch.manual_seed(0)
    self.model = O1FMForecaster(
      O1FMConfig(
        patch_len=2,
        d_model=32,
        num_heads=4,
        num_layers=2,
      )
    )

  def test_forward_shapes_and_monotone_quantiles(self) -> None:
    batch = make_batch(future_target_value=0.0, future_observed=False)
    out = self.model(batch)
    self.assertEqual(out.median.shape, (2, 1, 4))
    self.assertEqual(out.quantiles.shape, (2, 1, 4, 9))
    diffs = out.quantiles.diff(dim=-1)
    self.assertTrue(bool((diffs >= -1e-6).all()))
    torch.testing.assert_close(out.median, out.quantiles[..., 4], atol=1e-6, rtol=1e-6)

  def test_future_placeholder_values_cannot_leak(self) -> None:
    clean = make_batch(future_target_value=0.0, future_observed=False)
    poisoned = make_batch(future_target_value=9999.0, future_observed=True)
    clean_out = self.model(clean)
    poisoned_out = self.model(poisoned)
    torch.testing.assert_close(clean_out.median, poisoned_out.median, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(clean_out.quantiles, poisoned_out.quantiles, atol=1e-6, rtol=1e-6)

  def test_nonfinite_input_rejected(self) -> None:
    batch = make_batch(future_target_value=0.0, future_observed=False)
    batch.values[0, 0, 0] = float("nan")
    with self.assertRaisesRegex(ValueError, "finite"):
      self.model(batch)

  def test_loss_backward(self) -> None:
    batch = make_batch(future_target_value=0.0, future_observed=False)
    out = self.model(batch)
    target = torch.tensor(
      [[[7.0, 8.0, 9.0, 10.0]]],
      dtype=out.quantiles.dtype,
      device=out.quantiles.device,
    ).expand_as(out.median)
    loss = forecast_loss(out.quantiles, target, self.model.config.quantiles)
    loss.backward()
    grads = [p.grad for p in self.model.parameters() if p.grad is not None]
    self.assertTrue(grads)
    total = sum(g.abs().sum() for g in grads)
    self.assertTrue(torch.isfinite(total))
    self.assertGreater(total.item(), 0.0)
