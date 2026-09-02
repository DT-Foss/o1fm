# Changelog

## Unreleased

- Created O1FM as a forecasting-native sibling of IMMER.
- Defined the one-pass multivariate quantile backbone and continuous causal
  control-plane architecture.
- Recorded official TimesFM-3 source and weight-license boundaries.
- Implemented strict forecasting inputs, targets, predictions, masks, and
  deterministic multivariate regime data.
- Implemented the compact PyTorch patch backbone with RevIN, causal temporal
  attention, Sinkhorn variate transport, and a monotone nine-quantile head.
- Implemented persistent O1 residual state, CUSUM regimes, an append-only
  lagged causal graph, Forecast Crystals, and the economic Action Bank.
- Added online graph-route hysteresis with topology-complete identities and
  immediate score refresh for stable routes.
- Made partial horizon settlements train Forecast Crystals while calibration
  and error totals count only observed values.
- Added the end-to-end CLI proof: 6 of 12 foundation forwards replaced by
  Forecast Crystals, MAE 0.3411 → 0.2788, pinball 0.1335 → 0.1231.
