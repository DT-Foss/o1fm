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
- Added the persistent Directed O1 Conformal Field: asymmetric lower/upper CQR
  reserves, joint trajectory envelopes, pooled cold-profile transfer,
  finite-sample ranks and O1 surprise/CUSUM adaptation.
- Integrated calibration into every runtime action and ActionBank shadow score.
  The live receipt raises coverage 57.64% → 79.86%, reaches 11/12 joint path
  coverage, and preserves the 50% forward reduction.
- Added the end-to-end CLI proof: 6 of 12 foundation forwards replaced by
  Forecast Crystals, MAE 0.3411 → 0.2788, pinball 0.1335 → 0.1231.
- Added lazy native/Toto-2/Chronos-2 teacher adapters and removed the runtime's
  hard dependency on the native PyTorch forecaster.
- Added a real Toto-2.0 multiscale receipt pinned to both loaded tensor hashes:
  4M local authority plus periodic 22M refreshes remove 70.83% of large-model
  forwards and 51.92% of parameter-equivalent work while improving coverage.
- Added `local` as a priced ActionBank action and periodic foundation refreshes
  that bound local streaks and renew authority under shift.
- Implemented the exact Foss causal-transport identity and stack geodesic check,
  then wired normalized per-target route `W1` into Crystal features and result
  receipts without fragmenting topology identities.
- Added the optional `.causal` Forecast Atlas over IMMER LiveStore. The real
  Toto run commits and remount-verifies 72 directly citable forecast,
  transport and settlement segments without raw targets or weights.
- Batched causal-graph history append into one reload and directory sync per
  forecast while preserving byte-identical event revisions and graph edges.
- Added explicit Crystal fitting modes: native O1FM learns realized local
  dynamics, while external teachers compile their raw trajectories; both keep
  realized targets as residual, coverage and promotion authority.
