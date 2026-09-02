# O1FM

O1FM is a forecasting-native runtime that learns where prediction work belongs.
It combines a multivariate patch forecaster with continuous O1 error state, an
append-only causal graph, reusable Forecast Crystals, and an economic Action
Bank. The foundation model handles genuine novelty. Repeated regimes move into
smaller verified kernels.

```text
multivariate history + known future covariates
                    │
             causal routing graph
                    │
          O1 regime and error state
                    │
       Action Bank chooses forecast work
       ┌────────┬────────┬──────────┐
    Crystal   local FM  structural  foundation
     kernel   authority   baseline     model
       └────────┴────────┴──────────┘
                    │
       directed O1 conformal field
                    │
       invariant + quantile verification
                    │
             observed outcome
                    │
       O1 / graph / Crystal settlement
```

## Core design

- **One-pass probabilistic horizon.** Patch tokens are processed by alternating
  causal temporal attention and cross-variate Sinkhorn attention. The output
  head predicts the full horizon and ordered quantiles in one forward pass.
- **Forecasting-native O1 state.** Every realized target updates normalized
  residuals, coverage, learning progress, surprise, and regime identity.
- **Directed conformal control.** Separate lower/upper reserves and a joint
  trajectory field adapt to O1 surprise without changing the point forecast.
- **Causal covariate routing.** Online lagged influence scores select past-only
  and known-future signals. Topology hysteresis accepts real route changes while
  ignoring one-window score jitter.
- **Forecast Crystals.** Verified regimes charge compact ridge kernels that can
  release future forecasts without another foundation forward.
- **Economic execution.** The Action Bank selects the cheapest action that has
  preserved verified forecast quality in the current graph and regime.
- **Hard output contract.** Forecasts must be finite, shape-correct, and
  monotone across quantiles before they can be returned or learned from.

## First end-to-end result

The initial proof runs the complete system on a deterministic three-regime
multivariate stream. It trains on 60 windows, then settles 12 future windows of
3 targets × 12 horizon steps in temporal order.

| Metric | Direct foundation | O1FM runtime |
|---|---:|---:|
| Foundation forwards | 12 | **6** |
| Forecast Crystal executions | 0 | **6** |
| MAE | 0.3411 | **0.2788** |
| Mean pinball loss | 0.1335 | **0.1231** |

The runtime removed 50% of foundation forwards while improving both point and
probabilistic error on this stream. The live graph accumulated 72 lagged edges,
55 reached routing support, and O1 settled 432 realized target values. This is
the committed architecture proof; public benchmark integration is the next
stage. The exact result is produced by:

```bash
python -m o1fm demo
```

The committed receipt is [`results/o1fm-demo-v1.json`](results/o1fm-demo-v1.json),
SHA-256 `f150251bb0a1dab3cb4512f9803d9cb1a608694967d9778b1f759349800c2c85`.

## Directed calibration result

The next receipt runs the same 12-window sequence through the persistent O1
Conformal Field. Calibration uses only previously settled outcomes; new actions,
routes and regimes inherit a pooled parent field and then accumulate exact local
evidence.

| Metric | Same selected action, raw | Directed O1 field |
|---|---:|---:|
| Marginal 10–90% coverage | 57.64% | **79.86%** |
| Joint whole-trajectory coverage | 0/12 | **11/12** |
| Mean interval width | 0.6432 | 1.3403 |
| Mean pinball loss | 0.12316 | 0.12429 |
| Foundation forwards | 6/12 | **6/12** |

The field recovered 22.22 coverage points, finished 0.14 points from the 80%
target, and preserved the 50% foundation-forward reduction. Its finite-sample
`(n+1)` conformal rank, partial-label updates, O1-driven adaptive alpha and
atomic action/route/regime state are all in the live runtime.

The committed receipt is
[`results/o1fm-directed-conformal-v1.json`](results/o1fm-directed-conformal-v1.json),
SHA-256 `29314f62dc9a8941a647627fa2c9f747d538061ecc33f146b05a178c0423571b`.

An independent `--no-conformal` control reproduced the same six-foundation /
six-Crystal action trace, 57.64% coverage and 0.12316 pinball. Its committed
receipt is
[`results/o1fm-no-conformal-control-v1.json`](results/o1fm-no-conformal-control-v1.json),
SHA-256 `84149dcfc27a0125e123030b915d186e7c3962d13d9fc7f04887395e3faef0e5`.

## Real open-weight teacher result

O1FM now runs a real Apache-2.0 Toto family through the common
`ForecastTeacher` contract: 22M is the foundation authority and 4M is the
continuous local authority. The receipt pins both loaded checkpoints by tensor
SHA-256 and executes 24 non-overlapping multivariate windows with past-only and
known-future covariates.

| Metric | Raw Toto-22M | Calibrated all-Toto control | Adaptive O1FM |
|---|---:|---:|---:|
| Toto-22M forwards | 24 | 24 | **7** |
| Toto-4M forwards | 0 | 0 | 24 |
| Selected local actions | 0 | 0 | **17** |
| 22M-equivalent forwards | 24 | 24 | **11.54** |
| Marginal coverage | 82.37% | 86.92% | **87.55%** |
| Joint trajectory coverage | 12.5% | 87.5% | **91.67%** |
| MAE | 0.14920 | 0.14920 | 0.15235 |
| Mean pinball | 0.06026 | 0.06471 | 0.06612 |

With an ActionBank pinball tolerance of `0.005` and a foundation refresh every
four local actions, the control plane removes 70.83% of 22M forwards. Counting
every 4M shadow/local forward at its parameter ratio still removes 51.92% of
foundation-equivalent work. Against the identically calibrated all-22M control,
pinball changes by `+0.00141`, coverage by `+0.63` percentage points and MAE by
`+0.00315`; joint coverage gains 4.17 points. Raw 22M and 4M forecasts average
13.6 ms and 5.0 ms respectively on the recorded arm64 CPU environment.

The strict gate selects 17 local-model actions and zero ridge Crystals.
Contextual local-teacher states are the next Crystal input upgrade.

The causal graph now appends each rolling history as one validated batch. It
reloads and fsyncs once per forecast instead of once per timestamp, removing the
dominant control-plane latency without changing event hashes or graph edges.

The committed receipt is
[`results/o1fm-toto22m-control-plane-v1.json`](results/o1fm-toto22m-control-plane-v1.json),
SHA-256 `0a0d20a6e2c77e3e533a1260168239ff8f81acb6b929c336e9684449d9e5cbaa`.
The loaded 22M and 4M tensors hash to
`18b094801c589af0e6c9f900bf7f868541ce975712be9ac4a06d4529b647e4b2`
and `41700039ffc479dbc93e79b69ca34adfb7343d15ad170a987537b7c72164508d`.

The same execution writes a live `.causal` Atlas through IMMER's optional
`LiveStore`: 24 forecasts produce 72 directly citable Forecast, Transport and
Settlement segments. Revision 72 verifies both before and after remount; every
record resolves by `[segment_sha256, record_index]`. Atlas records bind route
topology, route `W1`, prediction/calibration identities, O1 settlement and
forward economics without storing raw targets or model weights.

## Relationship to TimesFM

Google's TimesFM-3 established the strongest current reference geometry for
zero-shot multivariate forecasting: 32-step patches, temporal and variate
mixing, known-future covariates, contiguous horizon masking, and nine quantiles.
O1FM is an independent implementation and adds the continuous causal control
plane around that forecasting contract.

The official TimesFM source is Apache-2.0. TimesFM-3 pretrained weights use the
separate `timesfm-non-commercial-license-v1.0` license. O1FM does not redistribute
or require those weights. An optional research adapter may load a user-provided
checkpoint under its own license.

References:

- [TimesFM-3 architecture announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [Official TimesFM repository](https://github.com/google-research/timesfm)
- [Official TimesFM-3 model card](https://huggingface.co/google/timesfm-3.0-pytorch)

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m o1fm demo
```

Run the real Apache-2.0 Toto teacher path:

```bash
python -m pip install -e '.[toto]'
PYTHONPATH=/path/to/immer/src python -m o1fm teacher-demo \
  --teacher toto2 --atlas-dir artifacts/private/forecast-atlas.causal
```

O1FM is developed by David Tom Foss.

The full target architecture is documented in
[`docs/ENDGAME.md`](docs/ENDGAME.md).
