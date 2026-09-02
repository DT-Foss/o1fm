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
          ┌─────────┼──────────┐
     Crystal    structural    foundation
      kernel      baseline       model
          └─────────┼──────────┘
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
| Marginal 10–90% coverage | 57.64% | **79.40%** |
| Joint whole-trajectory coverage | 0/12 | **10/12** |
| Mean interval width | 0.6432 | 1.1640 |
| Mean pinball loss | 0.12314 | **0.12261** |
| Foundation forwards | 6/12 | **6/12** |

The field recovered 21.76 coverage points, finished 0.60 points from the 80%
target, and preserved the 50% foundation-forward reduction. Its finite-sample
`(n+1)` conformal rank, partial-label updates, O1-driven adaptive alpha and
atomic action/route/regime state are all in the live runtime.

The committed receipt is
[`results/o1fm-directed-conformal-v1.json`](results/o1fm-directed-conformal-v1.json),
SHA-256 `b0f62cc3b7a5de5c4494bd9099496bebfe73def2bc621b66ff5bc94335e59c8e`.

An independent `--no-conformal` control reproduced the same six-foundation /
six-Crystal action trace, 57.64% coverage and 0.12314 pinball. Its committed
receipt is
[`results/o1fm-no-conformal-control-v1.json`](results/o1fm-no-conformal-control-v1.json),
SHA-256 `97c9f70831c83d8c006dd009ca1a53cdd3a309c1e68517683da06657b33904bb`.

## Real open-weight teacher result

O1FM now runs unchanged over real Toto-2.0 weights through the common
`ForecastTeacher` contract. The receipt pins the loaded 21,915,584-parameter
checkpoint by its tensor SHA-256 and executes 24 non-overlapping multivariate
windows with past-only and known-future covariates.

| Metric | Raw Toto-22M | Calibrated all-Toto control | Adaptive O1FM |
|---|---:|---:|---:|
| Toto forwards | 24 | 24 | **15** |
| Verified structural executions | 0 | 0 | **9** |
| Marginal coverage | 82.37% | 86.92% | **85.74%** |
| Joint trajectory coverage | 12.5% | 87.5% | **87.5%** |
| MAE | 0.14920 | 0.14920 | 0.15933 |
| Mean pinball | 0.06026 | 0.06471 | 0.06789 |

With an ActionBank pinball tolerance of `0.005`, the control plane removes
37.5% of Toto forwards. Against the identically calibrated all-Toto control,
the adaptive policy changes pinball by `+0.00318`, coverage by `-1.17`
percentage points and MAE by `+0.01014`. The joint envelope remains unchanged.
The 22M teacher itself averages 12.3 ms per direct 3-target × 16-horizon
forecast on the recorded arm64 CPU environment.
The strict gate selects nine seasonal actions and zero ridge Crystals; contextual
teacher-state compilation is the next Crystal upgrade.

The causal graph now appends each rolling history as one validated batch. It
reloads and fsyncs once per forecast instead of once per timestamp, removing the
dominant control-plane latency without changing event hashes or graph edges.

The committed receipt is
[`results/o1fm-toto22m-control-plane-v1.json`](results/o1fm-toto22m-control-plane-v1.json),
SHA-256 `8a3547628d338f391a0884bc75942e67374a2ee675e6b573545dddfaf6daad7f`.
The loaded Toto tensors hash to
`18b094801c589af0e6c9f900bf7f868541ce975712be9ac4a06d4529b647e4b2`.

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
python -m o1fm teacher-demo --teacher toto2
```

O1FM is developed by David Tom Foss.

The full target architecture is documented in
[`docs/ENDGAME.md`](docs/ENDGAME.md).
