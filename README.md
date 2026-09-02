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

O1FM is developed by David Tom Foss.
