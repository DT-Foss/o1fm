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
SHA-256 `85c49fc48ea4abb70b260a81a7d7b796b7ed720dbbddbd96a35c01f7af520d7b`.

An independent `--no-conformal` control reproduced the same six-foundation /
six-Crystal action trace, 57.64% coverage and 0.12314 pinball. Its committed
receipt is
[`results/o1fm-no-conformal-control-v1.json`](results/o1fm-no-conformal-control-v1.json),
SHA-256 `31c812a498dc2d8dd23263dca9e7e5eb99f5ef6053602ee71c877e444d8d272c`.

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

The full target architecture is documented in
[`docs/ENDGAME.md`](docs/ENDGAME.md).
