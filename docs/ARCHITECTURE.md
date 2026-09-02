# O1FM Architecture

## Product contract

O1FM forecasts one or more target series from three signal classes:

1. target histories;
2. covariates known only in the past;
3. dynamic covariates known across both context and forecast horizon.

One request returns a point forecast and ordered quantiles for every target and
horizon step. A later settlement supplies the realized targets. Settlement is
the only source of forecast-quality authority.

## Execution path

```text
ForecastBatch
  -> CausalGraph.route(regime, horizon)
  -> O1State.regime
  -> ActionBank.recommend
       -> ForecastCrystal.predict       # zero foundation forwards
       -> StructuralBaseline.predict    # zero foundation forwards
       -> O1FMModel.forward             # novelty authority
  -> ForecastVerifier
  -> ForecastTicket

ForecastTicket + realized future
  -> proper scoring / residual vector / coverage
  -> O1State.observe
  -> CausalGraph.observe
  -> ForecastCrystalBank.observe
  -> ActionBank.settle
```

Every learned object is scoped by model identity, feature schema, horizon,
causal-graph revision, and O1 regime generation.

## Forecast backbone

Let `x[b,v,n,p]` be a batch of `p`-step patches for `v` target/covariate
channels. Each channel is normalized from observed context only. Values and
masks are projected into patch tokens:

```text
h_0 = ResidualMLP([RevIN(x), observed_mask, variate_role])
```

Each block alternates two axes:

```text
h'  = h + TemporalAttention(RMSNorm(h), causal=True)
h'' = h' + SinkhornVariateAttention(RMSNorm(h'))
h+  = h'' + SwiGLU(RMSNorm(h''))
```

Temporal attention never reads future target values. Cross-variate attention
operates only inside one aligned patch position, where known-future covariates
are legal. Sinkhorn normalization creates a bounded doubly-stochastic transport
matrix across variates instead of an unconstrained softmax winner.

Future target and past-only patches are masked placeholders. Known-future
covariate patches remain visible. The full horizon is decoded in one forward.

The head emits a median and positive quantile spacings. For lower/upper indices
around the median `m`:

```text
q[m-k] = median - cumulative_softplus(lower_gap[k])
q[m+k] = median + cumulative_softplus(upper_gap[k])
```

Quantile crossing is impossible by construction.

## O1 forecast state

For realized target `y`, median forecast `ŷ`, and predicted interval scale
`s`, O1 observes:

```text
z_t       = |y_t - ŷ_t| / (s_t + eps)
surprise  = max(0, z_t - EWMA(z))
CUSUM_t   = max(0, decay * CUSUM_{t-1} + surprise - drift)
```

The stream also tracks pinball loss, interval coverage, bias, and learning
progress. Crossing the CUSUM boundary increments the regime generation and
changes the keys used by Graph, Crystals, and ActionBank.

## Causal routing graph

For each candidate covariate `c`, target `y`, and lag `l`, the graph maintains
bounded exponentially weighted moments and scores:

```text
edge(c -> y, l) = |EWCov(Δc[t-l], Δy[t])| /
                  sqrt(EWVar(Δc[t-l]) * EWVar(Δy[t]) + eps)
```

Known-future covariates receive a separate intervention edge. Routing selects
the strongest stable incoming edges per target and records an append-only graph
revision. The graph never fabricates causality from a name; every edge is tied
to settled temporal evidence.

Every forecast recomputes the candidate topology. An unchanged topology
refreshes live edge scores immediately. A changed topology becomes active after
three consecutive candidate observations; an O1 regime change starts a fresh
route state immediately. The exact `(source, target, lag, kind)` tuples form the
graph identity used by Crystals and Action profiles. This keeps routing online
without fragmenting stored compute on one-window score jitter.

## Forecast Crystals

A Forecast Crystal is a compact local action, not a cached answer. Context is
encoded into fixed features `phi(x)` (level, slope, seasonal lags, routed
covariates, known-future summaries). Settled outcomes update additive ridge
statistics:

```text
G <- G + phi phi^T
B <- B + phi y_future^T
W  = (G + lambda I)^-1 B
```

Residual quantiles supply uncertainty. A Crystal becomes executable when its
sequential support, error, bias, and interval coverage satisfy its promotion
contract. Regime or graph changes select another Crystal rather than mutating a
foreign one. Partial settlements update every observed horizon cell immediately;
unobserved cells use the Crystal's pre-update point as a neutral training value
and never enter error or coverage totals.

## Economic Action Bank

Each completed ticket records action, wall time, predictor forwards, score,
coverage, regime, and graph revision. The Action Bank maximizes verified net
value:

```text
net(action) = avoided_foundation_cost
              - action_cost
              - quality_penalty
              - calibration_penalty
```

Unknown contexts fall through to the foundation predictor. A cheap action is
never allowed to become its own quality authority.
