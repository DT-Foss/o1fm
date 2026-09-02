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
       -> LocalTeacher.predict          # small priced authority
       -> StructuralBaseline.predict    # zero foundation forwards
       -> ForecastTeacher.predict       # novelty authority
  -> DirectedConformalField.apply       # prior settlements only
  -> ForecastVerifier
  -> ForecastTicket

ForecastTicket + realized future
  -> proper scoring / residual vector / coverage
  -> O1State.observe
  -> DirectedConformalField.observe
  -> CausalGraph.observe
  -> ForecastCrystalBank.observe
  -> ActionBank.settle
```

Every learned object is scoped by model identity, feature schema, horizon,
causal-graph revision, and O1 regime generation.

## Open teacher protocol

The novelty authority is a strict `ForecastTeacher`, not a concrete model
class. Native O1FM, Toto-2 and Chronos-2 all consume `ForecastInputs` and return
the identical immutable `ForecastPrediction` contract. External packages load
lazily and remain optional. Teacher identities bind adapter, checkpoint and
quantile inventory; executable receipts additionally hash the loaded tensors.
An immutable capability contract declares multiple-target, past-covariate,
known-future-covariate and probabilistic support. The branch runtime requires
that record before assigning intervention work; it is also available to
capability-aware ActionBank routing.

Toto-2.0 maps targets, past-only covariates and the historical prefix of known
covariates into multivariate context. Its published 2.0 release does not support
future exogenous variables, so it is factual/local authority and cannot execute
an intervention branch. Chronos-2 receives past and future covariate roles
through its official list-of-dictionaries API and is the current intervention
teacher. Everything after the teacher call—Conformal Field, O1, Graph, Crystals
and ActionBank—is shared.
An optional local teacher is evaluated as a priced candidate. ActionBank can
select it only after profile-local quality and coverage settlement. A bounded
local streak forces a foundation refresh, renewing authority, calibration and
compiler evidence under drift.

## Directed conformal control

The base predictor's outer quantiles define scale-normalized, non-negative CQR
scores independently for the lower and upper tails. Exact histories are keyed by
action, route and regime. A parent field shares evidence across cold actions and
new graph/regime identities. Forecast-time application reads committed history
only; settlement appends the realized score afterward.

The two tail alphas and the joint trajectory alpha follow adaptive conformal
updates. O1 surprise and CUSUM increase the update gain under drift. The point
forecast and inner quantiles remain untouched; only the outer interval expands.
The joint path envelope is returned separately and never forced into the
marginal prediction.

The temporal/variate distinction is structural: causal temporal attention is
row-stochastic, while exact Sinkhorn transport is applied only across
simultaneously visible variates. Exact bistochastic causal temporal attention
would collapse to identity.

## Citable `.causal` Atlas

`ForecastAtlas` adapts any content-addressed `SegmentStore` and lazily mounts
IMMER's `LiveStore` when requested. Forecast, normalized route transport and
settlement are separate immutable segments with exact direct citations.
Forecast records contain identities and array hashes, never raw model weights;
settlement records contain scores and state-transition hashes, never raw
targets. The normal O1FM import has no IMMER dependency.

The real Toto run writes 72 unique segments and reaches authenticated manifest
revision 72. A fresh process remounts the store, verifies the journal and every
segment, then resolves the first forecast record directly from its receipt.

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

For each target, positive routed edge scores are normalized into an explicit
causal transport distribution over lags. Its exact backward `W1` cost is the
score-weighted expected lag; intervention edges carry lag zero. The uniform
mean across targets is appended to the Crystal feature vector but not to the
topology identity, so live transport strength changes without fragmenting the
profile. Raw correlations themselves are never mislabeled as transport mass.

Rolling windows are appended through `observe_many()`. The complete batch is
validated before mutation, the persistent event chain is loaded once, and the
directory is synchronized once after the appended suffix. Serial and batched
paths produce identical graph revisions and edges.

## Forecast Crystals

A Forecast Crystal is a compact local action, not a cached answer. Context is
encoded into fixed features `phi(x)` (level, slope, seasonal lags, routed
covariates, known-future summaries). Settled outcomes update additive ridge
statistics:

```text
G <- G + phi phi^T
B <- B + phi z_fit^T
W  = (G + lambda I)^-1 B
```

External-teacher mode sets `z_teacher` to the uncalibrated teacher trajectory;
an executed Crystal supplies its own point as a neutral self-target. Unrelated
structural actions update residual evidence without changing the Crystal fit.
Native mode instead fits the realized local dynamics directly.
Both modes use realized outcomes for residual quantiles, error and promotion
authority. A Crystal becomes executable when its
sequential support, error, bias, and interval coverage satisfy its promotion
contract. Regime or graph changes select another Crystal rather than mutating a
foreign one. Partial settlements update every observed horizon cell immediately;
unobserved cells use the Crystal's pre-update point as a neutral training value
and never enter error or coverage totals.

## Intervention Response Crystals

An `InterventionResponseCrystal` compiles a local effect surface instead of a
complete factual forecast. Its exact key binds model weights, state-feature ABI,
intervention-vector ABI, target order/schema, route topology, regime, horizon
and target count. The executable input contains only factual state, route
transport and a known-future intervention schedule.

For paired factual and intervention targets, the settled effect is

```text
delta_y = y_intervention - y_factual
```

Centered additive ridge statistics learn `delta_y` over the intervention
vector. Prediction adds the compiled effect to one factual teacher trajectory.
Ordered effect quantiles are composed with the factual quantiles, then the
Directed O1 Conformal Field calibrates the resulting branch distribution from
prior settlements. The comonotonic quantile sum is therefore only the base
proposal; conformal coverage is the authority.

Readiness requires sequential pre-update support, effect MAE, direction cosine
and empirical effect coverage. Every observation uses a write-ahead record plus
the inner Crystal's content identity. A failed atomic persist rolls in-memory
state back to the last durable state before retry, preventing a process-local
replay from outrunning remounted evidence.

All branches at one forecast origin follow a hard barrier:

```text
predict every branch -> decide every action -> settle every branch
```

No alternative world's realized outcome can influence another simultaneous
branch prediction.

`InterventionResponseFeatureEncoder` supplies the executable geometry. Its
past-only state vector contains level, slope, volatility, observation masks and
timestamp-derived seasonal phase. Its route vector contains per-target `W1`,
incoming score/lag mass and every known-channel × target edge cell. Its operator
tensor retains operation kind, applied delta, sign, time, contiguous-run
position/duration, pulse and ramp structure for every known-future cell. A
schema-seeded signed projection gives Cartographer and the ridge compiler one
bounded D-optimal design space without a future-target channel.

Exact topology-bound schemas retain their own identity and route validation.
Compatible schemas may declare one explicit projection and fit scope. This
shares parent response parameters while keeping exact graph/regime wrapper
statistics separate:

```text
shared parent prediction
    + unknown exact ActionBank profile -> foundation fallback
    + exact direction/coverage/action support -> local release
```

A new topology can therefore benefit from transferred structure immediately
without making the transferred predictor its own authority.

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

## Exactly-once incremental settlement

Realized horizons may arrive cell by cell. `SettlementLedger` stores the
already-observed mask and value union per forecast ticket. Every request yields
only its previously unseen cells and one content-derived fragment identity.
O1, Conformal, Crystal and ActionBank consume that same fragment identity.

The ledger commits only after every consumer and the Atlas succeed. A crash
before commit therefore replays the identical fragment through idempotent
consumers; a growing cumulative target learns only the new suffix; changing an
already observed value is a hard conflict. Settlement Atlas v2 records the
fragment hash and observed-value count directly.

## Executable branch cartography

`TeacherBranchRuntime` executes factual and future-known intervention inputs
through any capability-compatible teacher. `BranchExecutor` classifies a branch
as graph-supported only when lag-zero intervention evidence covers the complete
Cartesian set of edited sources × declared targets. Partial support remains an
explicit counterfactual hypothesis.

`ForecastCartographer` maintains the persistent measurement frontier. Its
execution bridge coalesces target and horizon cells that share one physical
model execution, allocates measured work and payload bytes exactly once, and
updates only a private shadow O1 stream. Live forecast authority is never
advanced by exploratory probes.

The synthetic world retains its latent target, exogenous process, structural
weights, regimes and exact sampled noise. `counterfactual_target()` replays an
edited known-future schedule from the same origin and noise path. This gives
every simulated intervention its own realized labels without putting future
targets into `ForecastInputs`.
