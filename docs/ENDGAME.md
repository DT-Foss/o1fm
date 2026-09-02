# O1FM Endgame

## The Self-Compiling Causal Forecast Fabric

A foundation forecaster predicts the next trajectory. O1FM builds and operates
the executable model of the world that produces trajectories.

The endgame is a temporal operating system with five coupled capabilities:

1. forecast factual multivariate trajectories;
2. maintain calibrated marginal and joint uncertainty under live drift;
3. branch over explicit interventions and counterfactual conditions;
4. compile settled, repeated dynamics into tiny executable Forecast Crystals;
5. spend observations, teacher forwards and idle compute where they remove the
   most decision-relevant uncertainty.

```text
live stream + known interventions
                |
          causal Atlas
                |
       O1 state and regime
                |
     directed conformal field
                |
       economic action compiler
        /          |          \
   Crystal     local model     open teacher
        \          |          /
     factual trajectory + joint envelope
     counterfactual branch graph
     decision-regret envelope
                |
             settlement
                |
 O1 + Atlas + Conformal + Crystal learning
                |
 active sensing and background branch farming
```

## Three categories beyond forecasting

### 1. Governed trajectories

Every prediction carries a route, regime, action and model identity. The runtime
returns ordered quantiles, asymmetric conformal reserves and a whole-trajectory
envelope. Realized outcomes are the only authority that can change coverage,
promote a cheap action or charge a Crystal.

### 2. Executable causal branches

Known-future covariates become named interventions. The live graph separates
observational edges from intervention edges. O1FM evaluates factual and
interventional trajectories through the same settlement contract, then compiles
stable local response surfaces into Forecast Crystals. A Crystal is a local
dynamics operator, not a cached answer.

### 3. Autonomous temporal computation

O1FM computes the value of another measurement, intervention, teacher call or
simulation. Idle compute explores graph-reachable branches and materializes the
ones with positive expected live value. Past compute is released later as a
verified local response surface. This is the forecasting equivalent of a charged
power bank: work is prepaid over a causal state/action region and discharged when
the live state enters that region.

## The backbone is a replaceable teacher

O1FM owns the control plane. External foundation models are optional teachers,
not the architecture.

- **Toto 2.0 22M** is the foundation trajectory authority. Its Apache-2.0
  checkpoint is 84 MB, emits nine quantiles and alternates causal temporal with
  full variate attention. **Toto 2.0 4M** is the continuous local authority;
  O1FM periodically refreshes it against 22M. The published 2.0 release does
  not support future exogenous variables; its role is factual and local
  trajectory authority, not intervention execution.
  [Official model family](https://huggingface.co/Datadog/Toto-2.0-22m)
- **Chronos-2 120M** is the covariate teacher. It is Apache-2.0 and natively
  supports multivariate targets, past-only covariates and known-future
  covariates. [Official model card](https://huggingface.co/amazon/chronos-2)
- **TimesFM-3** remains an architecture reference. Its source is Apache-2.0 and
  its default checkpoint uses the separate non-commercial license.
  [Google Research](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)

ActionBank can route between teachers by capability and economics. O1FM's own
backbone remains the sovereign trainable implementation and the destination for
compiled knowledge.

## Directed O1 Conformal Field

A symmetric global width multiplier is too small for directed regime drift.
O1FM maintains separate lower-tail and upper-tail reserves for every
`[route, regime, action, horizon, target]` cell, plus a joint path reserve.

For realized target `y`, outer quantiles `q_lo`, `q_hi` and point forecast `m`:

```text
scale   = max((q_hi - q_lo) / 2, eps * (1 + abs(m)))
s_lower = max(0, (q_lo - y) / scale)
s_upper = max(0, (y - q_hi) / scale)
```

Exact local score quantiles shrink toward a model/horizon parent field. This
lets a new Crystal, route or O1 regime inherit useful calibration immediately,
then become action-specific as its own settlements arrive.

Each side follows an O1-driven adaptive conformal update:

```text
gain = base_gain * f(O1_surprise, O1_cusum)
alpha_side <- clip(
    alpha_side + gain * (target_tail_error - observed_tail_error)
)
```

Undercoverage lowers `alpha` and expands the next interval. Overcoverage releases
reserve. Surprise accelerates adaptation exactly when O1 detects that the old
error distribution stopped describing the stream. The current target is 80%
marginal coverage from the outer 10–90% quantiles, with a separate 80% joint
trajectory envelope.

The algorithm extends conformalized quantile regression and adaptive conformal
inference to O1FM's route/action hierarchy. The underlying score and online
alpha update follow established work; the persistent directed field, causal
identity hierarchy and compute/action coupling are O1FM's architecture.

References:

- [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170)
- [Copula Conformal Prediction for Multi-Step Time Series](https://openreview.net/pdf?id=ojIJZDNIBj)
- [MultiDimSPCI](https://proceedings.mlr.press/v235/xu24m.html)

## Mathematical constraints from the Foss papers

### Temporal causality and Sinkhorn are different axes

Exact bistochasticity and strict causal lower-triangular flow intersect only at
identity. Temporal attention therefore remains causal and row-stochastic.
Sinkhorn transport acts only across simultaneously visible variates inside an
aligned patch. A Prefix-Usage temporal operator is a separate causal relaxation
and must be co-trained with every layer on which it acts.

### Causal transport is an additive budget

For a causal row-stochastic kernel `A`, query distribution `q` and induced key
usage `u=qA`, the exact identity

```text
W1(q, u) = sum_ij q_i A_ij (x_i - x_j)
```

makes direct backward reach and usage imbalance the same quantity. Across a
stack these distances add without cancellation. O1FM can therefore attach risk,
memory and compute reserves to a causal path with an exactly compositional
transport budget.

This primitive is now executable. O1FM independently evaluates coupling cost
and discrete CDF-`W1`, enforces stack additivity, constructs normalized
per-target route transports, and feeds the live mean route distance into every
Crystal profile. The native receipt measures mean route `W1 = 1.9201`; the real
Toto receipt measures `2.0580` with a maximum of `2.1591`.

### Directed state needs more than one scalar

The scalar Möbius no-go classifies every state-independent positive multiplier
and shows that an even Lorentz-like dilation requires a two-component lift.
O1FM applies the design consequence to uncertainty: signed lower/upper drift is
preserved as a vector field instead of collapsed into one symmetric width.

## The compilation loop

```text
teacher/local forward
    -> calibrated trajectory ticket
    -> realized settlement
    -> route-specific residual and intervention evidence
    -> additive Crystal statistics
    -> ActionBank quality/coverage/cost receipt
    -> O1 novelty and learning-progress update
    -> compile, retain, replace or discard local operator
```

This differs from static TSFM distillation. DistilTS compresses one foundation
model into one fixed student and reports up to 1/150 the parameters and 6000x
inference acceleration. O1FM continually produces many tiny, regime- and
intervention-specific operators and lets the foundation teacher remain the
fallback for novelty. [DistilTS](https://arxiv.org/abs/2601.12785)

## Route to the full system

1. Directed O1 Conformal Field with marginal and joint receipts.
2. Toto-2.0-22M and Chronos-2 adapters behind one teacher protocol. **Running.**
3. Factual and intervention-aware trajectory branches plus local response
   surfaces. **Teacher branches and Intervention Response Crystals running.**
4. `.causal` live Atlas records for routes, field revisions and compiled
   response surfaces. **Forecast/transport/settlement citations running.**
5. O1 Cartographer jobs for autonomous horizon/target/action/intervention
   measurement. **Running with D-optimal acquisition and physical-forward
   coalescing.**
6. Battery policy for profitable idle-time branch charging.
7. Markov/Seed policy learning over action, consequence, quality and work cost.
8. Active value-of-information selection over sensors and interventions.

The terminal product does not merely answer “what happens next?” It maintains a
live executable map of what can happen, why, how confident the system is, what
changes under intervention, and which future computation has already been paid.

## First open-teacher execution

The Toto-2.0 family has now completed the full loop with actual Apache-2.0
weights. Toto-4M runs as continuous local authority; Toto-22M refreshes authority
after at most four local actions. Across 24 windows this removes 17 large-model
calls (70.83%) and 51.92% of parameter-equivalent forward work. Marginal
coverage rises from the all-22M control's 86.92% to 87.55%, joint coverage rises
from 87.5% to 91.67%, and pinball changes by only 0.00141. Both checkpoints are
pinned by their loaded tensor hashes. No ridge Crystal passed the strict gate,
which fixes the next compiler task: replace summary-only Crystal inputs with
contextual local-teacher states and causal path features.

Shadow settlement now charges every fully computed candidate's exact Conformal
Field without duplicating parent evidence. The local-base residual Crystal
reaches 78.22% shadow coverage and 0.07372 pinball, but 4M remains superior at
89.23% and 0.06486 for the same forward cost, so ActionBank correctly keeps the
Crystal out of authority.

The same run commits 72 `.causal` segments. Forecast origin, prediction,
normalized transport and settlement form a directly addressable evidence chain;
the manifest and segments verify after a fresh remount. This is the first live
substrate for the future counterfactual branch compiler.

## First counterfactual cartography execution

Chronos-2 120M now executes three aligned worlds from the same forecast origin:
the factual schedule, a graph-supported seasonal-driver shift, and an explicit
intervention-off hypothesis. `SyntheticStructuralState` replays each world with
the identical latent state, exogenous causal process and sampled noise, so every
branch has its own legitimate target.

The Cartographer starts from 36 horizon × target-group × intervention cells.
D-optimal acquisition selects 12; the execution bridge evaluates them with
three physical forwards, a 75% reduction, while persisting exact work, payload,
O1-shadow and result receipts. The `.causal` origin/transport/settlement chain
verifies live and after remount.

The measurement also identifies the next original O1FM operator. Chronos
recovers 17.9% of the magnitude of the graph-supported effect and 1.24% of the
unseen intervention effect, with the effect direction wrong in both cases.
These are now named response-surface debts, not invisible teacher error. O1 can
prioritize those cells and compile their settled residual field into an
intervention Crystal, leaving Chronos as the trajectory prior and novelty
fallback.

## First intervention compiler execution

The measured response debt is now executable. Eight earlier same-regime windows
settle eight distinct magnitudes for two causal operators and one structural
placebo. Four later windows use four magnitudes absent from training. The
Response Crystal receives only route `W1` and the known-future delta schedule;
every simultaneous branch is predicted before any branch at that origin is
settled.

On the later magnitudes, raw Chronos recovers 9.16% of the seasonal-driver
effect and 2.80% of the intervention-level effect. The local operator recovers
99.97% and 99.96%, with direction cosine effectively one. Mean MAE falls from
0.25382 to 0.13812 and mean pinball from 0.10693 to 0.05473. The placebo remains
neutral. ActionBank selects the Response Crystal for all 12 branches.

One factual Chronos forward now releases three alternative trajectories. The
selected serving policy uses four forwards where direct branch execution needs
16, a 75% reduction. The measurement command still executes the 12 raw branch
forwards as benchmark-only calls and records both actual measurement compute and
selected serving compute separately.

The response effect field reaches 80.53% empirical coverage. The Directed O1
Conformal Field settles 30 compiled branches at exactly 80% marginal coverage;
the four-window unseen-magnitude suffix rises from 47.40% raw coverage to
71.88%. The remaining suffix calibration debt, new operator shapes, topology
shifts and regime transfer are now the next Cartographer targets rather than
missing architecture.

## First response transfer execution

The compiler now receives full factual-state, route and time-resolved operator
geometry. Sixty pulse/constant candidates span four factual origins; the
Cartographer acquires 48 at design rank 48. No ramp, masked interval or
double-pulse operator appears in that fit set.

Later regime-0 windows evaluate exactly those unseen shapes. A second structural
regime supplies a genuinely different authenticated route topology. Both exact
domains share one explicit parent fit and projection scope, while ActionBank,
Conformal, direction and coverage authority remain graph/regime exact.

The new topology demonstrates the intended state machine:

```text
parent response exists
    -> exact profile unknown
    -> 9/9 branches use Chronos
    -> three audit origins settle exact evidence
    -> next 12/12 branches use the parent Crystal
```

After release, MAE falls from 0.15764 to 0.13614, pinball from 0.06192 to
0.05110, and marginal coverage rises from 58.33% to 89.93%. Serving the four
origins requires one factual forward each instead of factual plus three branch
forwards: 16 → 4. Across all response settlements, the directed O1 field holds
80.24% marginal coverage.

Authority O1 stays in generation 0 across six stable regime-0 origins. CUSUM
reaches `0.44243` without firing at origin 496; the first regime-1 forecast at
origin 512 contributes surprise `0.85018`, crosses the measured threshold and
advances generation `0 -> 1`. The new exact response key binds both O1
generation 1 and the new graph topology. The parent/fallback/audit/release cycle
therefore now holds across state, unseen operator shape, structural regime,
authenticated topology and an actual continuous O1 generation transition.
