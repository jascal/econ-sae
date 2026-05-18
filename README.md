# econ-sae

**A stock-flow-consistent macroeconomy as a tensor bundle, used as ground-truth substrate for SAE interpretability research.**

econ-sae packages a small but exact macroeconomy --- households (in three
cohorts), firms (in three goods sectors), a government, and a bank --- as
a stack of agent vectors and a transaction log. Every transaction
satisfies double-entry bookkeeping by construction, so each trajectory
has an exactly known feature factorization that mixes hard categorical
axes (cohort, sector, transaction kind, shock kind), continuous bucketed
axes (debt bucket, cash bucket, mpc bucket, leverage), and *conjunctive*
features that are deliberately polysemantic-trap candidates (e.g.,
"firm AND indebted AND high-inventory"). A sparse autoencoder trained on
this substrate can be scored against ground truth at multiple difficulty
tiers --- not just the clean factorial features sm-sae provides.

The double meaning is intentional: **econ-sae** = "economy SAE", the
economic-conservation-obedient-network complement to
[sm-sae](https://github.com/jascal/sm-sae), which does the same thing for
the Standard Model of particle physics.

## Status

Pre-alpha, **Phase 1**. The multi-good, multi-cohort, credit-enabled
simulator runs cleanly; all 6 accounting identities hold to machine
precision; the SAE training + AUC alignment pipeline trains 9 SAEs
(3 variants × 3 feeds) and grades them on the 51-feature ground-truth
vocabulary with a per-tier difficulty breakdown. The Polygram bridge
remains a stub for the next iteration.

### Headline results (32 trajectories × 60 periods, default Phase 1 run)

Coverage at AUC ≥ 0.95 by tier, on the world-model-activation feed
(the most realistic SAE setting):

|                       | categorical (24 feats) | bucketed (7) | conjunctive (8) | regime (6) |
|-----------------------|------------------------|--------------|-----------------|------------|
| **TopK** (n=256, k=12) | 16.7% | 14.3% | 25.0% | 0.0% |
| **L1** (n=256)         |  6.7% |  0.0% | 25.0% | 0.0% |
| **JumpReLU** (n=256)   | **33.3%** | **42.9%** | **50.0%** | 0.0% |

The expected difficulty hierarchy emerges clearly. Categorical features
(sector, firm_sector, cohort) are recovered cleanly at AUC = 1.0 because
they live directly on coordinate axes. Bucketed continuous-quantile
features and conjunctive 2-3-term compositions are partially recovered
on the SAE-friendly substrate (world-model h1 activations).
**Regime / phase features are never recovered** — they are
window-aggregated macro labels with no per-agent footprint, exactly the
hard tier the design brief asked for. Mean-best AUC across all 51
features tops out at 0.753 (JumpReLU on acts); on sm-sae the equivalent
numbers are far higher because the SM's feature factorization is purely
factorial.

### Width / duration sweep (acts feed, JumpReLU + variants)

A follow-up sweep on the acts feed tested whether more SAE capacity or
longer training pushes conjunctive recovery higher:

| config                  | width | epochs | conj cov95 | conj mAUC | VE    |
|-------------------------|-------|--------|------------|-----------|-------|
| baseline JumpReLU       | 256   | 200    | 50.0% (4/8) | 0.842    | 0.999 |
| **JumpReLU wider**      | 1024  | 200    | **62.5%** (5/8) | 0.844 | 0.984 |
| JumpReLU longer         | 256   | 500    | 50.0% (4/8) | 0.844    | 0.999 |
| JumpReLU wider + longer | 1024  | 500    | 50.0% (4/8) | 0.831    | 0.987 |
| TopK wider (k=20)       | 1024  | 200    | 25.0% (2/8) | 0.782    | 0.695 |
| **L1 wider + loose**    | 1024  | 200    | **62.5%** (5/8) | 0.844 | 0.274 |

Findings:

1. **Width helps modestly; duration alone is a no-op.** Going 256 → 1024
   features moves both JumpReLU and L1 from 4/8 to 5/8 conjunctive
   recovery. Going 200 → 500 epochs at fixed width changes nothing.
2. **Width + duration regressed.** 1024 features at 500 epochs falls
   back to 4/8 — longer training at wider widths spreads SAE features
   across easier categorical structure and unlearns the marginal
   conjunctive ones.
3. **Two conjunctive features are unreachable in the current 32-traj
   ensemble** (their threshold conditions never coincide):
   `durables_firm_high_inv` and `firm_AND_indebted_AND_high_inventory`.
   The *achievable* conjunctive ceiling is therefore 6/8 = 75%, and the
   wider configs reach 5/6 = 83% of that ceiling.
4. **One feature stays just out of reach.** `young_AND_indebted` sits
   at AUC ≈ 0.88 across every config tried — visible to the SAE but not
   cleanly monosemantic. That's the one currently-feasible conjunctive
   feature that more SAE capacity didn't recover.
5. **TopK scales poorly** at fixed `k`. Going 256 → 1024 features with
   k=20 leaves most features dead; variance-explained collapses to
   0.70 and recovery drops across the board. TopK needs `k` scaled
   with width.

The conjunctive ceiling here is a substrate limit, not an SAE limit.
Pushing past 5/6 likely requires a richer world model (deeper / wider
h1, or a transformer-style attention block over agents) so that more
conjunctive features get encoded in the activations the SAE sees.

### Scale experiment: bigger ensemble + deeper world model

To test that ceiling, a follow-up trained a `DeepWorldModel`
(hidden_dims=192/128/64, vs the baseline 96/48) on a 128 × 100
ensemble (215k samples, ~7× baseline), then trained the best two SAE
configs on the resulting 192-d activations.

| config              | conj cov95   | regime cov95   | regime mAUC | overall cov95 |
|---------------------|--------------|----------------|-------------|---------------|
| baseline (shallow)  | 50.0% (4/8)  | 0.0% (0/6)     | 0.595       | 33.3% |
| width sweep (shallow, 1024w) | 62.5% (5/8) | 0.0% (0/6) | 0.594 | 35.3% |
| scale (deep, 256w)  | 37.5% (3/8)  | **16.7% (1/6)** | **0.678** | 35.3% |
| scale (deep, 1024w) | 37.5% (3/8)  | 0.0% (0/6)     | 0.675       | 29.4% |

Per-feature conjunctive AUC trajectory:

| feature                            | baseline | width sweep | scale (deep, 1024w) |
|------------------------------------|----------|-------------|---------------------|
| food_firm_low_inv                  | 1.000    | 1.000       | 1.000               |
| services_firm_high_output          | 0.998    | 1.000       | 0.999               |
| prime_AND_high_cash                | 0.962    | 0.962       | 0.901               |
| retiree_AND_decumulating           | 0.952    | 0.972       | 0.938               |
| young_AND_high_mpc_AND_expansion   | 0.933    | 0.952       | **0.968**           |
| **young_AND_indebted**             | 0.891    | 0.865       | **0.928** (best)    |
| durables_firm_high_inv             | 0.500    | 0.500       | 0.500 (unreachable) |
| firm_AND_indebted_AND_high_inventory | 0.500  | 0.500       | 0.500 (unreachable) |

Three findings:

1. **Conjunctive mean AUC is stable around 0.84** across all five SAE
   configs. The 0.95 *threshold* moves up and down by ±1 feature config
   to config, but the underlying recovery quality is at a real substrate
   ceiling. Different configs simply catch different features near the
   boundary.

2. **`young_AND_indebted` finally moved** (0.87 → 0.93). It's the
   conjunctive feature most sensitive to substrate richness — needs
   both the deeper world model *and* the bigger ensemble before the SAE
   can resolve it. Still below 0.95 in every config tried.

3. **Regime features became partially recoverable for the first time.**
   Regime mAUC climbed from 0.595 (every Phase 1 config) to 0.678
   (every scale config), and one of the six regime features crossed
   AUC ≥ 0.95 in the small-SAE scale run. The deeper world model
   processes macro context through more layers, so phase information
   that was previously invisible in the substrate now lives in h1. This
   is the most interesting qualitative shift in the experiment.

4. **The 2 dead conjunctive features stayed dead.** Their threshold
   conditions never coincide in the simulator, with or without 7× the
   data — these are vocabulary-definition bugs, not data sparsity.

5. **L1 at the wider 192-d substrate collapsed** (cov95 31% → 2% despite
   VE 0.988 and L0 23.6). `l1_coeff=2e-3` was wrong for the new
   substrate; the SAE found a reconstruction-optimal basis that doesn't
   track ground truth. JumpReLU's hard-threshold mechanism is more
   robust to substrate changes.

### Attention over agents (`AttnWorldModel`)

The biggest qualitative win came from giving the world model
self-attention across agents within each period. The architecture:

```
per-period input (N_agents, agent_dim + macro_dim + shock_dim)
  -> input_proj    -> (N_agents, embed_dim=64)
  -> MultiheadAttn -> (N_agents, embed_dim)   [4 heads, 1 layer]
  -> fc1 + ReLU    -> (N_agents, h1_dim=192)  [SAE substrate]
  -> fc2 + ReLU + fc3 -> (N_agents, agent_dim)
```

At the same time, the 2 previously-unreachable conjunctive features had
their thresholds relaxed (p75 inventory instead of fixed multipliers)
so they fire at 1.5% / 2.6% prevalence — giving the SAE a real 8/8
conjunctive target for the first time.

Conjunctive per-feature AUC trajectory across all experiments
(JumpReLU, 1024w / 200ep where applicable):

| feature                              | baseline | width sweep | scale (deep) | **attn**  |
|--------------------------------------|----------|-------------|--------------|-----------|
| food_firm_low_inv                    | 1.000    | 1.000       | 1.000        | 1.000     |
| services_firm_high_output            | 0.998    | 1.000       | 0.999        | 1.000     |
| prime_AND_high_cash                  | 0.962    | 0.962       | 0.901        | **0.978** |
| retiree_AND_decumulating             | 0.952    | 0.972       | 0.938        | 0.937     |
| young_AND_high_mpc_AND_expansion     | 0.933    | 0.952       | 0.968        | 0.969     |
| young_AND_indebted                   | 0.891    | 0.865       | 0.928        | **0.940** |
| durables_firm_high_inv (was dead)    | 0.500    | 0.500       | 0.500        | **0.999** |
| firm_AND_indebted_AND_high_inv (was dead) | 0.500 | 0.500      | 0.500        | **0.919** |
| **mean**                             | 0.842    | 0.844       | 0.842        | **0.968** |
| **min**                              | 0.500    | 0.500       | 0.500        | **0.919** |

Two qualitative shifts that no amount of SAE-side tuning produced before:

1. **All 8 conjunctive features are now AUC ≥ 0.90.** The mean AUC
   jumped from 0.842 to 0.968 — a substrate change of a magnitude every
   prior SAE-side experiment failed to produce. Threshold-counting
   `cov95` reports 5/8 (the same nominal number as the width sweep
   reported on 6 reachable features), but the *true ceiling* is now
   8/8 reachable and the worst recovery is 0.92.
2. **The two newly-firing features were immediately recoverable**
   (0.999 and 0.919) — the AttnWorldModel encoded them in h1 without
   any SAE-side tuning. This is the cleanest evidence so far that the
   recovery ceiling is set by the world model, not by the SAE.

What did *not* improve:
- Regime / phase features (0% cov95, mAUC ≈ 0.61) — attention over
  agents within a single period can't recover features defined over a
  multi-period window. These need a temporal model.
- Categorical and bucketed coverage — both saturated at the same
  ceiling as the scale experiment.

The lesson generalizes: each tier of ground-truth difficulty needs a
matching capability in the world model. Categorical works with anything;
conjunctive needs cross-agent context (now provided by attention);
regime needs cross-time context (would need a recurrent or
sequence-attending model).

### Temporal experiment: architecture is not enough

The natural follow-up was to add cross-period recurrence to the
AttnWorldModel and test whether the regime tier moves. The hypothesis:
"GRU hidden state across periods will encode the 5-period trailing
macros that phase labels are computed from, so the SAE substrate will
finally include phase information." The architecture:

```
per-period cross-agent attention   (same as AttnWorldModel)
  -> per-agent GRU across periods  (NEW: gru_hidden=128)
  -> per-(period, agent) MLP head
```

Result: **the architectural extension did not move the regime tier.**
Per-feature regime AUCs from `TemporalWorldModel` 1024w:

```
  phase:contraction      0.566
  phase:expansion        0.583
  phase:fiscal_active    0.526
  phase:high_leverage    0.622
  phase:high_rate        0.574
  phase:monetary_active  0.504
```

vs the attention-only model (regime mAUC 0.610): temporal is **worse**
on average, and conjunctive recovery is essentially unchanged
(mAUC 0.964 vs 0.968). The full comparison:

| substrate          | conj mAUC | conj cov95 | regime mAUC | regime cov95 | WM params |
|--------------------|-----------|------------|-------------|--------------|-----------|
| shallow WM (Phase 1) | 0.842   | 4/8 reach. | 0.595       | 0%           | 7k        |
| DeepWorldModel (scale) | 0.842 | 3/8 reach. | 0.678 (256w)| 16.7% (256w) | 43k       |
| AttnWorldModel      | 0.968   | 5/8 (8/8 reach.) | 0.610  | 0%           | 60k       |
| TemporalWorldModel  | 0.964   | 5/8 (8/8 reach.) | 0.562  | 0%           | 147k      |

This is a *more* informative outcome than "regime mAUC climbed" would
have been. The diagnosis:

- The simulator is one-step Markovian. Next-state MSE — the world-model
  training objective — only rewards information that helps predict
  `state_{t+1}` from `state_t + shock_t + macro_t`.
- Phase labels (`expansion`, `contraction`, ...) are functions of *past*
  GDP. They are not informative for predicting the next state once
  current macros are visible.
- The GRU **could** carry 5-period trailing GDP in its hidden state. It
  has no gradient pressure to. Hidden state collapses to whatever helps
  one-step MSE, which doesn't include history.

**The generalization**: an SAE substrate only encodes what the world
model's training objective rewards encoding. Architecture + data isn't
enough; the loss has to *demand* the feature for it to land in
activations. To recover regime features, the world-model objective
itself has to change — e.g., add an auxiliary head that predicts the
next period's macros (or directly the phase label), or train against a
multi-step rollout MSE so that early-step hidden state has to plan over
horizons longer than 1.

### Sentiment-driven MPC: the diagnosis was wrong

A natural test of the "Markovian-simulator" diagnosis is to make the
simulator *itself* non-Markovian: introduce a behavioral rule that
depends on a multi-period window of macros not stored in any agent
coord. The cleanest such mechanism is sentiment-driven MPC --- each
household scales its consumption target by +/-20% when the perceived
regime (computed from a 5-period trailing GDP buffer living outside
agent state) is expansion / contraction. Now `state_{t+1}` provably
depends on past macros the world model can only know by accumulating
them itself, so the GRU should have gradient pressure to encode them.

Result: regime mAUC stayed at 0.56, essentially unchanged from the
no-sentiment temporal run.

| metric                       | temporal (no sentiment) | **temporal + sentiment** |
|------------------------------|-------------------------|--------------------------|
| regime mAUC (1024w)          | 0.562                   | **0.559** (no change)    |
| conjunctive mAUC (1024w)     | 0.964                   | 0.972                    |
| world-model final MSE (z)    | 0.0230                  | 0.0203 (slightly better) |
| `phase:expansion` AUC        | 0.583                   | 0.606                    |
| `phase:high_leverage` AUC    | 0.622                   | 0.616                    |
| `phase:fiscal_active` AUC    | 0.526                   | 0.520                    |

The hypothesis was clean and the experiment falsified it. The fix:

1. **Agent state already encodes most of "recent macro history" implicitly**
   through accumulated-stock coords. `inventory` integrates production
   minus sales, `debt_liab` integrates loan flows, `expectation` is an
   EMA of past income, etc. So the simulator isn't really Markovian in
   the (state) coordinate alone; the *state itself is a compressed
   proxy for recent macro history*.
2. With sentiment on, world-model MSE actually got *better*
   (0.023 -> 0.020), confirming the model uses regime info for next-
   state prediction. But it uses it via the stock-coord proxies it
   already had access to --- not by learning new temporal features in
   the GRU.
3. **The right framing isn't Markovian-vs-not, it's continuous-vs-step.**
   The world model encodes something like a smooth "recent economic
   activity index"; the ground-truth phase labels are step functions
   (`GDP[t] > 1.10 * trailing_mean`). A smooth activation feature
   correlates with the binary label at AUC ~0.6 (exactly what we see),
   because it can't fire cleanly above a threshold without saturating.

### Continuous regime intensities: the second wrong diagnosis

The natural follow-up: replace the binary phase labels with continuous
"regime intensity" values (e.g., `(GDP[t] - trailing_mean) / trailing_mean`)
and compute Pearson correlation between each SAE feature and each
continuous label. If the smooth signal is there but threshold-binning
discards it, max correlations should be high (~0.8+) while binary AUCs
stay low.

Max |Pearson| across all 1024 SAE features (`temporal+sentiment` substrate):

| continuous label         | max \|corr\| | best SAE # | corresponding binary AUC |
|--------------------------|--------------|------------|--------------------------|
| leverage                 | **0.511**    | #86        | 0.616 (`phase:high_leverage`) |
| cons_share_food          | 0.419        | #430       | —                            |
| cons_share_durables      | 0.382        | #934       | —                            |
| gdp_volatility           | 0.281        | #982       | —                            |
| gdp_trend                | 0.240        | #418       | 0.555 / 0.606                |
| gdp_dev                  | 0.227        | #418       | 0.555 / 0.606                |
| rate                     | 0.191        | #53        | 0.549 (`phase:high_rate`)    |

Result: correlations match the binary AUCs almost exactly under the
standard relation `AUC ≈ 0.5 + 0.5 * |corr|`. **The smooth signal isn't
hidden behind the threshold — it just isn't strongly encoded.** My
binary-threshold-mismatch hypothesis was wrong.

### Revised diagnosis (third attempt)

Contrast the substrate's behavior on the two tiers:

| tier             | best AUC | structure                                                  |
|------------------|----------|------------------------------------------------------------|
| conjunctive      | 1.000    | per-AGENT property; lives in that agent's input row        |
| regime           | 0.62 max | per-PERIOD (global) property; same value for all 17 agents |

The substrate is per-(period, agent), and SAE features run per-(period,
agent) too. A per-agent property fits cleanly: one SAE feature fires
for the agent rows where the property holds. A per-period property is
harder: to encode `phase:expansion` cleanly, *the same SAE feature
would have to activate identically on all 17 agents in that period*.
The natural SAE compression assigns one feature per local concept, not
17 redundant copies of a global one, so global features get spread
across many features at low individual correlation -- exactly the
0.2-0.5 max we observe.

This is a *structural* limit, not an architectural / objective /
simulator-side one. To recover regime cleanly we'd need either:

- An SAE trained on a **macro feed** where each sample is one period's
  aggregate, not 17 per-agent rows. Then global features have
  one-sample-per-instance fit.
- A **pooling step** between the world model's h1 and the SAE: aggregate
  the 17 per-agent activations into a single per-period activation
  vector and SAE-decode that.
- Reframing the regime label as a **per-agent property** (e.g., "this
  HH would change its MPC in current conditions") so it lives at the
  same granularity as the substrate.

Conjunctive recovery stayed strong (mAUC 0.972) -- the architecture
wins persist; we just hit a structural limit on regime that no amount
of world-model tuning will fix.

### Macro-feed SAE: the structural diagnosis confirmed

The natural test of the per-agent-vs-per-period diagnosis: train a
*second* SAE on a per-period feed where each sample is one
trajectory-period, with input = `[macro_vec(10) + shock_vec(10) +
mean_pooled_h1_over_agents(192)] = 212-dim` and labels restricted to
the per-period vocabulary (`phase:*`, `shock_period_has:*`,
`txn_period_has:*`). Same trained TemporalWorldModel; only the
decoding granularity changes.

| feature                | per-agent SAE (best) | **macro-feed SAE (best)** | change |
|------------------------|----------------------|---------------------------|--------|
| **`phase:high_leverage`** | 0.616             | **0.974** (jr_w512)       | **+0.36** |
| `phase:expansion`      | 0.606                | 0.746 (jr_w512)           | +0.14 |
| `phase:contraction`    | 0.555                | 0.673 (jr_w256)           | +0.12 |
| `phase:high_rate`      | 0.549                | 0.649 (jr_w256)           | +0.10 |
| `phase:fiscal_active`  | 0.520                | 0.581 (jr_w512)           | +0.06 |
| `phase:monetary_active`| 0.509                | 0.509                     | 0     |
| **regime tier mAUC**   | 0.562                | **0.687**                 | **+0.125** |

The largest single-feature jump (+0.36 AUC) and the largest regime-tier
mAUC jump (+0.125) of any experiment in the project, validating the
structural diagnosis: **regime features are per-period; per-agent SAEs
can't cleanly recover them because the natural compression spreads
global signals across many agent-row features at low individual
correlation. Decoding at period granularity unlocks them.**

Three nuances on the unimproved features:

- **`phase:monetary_active` / `phase:fiscal_active` stayed flat.** The
  fix is informational, not structural: `shock_vec[8]` carries the
  *current* interest rate, which persists after a monetary impulse;
  there's no "impulse fired this period" indicator in the input. Adding
  one would lift these features immediately.
- **`phase:expansion` / `phase:contraction` climbed to 0.7-0.75 but not
  0.95.** These depend on `GDP[t]` vs the trailing-5 mean. A
  single-period sample has `GDP[t]` directly but only carries the
  trailing window through the world-model's mean-pooled h1, which the
  GRU has compressed lossily. Including a manual macro-window in the
  input (`GDP[t-4..t]`) would likely cross 0.95.
- **`phase:high_leverage` reached 0.974** because `debt_outstanding /
  money_stock` is computable directly from the macro vector at the
  current period -- no temporal context needed. This is the regime
  feature most "in distribution" for what the per-period feed naturally
  encodes.

The lessons compose into a clean framework:

| feature tier         | requires                                                       |
|----------------------|----------------------------------------------------------------|
| categorical          | nothing special                                                |
| bucketed             | continuous-coord visibility (always true here)                 |
| conjunctive          | cross-agent context (attention block)                          |
| regime / global      | per-period decoding granularity (macro-feed SAE)               |
| windowed-regime      | per-period decoding **+** explicit history in the input        |
| impulse-regime       | per-period decoding **+** impulse-flag in the input encoding   |

Each tier requires a *structural* match between what the substrate
encodes, how the labels are defined, and how the SAE decodes. No
single SAE / world-model / loss tweak gets you all of them; the
project is now better described as a **multi-decoder benchmark** where
different feature classes need different (substrate, decoder) pairs.

### Macro-feed v2: closing the impulse-regime gap

Phase 1.9 left two regime sub-gaps open:

- **Windowed regimes** (`phase:expansion`, `phase:contraction`) needed
  explicit history in the input — a single period's macro vector has
  `GDP[t]` but not the trailing window it must be compared against.
- **Impulse regimes** (`phase:monetary_active`, `phase:fiscal_active`)
  needed an "impulse fired this period" indicator — `shock_vec[8]`
  carries the *current* interest rate, which persists after a monetary
  impulse, so there was no way for the SAE to know whether the impulse
  fired *this* period.

Macro-feed v2 adds 7 extra input dimensions per sample:

```
+ GDP[t-4..t]           (5 dims, trailing-window history)
+ monetary_impulse_flag (1 dim, 1.0 iff monetary kind in shock_kinds[t])
+ fiscal_impulse_flag   (1 dim, 1.0 iff fiscal kind in shock_kinds[t])
```

Plus per-dim z-score normalization (macro values span 0.02 to 2090
across coords; without normalization L1/JumpReLU sparsity penalties
favor large-magnitude dims and the SAE collapses on reconstructing big
macros instead of finding small-magnitude regime features).

Results across both macro-feed variants, best per feature:

| feature                  | per-agent | macro v1 | macro v2 (z-scored) | best   |
|--------------------------|-----------|----------|---------------------|--------|
| `phase:high_leverage`    | 0.616     | **0.974**| 0.869               | **0.974** |
| `phase:fiscal_active`    | 0.520     | 0.581    | **1.000**           | **1.000** |
| `phase:monetary_active`  | 0.509     | 0.509    | **0.999**           | **0.999** |
| `phase:expansion`        | 0.606     | 0.746    | 0.792               | 0.792  |
| `phase:contraction`      | 0.555     | 0.673    | 0.782               | 0.782  |
| `phase:high_rate`        | 0.549     | 0.649    | 0.793               | 0.793  |
| **regime tier mAUC**     | 0.562     | 0.687    | **0.864**           | —      |

**Three of six regime features at AUC >= 0.97**:

- `phase:high_leverage` (closed in v1; current macros are sufficient)
- `phase:monetary_active` (closed by impulse flag)
- `phase:fiscal_active` (closed by impulse flag)

The remaining three (`expansion` / `contraction` / `high_rate`) climbed
0.15-0.25 above their per-agent baseline but plateaued at ~0.78-0.79.
They encode *thresholds on a function of a continuous window*
(e.g., `GDP[t] > 1.10 * mean(GDP[t-4..t-1])`) -- closer to nonlinear
than the impulse-flag pair, and apparently beyond the JumpReLU SAE's
linear-threshold capacity at this dataset size. They're the natural
target for next-iteration SAE architecture work (gated SAEs,
multi-feature compositions, or supervised regime auxiliary heads).

### Final benchmark scoreboard

| tier              | best (best feed)         | comment                                          |
|-------------------|--------------------------|--------------------------------------------------|
| categorical (30)  | most at 1.000 (per-agent / attn) | matches sm-sae; recovered by any decent SAE |
| bucketed (7)      | 0.89-0.93 mAUC (per-agent / attn) | continuous-quantile labels                  |
| conjunctive (8)   | 0.97 mAUC (per-agent / attn)      | needs cross-agent attention                |
| regime (6)        | 0.86 mAUC (macro-feed v2)         | 3/6 fully recovered; 3/6 plateaued at 0.78  |

econ-sae is now a benchmark with a *known difficulty profile* across
tiers and a *known structural recipe* for each:

1. Train an **attention-enabled** world model on the multi-trajectory
   ensemble.
2. Train a **per-agent SAE** on its h1 to recover categorical /
   bucketed / conjunctive features (AUC 0.97 mean on conjunctive).
3. Train a **per-period (macro-feed) SAE** on macro + shock + impulse
   flags + GDP window + pooled h1 to recover regime features (AUC 0.86
   mean, 3/6 fully recovered).

That two-decoder recipe is the headline result of the project.

## Why econ-sae is *harder* than sm-sae

The Standard Model factorizes cleanly: every particle is a tensor product
of categorical features (charge × baryon × lepton × color × generation ×
chirality). SAE recovery on sm-sae works because those features are
independent axes the SAE can monosemantically discover.

econ-sae deliberately breaks that pattern:

- **Continuous × discrete mixes**: cohort tag (discrete) interacts with
  mpc (continuous) and wealth (continuous) so that "young, high-MPC,
  credit-constrained" is a genuine conjunctive feature with no clean
  tensor decomposition.
- **Correlated stochastic shocks**: TFP shocks load on a latent factor
  shared across sectors, plus sector-specific idiosyncratic noise. The
  SAE cannot recover them by reading any single coordinate.
- **Causal lag**: today's credit constraint manifests in next period's
  consumption. Features have a time-shifted causal structure.
- **Regime / phase features**: expansion vs. contraction is a macro-level
  state computed from a window of GDP; every agent in a contracting
  period gets the label, but the contraction is endogenous.
- **9 transaction kinds**: `wage`, `purchase` (× 3 sector subkinds),
  `tax`, `transfer`, `loan_origination`, `loan_repayment`, `interest`,
  `deposit_interest`, `dividend`. Each is a ground-truth label *and* a
  conserved-coord transformer.

## Layout

```
econ-sae/
├── econsae/
│   ├── __init__.py
│   ├── sectors.py              # cohort + goods-sector taxonomy
│   ├── embeddings.py           # agents as vectors in R^23 (14 conserved + 9 side)
│   ├── ground_truth.py         # feature vocabulary + (X, Y) builder for SAE alignment
│   ├── simulator/
│   │   ├── core.py             # multi-good, multi-cohort, credit-enabled SFC simulator
│   │   ├── shocks.py           # stochastic AR(1)-factor shock schedules
│   │   └── ensemble.py         # multi-trajectory runner
│   └── sae/
│       ├── models.py           # TopK, L1, JumpReLU SAEs (shared base class)
│       ├── train.py            # generic training loop w/ dead-neuron resampling
│       ├── data.py             # three feeds: raw, embedded, acts
│       ├── world_model.py      # per-agent MLP whose h1 activations feed the SAE
│       └── evaluation.py       # vectorized AUC alignment + per-tier breakdown
├── scripts/
│   ├── build_data.py           # generate data/econ_ensemble.npz
│   ├── train_world_model.py    # train the MLP; dump h1 acts under runs/
│   ├── train_all.py            # train 3 variants x 3 feeds; checkpoints to runs/
│   ├── evaluate.py             # AUC alignment + per-tier table; alignment_summary.json
│   └── polygram_demo.py        # stub (Polygram bridge lands next iteration)
├── data/econ_ensemble.npz      # 32 trajectories x 60 periods x 17 agents x 23 coords (~2.3 MB)
├── runs/                       # SAE checkpoints + eval (gitignored)
├── tests/test_smoke.py         # 10 passing tests covering simulator + shocks + ensemble + ground-truth
├── pyproject.toml
├── LICENSE
└── README.md
```

## Conserved quantities (the SAE ground-truth axes)

Each agent is a point in R^23 with two coordinate blocks:

| coord                   | block      | meaning                                          |
|-------------------------|------------|--------------------------------------------------|
| `money`                 | conserved  | deposit balance; sum across agents invariant     |
| `inv_food` `inv_services` `inv_durables` | conserved | per-sector inventory          |
| `debt_liab` `debt_asset` | conserved | every loan: lender +asset, debtor +liab           |
| `labor_supply` `labor_demand` | conserved | sum_supply = sum_demand                    |
| `goods_out_*` `goods_in_*` | conserved | per-sector flow identities                     |
| `productivity` `wage` `price` `mpc` `tax_rate` | side | agent parameters             |
| `sector` `subsector`    | side       | discrete categorical tags                        |
| `expectation` `credit_limit` | side  | one-period predictive state                      |

Identities verified by `check_conservation` (each residual ~1e-13):

```
sum_i money[i]                       = M_initial                    (closed economy)
sum_i (debt_liab - debt_asset)[i]    = 0                            (loans two-sided)
sum_i labor_supply[i]                = sum_i labor_demand[i]
sum_i goods_in_food[i]               = sum_i goods_out_food[i]      (and services, durables)
```

## Transactions = vertices

Nine transaction kinds, each tagged so it becomes a ground-truth feature
for SAE alignment:

| kind                | sender   | receiver   | side effect                       |
|---------------------|----------|------------|-----------------------------------|
| `wage`              | firm     | household  | labor_demand += hours; labor_supply += hours |
| `purchase` (×3 sec) | household| firm       | per-sector goods_in / goods_out / inventory  |
| `tax`               | household| government | pure money flow                  |
| `transfer`          | government| household | pure money flow                  |
| `loan_origination`  | bank     | debtor     | bank.debt_asset += L; debtor.debt_liab += L  |
| `loan_repayment`    | debtor   | bank       | debtor.debt_liab -= P; bank.debt_asset -= P  |
| `interest`          | debtor   | bank       | pure money flow                  |
| `deposit_interest`  | bank     | household  | pure money flow                  |
| `dividend`          | firm     | household  | pure money flow                  |

## Ground-truth feature vocabulary

51 features across four difficulty tiers (see `econsae/ground_truth.py`):

- **Hard categorical** (24): sector / cohort / firm-sector / txn-kind-present / shock-kind-present
- **Continuous bucketed** (7): debt-bucket-high, cash-poor/rich, mpc-high, leverage-high, etc.
- **Conjunctive** (8): the polysemantic-trap targets, e.g.
  `young_AND_indebted`, `prime_AND_high_cash`, `food_firm_low_inv`,
  `firm_AND_indebted_AND_high_inventory`, ...
- **Regime / phase** (6): expansion / contraction / high-leverage / high-rate / fiscal-active / monetary-active

In the default 32-trajectory × 60-period bundle, 34 of 51 features have
prevalence in [5%, 95%] (good AUC scoring range), with the rest deliberately
rare-event tail features.

## Install

```bash
git clone https://github.com/jascal/econ-sae.git
cd econ-sae
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the polygram bridge (Phase C, next iteration):

```bash
pip install -e ".[dev,polygram]"
```

## Quickstart

```python
from econsae.simulator.core import Economy, check_conservation
from econsae.simulator.shocks import draw_shock_schedule

econ = Economy.small(households_per_cohort=4, firms_per_sector=1)
sched = draw_shock_schedule(n_periods=60, seed=0)
traj = econ.rollout(60, shocks=sched.shocks)

print(traj.stack_states().shape)        # (60, 17, 23)
print(traj.macros[10])                   # GDP, C_food, C_services, C_durables, ...
print(check_conservation(traj))          # all residuals ~1e-13
```

Build the SAE-ready ensemble bundle:

```bash
python scripts/build_data.py
# -> data/econ_ensemble.npz (~2.3 MB) with X, Y, sample_index, vocab, states, macros, ...
```

Train the full pipeline:

```bash
python scripts/train_world_model.py    # ~30s: trains the predictive MLP, dumps h1 acts
python scripts/train_all.py            # ~15min: trains 9 SAEs (3 variants x 3 feeds)
python scripts/evaluate.py             # ~30s: AUC alignment + per-tier table
```

Per-(feed, variant) checkpoints land in `runs/{feed}__{variant}.pt`;
the full alignment matrix and per-tier metrics are written to
`runs/alignment_summary.json`.

## Roadmap

- **Phase 0.5**: multi-good, multi-cohort, credit-enabled simulator;
  stochastic shock schedules; ensemble runner; ground-truth feature
  matrix; 9 transaction kinds; exact SFC accounting.
- **Phase 1** (this iteration): SAE training + AUC alignment with per-tier
  breakdown. Three feeds (raw / random-projection embedded / world-model
  h1 activations), three SAE variants (TopK, L1, JumpReLU). Headline:
  categorical features cleanly recovered, conjunctive features partially
  recovered on the activation feed, regime features unrecoverable.
- **Phase 1.5**: vocabulary fix for 2 dead conjunctive features;
  scale experiment (128 traj x 100 periods, DeepWorldModel); attention
  experiment (AttnWorldModel) that lifted conjunctive mean AUC from 0.84
  to 0.97.
- **Phase 1.6**: TemporalWorldModel (AttnWorldModel + cross-period GRU) —
  falsified "architecture alone unlocks regime features". Diagnosed as a
  training-objective issue.
- **Phase 1.7**: sentiment-driven MPC in the simulator — falsified the
  "Markovian simulator" diagnosis. Agent state already encodes most
  history through accumulated-stock coords.
- **Phase 1.8**: continuous regime-intensity correlation analysis —
  falsified the "binary threshold mismatch" diagnosis. Pointed to a
  structural mismatch: per-agent SAE vs per-period regime labels.
- **Phase 1.9**: macro-feed SAE — confirmed the structural diagnosis.
  `phase:high_leverage` jumped to AUC 0.974 the moment we decoded
  per-period samples. Regime tier mAUC climbed +0.125 overall.
- **Phase 2.0** (latest): macro-feed v2 — closed the impulse-regime gap.
  Added explicit GDP[t-4..t] window and monetary/fiscal impulse flags
  to the per-period input, with z-score normalization across mixed-scale
  dims. `phase:monetary_active` and `phase:fiscal_active` both crossed
  AUC 0.99. Regime tier mAUC climbed to 0.864; 3/6 regime features fully
  recovered. The remaining 3 (expansion / contraction / high_rate)
  plateaued at ~0.79 — they're threshold-on-continuous-window features
  that need richer SAE compositions to fully recover.
- **Phase 3**: Polygram bridge (Dictionary + InterferenceSweep across
  the multi-decoder benchmark); gated SAE / supervised regime head to
  close the remaining 3 threshold-on-window features; input-output
  firm network; central-bank Taylor rule; calibration to historical
  macro data.

## Acknowledgements

Sibling to [sm-sae](https://github.com/jascal/sm-sae); pattern, conventions,
and the conserved-features-as-ground-truth methodology are copied directly.
Stock-flow-consistent accounting from Godley & Lavoie. CES expenditure
allocation from Dixit-Stiglitz.
