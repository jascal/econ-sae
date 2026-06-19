# Phase 11 — The held-out label-free recovery test (regime): the gap is allocation, not presence

*The experiment `RESEARCH_MANIFESTO.md` Reckoning #1 (the unsupervised-ceiling
contradiction) and the workspace `SUPERVISION_DEPENDENCE.md` demand but leave out of
scope. Reproduce: `scripts/regime_granularity_experiment.py`,
`scripts/regime_sae_allocation_experiment.py`,
`scripts/regime_allocation_followups.py` (summaries in `runs/regime_*.json`).*

## The question

The `regime` tier is the program's single largest supervision gap: unsupervised
mAUC 0.605 / **cov95 0.00**, supervised 0.991 / 1.00 (Δ +0.386). If the answer key
is needed *while training the SAE*, the substrate may be testing supervised probing,
not unsupervised discovery. The decisive test: **does an unsupervised SAE recover
regime when scored at matched granularity, without the labels?**

## Design

Isolate the **label-free granularity** knob on the *same* unsupervised
`TemporalWorldModel` `h1` (next-state objective only — **no regime labels** in WM or
SAE training; labels used only to *score*). Per condition, report both the
unsupervised JumpReLU SAE recovery and a **ridge-LDA linear-probe ceiling** — a
label-using diagnostic of how much regime is *linearly present* — to separate two
very different failure modes:

- probe **high** & SAE **low** → regime is present, the SAE doesn't surface it
  (an objective/allocation limit, not a substrate absence);
- probe **low** & SAE **low** → regime isn't in the representation at that granularity.

Conditions, increasingly granularity-matched to the agent-invariant,
window-aggregated regime tier (3 seeds, 48 trajs × 80 periods):

| condition (label-free) | dim | **SAE** mAUC / cov95 | **probe** mAUC / cov95 |
|---|--:|--|--|
| C0 raw per-(agent,period) — the published floor | 192 | 0.585 / **0%** | 0.769 / 0% |
| C1 agent-pooled (regime is agent-invariant) | 192 | 0.723 / 0% | 0.893 / 50% |
| C2 + window **mean**-pool | 192 | 0.668 / 0% | 0.794 / 33% |
| C3 + window **concat** | 960 | 0.716 / 0% | 0.918 / **72%** |
| Cref macro-feed (engineered, label-free) | 223 | 0.838 / **17%** | 0.987 / **100%** |

## Result — two distinct gaps

**1. Presence gap — DISSOLVES label-free.** Matching granularity (agent-pool +
window-concat), with no labels and no feature engineering, lifts the probe ceiling
0.77 → 0.92 (cov95 0 → 72%); the engineered feed reaches 0.99 / 100%. **Supervision
is not needed to make regime decodable.** *Methodological nugget:* temporal
**mean**-pool **hurts** (washes out the per-period signal); **concat** that preserves
the trailing window helps — the granularity op must *preserve*, not average.

**2. Allocation gap — does NOT close under any label-free SAE recipe.** Even where
the probe proves regime is 72% present, the unsupervised SAE surfaces **0%**
(pure-granularity) / **17–33%** (engineered) at cov95. A full label-free SAE-side
sweep moves it by essentially nothing:

| feed | width 256→1024 | TopK k8/16 | PCA-whiten | `l0` 1e-4→1e-2 |
|---|---|---|---|---|
| C3 (pure granularity) | 0% | 0% | 0% (hurts mAUC) | 0% (flat) |
| Cref (engineered) | 17%→33%, plateaus | ≤17% | 0% (hurts) | 33% (flat) |

The gap is invariant to **capacity *and* sparsity** — so it is a property of the
reconstruction **objective's incentives**, not of tuning.

**3. Locus control.** Run the regime-*supervised* WM's `h1` through the *same* feeds
+ *same* SAE: recovery jumps from **0% → 50%** at matched granularity (and 0 → 17%
even raw) — only the WM-level supervision differs (single-head; the published
dual-head reaches 6/6). So supervision's role is **WM-level representation shaping**
(it makes regime a higher-variance, allocatable direction), and it **compounds** with
the label-free granularity lever — *not* scoring-time answer injection.

## Verdict

The "supervised probing with extra steps" worry is **half-dissolved and precisely
relocated.** Regime is recoverable label-free *as a probe* — presence is not the
issue — but the vanilla unsupervised SAE objective will not allocate a sharp latent
to a present-but-low-variance feature; that is what supervision (or, partly, feature
engineering) provides. In one line:

> **Compression is variance-greedy; meaning is variance-cheap.**

An SAE spends its latents to cut reconstruction error (variance-weighted), so it
models the loud directions; a feature like regime (low-prevalence binary phases) is
cheap to ignore even though it's sitting right there linearly. This unifies `econ
regime` with `bio motif`'s two-lever result (occurrence-scoring ≈ the
presence/granularity fix; supervision ≈ the allocation fix) — **two substrates, one
mechanism** — and ties to the forge-tax / cosine-vs-capability findings (all =
compression under-serving variance-cheap structure).

## The decisive label-free test: a variance-equalized objective — and why it fails

The verdict pointed at one route that could close the allocation gap label-free: an
objective that doesn't price directions by variance. We tested it
(`scripts/regime_whitened_loss_experiment.py`) — reconstruct in **whitened space**,
loss = ‖M·(x − x̂)‖² with M = (Σ + λI)^(−α), so α=0 is the standard-L2 control and
α=0.5 fully equalizes the principal directions. **Crucially this whitens the *loss*,
not the *input*** — the encoder still sees the natural `x` (input-whitening, already
tried, *hurt*); only the error metric is reweighted. Result (2 seeds):

| α (whiten strength) | C3 cov95 (probe 67%) | Cref cov95 (probe 100%) |
|--:|--|--|
| 0.0 (L2 control) | 0% | 33% |
| 0.25 | 0% | 25% |
| 0.5 | 0% | 17% |
| 0.75 | 0% | 0% |
| 1.0 | 0% | 0% |

**It does not close the gap — and beyond mild strength it strictly *hurts* (Cref
33%→0% monotone).** This is the *principled* failure that sharpens the whole result:
equalizing variance cannot separate a low-variance **signal** (regime) from
low-variance **noise**, so the freed budget is spent chasing noise. The
discrimination "this quiet direction is meaningful, that one is noise" is exactly
what supervision injects — which is why no purely *variance*-based, label-free move
can work. In one line: *you can't buy meaning back by re-pricing variance, because
meaning and noise are equally cheap.*

## Honest limits & the (refined) open question

- Single substrate (econ); the **`bio motif` replication** is the generalization test
  that would promote this to a cross-substrate law.
- Linear / probe-present features and L2-reconstruction SAEs specifically.
- The supervised control is the single-head WM (3/6), not the published dual-head (6/6).
- **No label-free move tested** — granularity, width 256→1024, TopK, input-whitening,
  sparsity `l0` 1e-4→1e-2, *or* the variance-equalized loss — closes the allocation
  gap; only supervision (representation shaping) does. The remaining open route is no
  longer a *variance*-aware objective but a **structure-aware** one: a
  non-reconstruction term that detects *predictable / mutually-informative* low-variance
  directions without labels (e.g. a predictive-coding or coverage/diversity objective,
  or polygram-style clustered encodings). Achievability is OPEN, not "impossible" — but
  the search space is now precisely characterised.
