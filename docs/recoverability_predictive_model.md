# Phase 12 — A predictive recoverability model: presence = Fisher, allocation = variance-share

*The structural-match law, made quantitative and predictive. Reproduce:
`scripts/regime_recoverability_theory.py` (summary in
`runs/regime_recoverability_theory_summary.json`). Formalised + kernel-proved in
i-orca `examples/recoverability` (`present_not_allocated`).*

## The model

A ground-truth feature, read off a representation, is characterised by two scalar,
*separately measurable, scale-free* functionals — no SAE training needed to predict
recovery:

- **Presence** (can a probe read it?) = **Fisher SNR**  `‖μ₊−μ₋‖²_{Σ_w⁻¹}`  — detection
  theory (the optimal linear detector's separation).
- **Allocation** (does the unsupervised SAE surface it?) = **variance-share**
  `p(1−p)‖μ₊−μ₋‖² / tr(Σ)` — rate–distortion / reverse water-filling (the between-class
  variance the coder would pay to reconstruct; dropped below a budget-set level).

They are linked only through the direction's variance, so they decouple — a feature
can be maximally detectable yet have negligible reconstruction-relevance.

## Validation (raw per-agent feed, all 52 features, 3 seeds)

Across all tiers var_share and Fisher **co-vary**, so both correlate with SAE recovery
(raw Spearman +0.89 and +0.97). The confound-free tests:

**Presence is Fisher-driven, not variance** (partial Spearman, controlling for the
other):

| | → SAE_AUC (allocation) | → probe_AUC (presence) |
|---|--:|--:|
| var_share \| Fisher | +0.40 | +0.21 |
| Fisher \| var_share | +0.85 | **+0.97** |

Presence (probe) is essentially pure Fisher (+0.97); var_share adds little (+0.21).

**Allocation is variance-share-driven — once presence is controlled.** Within the
`regime` tier (all features made *present* by granularity matching, so Fisher is high
across them), the axes decouple cleanly:

> Spearman(var_share → SAE) = **+0.94**   vs   Spearman(Fisher → SAE) = **+0.37**

**The exemplar** (`fiscal_active`, C3 feed): Fisher **168.7** (by far the most
*detectable* regime feature) yet var_share **0.0006** (least *reconstructible*) and SAE
recovery **0.67** (poorly *recovered*). Detectability ≠ recoverability, as a single
data point.

## Per-tier means (raw feed)

| tier | var_share | Fisher | SAE_AUC |
|---|--:|--:|--:|
| categorical | 0.054 | 378 | 0.71 |
| bucketed | 0.087 | 40 | 0.91 |
| conjunctive | 0.031 | 69 | 0.96 |
| regime | **0.0018** | 3.8 | 0.59 |

Regime has the lowest variance-share by ~20×, and the lowest recovery — exactly the
rate–distortion prediction.

## Honest scope

The *cross-tier* partial is confounded (categorical features have both high Fisher and
high var_share, so Fisher tracks recovery there); the clean, decisive test is the
*presence-controlled* regime-internal correlation (+0.94 vs +0.37) plus the presence
partial (+0.97) and the exemplar. The formal model (i-orca) proves the decoupling is
real (`present_not_allocated`); this measures that it happens on a real substrate.
