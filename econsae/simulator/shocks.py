"""Stochastic shock schedules for the multi-trajectory ensemble.

The point of the shock generator is to produce a rich, partially-correlated
training distribution where each trajectory exposes the simulator to a
different combination of shocks. The schedule additionally records, for
every period, the *set of shock kinds* active in that period --- these are
ground-truth labels used in SAE alignment scoring.

SHOCK KINDS (each is a ground-truth feature axis for the SAE)
-------------------------------------------------------------
    tfp_<sector>           : sector-specific productivity shock (food/services/durables)
    sentiment_<cohort>     : cohort-specific MPC shock (young/prime/retiree)
    monetary               : policy-rate impulse
    fiscal                 : fiscal-transfer impulse
    productivity_aggregate : economy-wide TFP shock (rare, big)

Correlation structure
---------------------
TFP shocks across sectors are correlated through a single latent factor
plus sector-specific idiosyncratic noise. Sentiment shocks similarly load
on a common consumer-confidence factor. Monetary and fiscal shocks are
sampled as a sparse Bernoulli-Gaussian process (mostly zero, occasional
impulses). The latent-factor structure means that no clean feature
factorization recovers the shock kinds via element-wise inspection: the
SAE has to disentangle the latent factors. This is intentional --- it is
the difficulty bump asked for in the design brief.

USAGE
-----
    from econsae.simulator.shocks import draw_shock_schedule

    sched = draw_shock_schedule(n_periods=60, seed=0)
    traj = Economy.small().rollout(60, shocks=sched.shocks)
    # sched.kinds[t] is the set of shock kinds active in period t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from econsae.sectors import GOODS_SECTORS, HH_COHORTS


@dataclass
class ShockSchedule:
    """A stochastic shock schedule with per-period kind labels."""
    shocks: list[dict] = field(default_factory=list)
    kinds: list[set[str]] = field(default_factory=list)        # active shock kinds per period
    metadata: dict = field(default_factory=dict)


def draw_shock_schedule(
    n_periods: int,
    seed: int = 0,
    # TFP shocks: AR(1) latent factor with cross-sector loadings
    tfp_ar: float = 0.7,
    tfp_factor_vol: float = 0.04,
    tfp_idio_vol: float = 0.02,
    # Sentiment shocks (affect MPC): AR(1) common factor + cohort-specific
    sentiment_ar: float = 0.6,
    sentiment_factor_vol: float = 0.03,
    sentiment_idio_vol: float = 0.015,
    # Monetary: rare impulses to the policy rate
    monetary_prob: float = 0.10,
    monetary_step: float = 0.01,
    base_interest_rate: float = 0.02,
    # Fiscal: occasional transfer impulses on top of the balanced-budget baseline
    fiscal_prob: float = 0.08,
    fiscal_impulse: float = 0.4,
    # Aggregate TFP: very rare, large
    agg_tfp_prob: float = 0.02,
    agg_tfp_size: float = 0.10,
) -> ShockSchedule:
    """Draw a shock schedule of length n_periods.

    All component processes are seeded by `seed`; rerunning with the same
    seed reproduces the exact same schedule. Shocks compose additively on
    multiplicative coords (productivity_mult_<sec>) and additively
    elsewhere (mpc_add_<cohort>, transfer_per_hh, interest_rate). Shock
    kinds per period are recorded for the ground-truth label module.
    """
    rng = np.random.default_rng(seed)

    # --- TFP factor & sector-specific paths ----
    tfp_factor = np.zeros(n_periods)
    for t in range(1, n_periods):
        tfp_factor[t] = tfp_ar * tfp_factor[t - 1] + rng.normal(0, tfp_factor_vol)
    sector_loadings = np.array([0.3, 0.7, 1.2])  # food / services / durables: durables most procyclical
    tfp_paths = {
        sec: tfp_factor * sector_loadings[i] + rng.normal(0, tfp_idio_vol, n_periods)
        for i, sec in enumerate(GOODS_SECTORS)
    }

    # --- Sentiment factor & cohort-specific paths ----
    senti_factor = np.zeros(n_periods)
    for t in range(1, n_periods):
        senti_factor[t] = sentiment_ar * senti_factor[t - 1] + rng.normal(0, sentiment_factor_vol)
    cohort_loadings = np.array([1.3, 1.0, 0.5])  # young / prime / retiree: young most sentimental
    sentiment_paths = {
        c: senti_factor * cohort_loadings[i] + rng.normal(0, sentiment_idio_vol, n_periods)
        for i, c in enumerate(HH_COHORTS)
    }

    # --- Monetary impulses ----
    monetary_impulse = rng.random(n_periods) < monetary_prob
    monetary_sign = rng.choice([-1.0, 1.0], size=n_periods)
    monetary_step_path = monetary_impulse.astype(float) * monetary_sign * monetary_step

    # --- Fiscal impulses ----
    fiscal_impulse_mask = rng.random(n_periods) < fiscal_prob
    fiscal_path = fiscal_impulse_mask.astype(float) * fiscal_impulse

    # --- Aggregate TFP rare shocks ----
    agg_tfp_mask = rng.random(n_periods) < agg_tfp_prob
    agg_tfp_sign = rng.choice([-1.0, 1.0], size=n_periods)
    agg_tfp_path = agg_tfp_mask.astype(float) * agg_tfp_sign * agg_tfp_size

    # --- Compose into per-period shock dicts and kind labels ----
    sched = ShockSchedule(metadata={"seed": seed, "n_periods": n_periods})
    cur_rate = base_interest_rate
    for t in range(n_periods):
        shock: dict = {}
        kinds: set[str] = set()

        for sec in GOODS_SECTORS:
            s = float(tfp_paths[sec][t])
            if abs(s) > 1e-6:
                shock[f"productivity_mult_{sec}"] = float(np.exp(s))
                kinds.add(f"tfp_{sec}")

        if abs(agg_tfp_path[t]) > 1e-6:
            shock["productivity_mult"] = float(np.exp(agg_tfp_path[t]))
            kinds.add("productivity_aggregate")

        for c in HH_COHORTS:
            s = float(sentiment_paths[c][t])
            if abs(s) > 1e-6:
                shock[f"mpc_add_{c}"] = s
                kinds.add(f"sentiment_{c}")

        if monetary_impulse[t]:
            cur_rate = max(0.0, cur_rate + float(monetary_step_path[t]))
            shock["interest_rate"] = cur_rate
            kinds.add("monetary")

        if fiscal_impulse_mask[t]:
            shock["transfer_per_hh"] = float(fiscal_path[t])
            kinds.add("fiscal")

        sched.shocks.append(shock)
        sched.kinds.append(kinds)

    return sched


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sched = draw_shock_schedule(n_periods=30, seed=42)
    print(f"schedule with {len(sched.shocks)} periods, seed={sched.metadata['seed']}")
    kind_counts: dict[str, int] = {}
    for ks in sched.kinds:
        for k in ks:
            kind_counts[k] = kind_counts.get(k, 0) + 1
    print("shock-kind occurrences across periods:")
    for k in sorted(kind_counts):
        print(f"  {k:<28s} {kind_counts[k]}")
    print("\nfirst 3 periods:")
    for t in range(3):
        print(f"  t={t}: kinds={sorted(sched.kinds[t])}")
        print(f"         shock={ {k:round(v,4) for k,v in sched.shocks[t].items()} }")
