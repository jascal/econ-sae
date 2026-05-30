"""Scale-invariant macro moments + a weighted moment-matching objective.

The simulator's GDP / money / price are in arbitrary synthetic units, so
calibration targets only quantities that are invariant to that scale:
growth *rates*, volatilities, autocorrelations, ratios, and event
frequencies. Levels are deliberately excluded.

`compute_moments` takes per-key macro arrays shaped `(n_traj, T)` -- the same
layout `build_data.py` packs into the ensemble bundle -- and returns a flat
dict of moment_name -> value. `moment_distance` scores a simulated moment
dict against a target dict with per-moment weights.
"""

from __future__ import annotations

import json

import numpy as np

# Canonical recession rule -- MUST match `phase:contraction` in
# `econsae/ground_truth.py:_phase_features_for_period` (trailing-5 mean,
# 0.90 threshold) so the calibration target and the GT label agree.
RECESSION_WINDOW = 5
RECESSION_RATIO = 0.90

MOMENT_KEYS = (
    "gdp_growth_mean",
    "gdp_growth_vol",
    "gdp_growth_ac1",
    "recession_freq",
    "fedfunds_mean",
    "fedfunds_vol",
    "inflation_mean",
    "inflation_vol",
    "debt_to_money",
)

_EPS = 1e-9


def _log_growth(series: np.ndarray) -> np.ndarray:
    """Per-trajectory log-growth diffs, flattened. `series` is (n_traj, T)."""
    s = np.maximum(np.asarray(series, dtype=np.float64), _EPS)
    return np.diff(np.log(s), axis=1).ravel()


def _lag1_autocorr_per_traj(series: np.ndarray) -> float:
    """Mean over trajectories of the lag-1 autocorrelation of log-growth."""
    s = np.maximum(np.asarray(series, dtype=np.float64), _EPS)
    g = np.diff(np.log(s), axis=1)            # (n_traj, T-1)
    acs: list[float] = []
    for row in g:
        if row.size < 2 or np.std(row) < _EPS:
            continue
        acs.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
    return float(np.mean(acs)) if acs else 0.0


def _recession_freq(gdp: np.ndarray) -> float:
    """Fraction of (traj, period) marked contraction by the GT rule.

    Mirrors `_phase_features_for_period`: window = GDP[t-5..t-1]; a period
    is a contraction when GDP[t] < 0.90 * mean(window). t=0 (empty window)
    is skipped, matching the label builder.
    """
    g = np.asarray(gdp, dtype=np.float64)
    n_traj, T = g.shape
    hits = 0
    total = 0
    for ti in range(n_traj):
        for t in range(1, T):
            lo = max(0, t - RECESSION_WINDOW)
            window = g[ti, lo:t]
            if window.size == 0:
                continue
            total += 1
            if g[ti, t] < RECESSION_RATIO * float(np.mean(window)):
                hits += 1
    return hits / total if total else 0.0


def compute_moments(macros: dict[str, np.ndarray]) -> dict[str, float]:
    """Scale-invariant summary moments of an ensemble's macro series.

    `macros` must provide (n_traj, T) arrays under keys: GDP, interest_rate,
    price_level, debt_outstanding, money_stock.
    """
    gdp = np.asarray(macros["GDP"], dtype=np.float64)
    rate = np.asarray(macros["interest_rate"], dtype=np.float64)
    price = np.asarray(macros["price_level"], dtype=np.float64)
    debt = np.asarray(macros["debt_outstanding"], dtype=np.float64)
    money = np.asarray(macros["money_stock"], dtype=np.float64)

    gdp_growth = _log_growth(gdp)
    infl = _log_growth(price)
    dtm = debt / np.maximum(money, _EPS)

    return {
        "gdp_growth_mean": float(np.mean(gdp_growth)),
        "gdp_growth_vol": float(np.std(gdp_growth)),
        "gdp_growth_ac1": _lag1_autocorr_per_traj(gdp),
        "recession_freq": _recession_freq(gdp),
        "fedfunds_mean": float(np.mean(rate)),
        "fedfunds_vol": float(np.std(rate)),
        "inflation_mean": float(np.mean(infl)),
        "inflation_vol": float(np.std(infl)),
        "debt_to_money": float(np.mean(dtm)),
    }


def moment_distance(
    sim: dict[str, float],
    target: dict[str, float],
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> float:
    """Weighted normalized squared error between simulated and target moments.

    Each residual is divided by a per-moment `scale` (a tolerance: "a
    deviation this large = one unit of badness"). Explicit scales keep
    moments commensurable -- a moment whose simulator value lives on a
    different absolute scale than its real-world counterpart (e.g.
    `debt_to_money`) can't swamp a small growth rate, and an unreachable
    moment contributes a bounded amount rather than dominating. When no
    scale is given for a moment, the target magnitude is used (relative
    error). Moments with weight 0 (or absent from `target`) are skipped.
    """
    weights = weights or {}
    scales = scales or {}
    total = 0.0
    for k, tgt in target.items():
        if k not in sim:
            continue
        w = float(weights.get(k, 1.0))
        if w == 0.0:
            continue
        scale = float(scales.get(k, max(abs(float(tgt)), 1e-6)))
        scale = scale if abs(scale) > 1e-12 else 1e-6
        total += w * ((float(sim[k]) - float(tgt)) / scale) ** 2
    return float(total)


def macros_from_ensemble(ens) -> dict[str, np.ndarray]:
    """Extract the (n_traj, T) macro arrays `compute_moments` needs.

    Avoids a hard import of the simulator (duck-typed `ens.trajectories`),
    keeping this module torch/simulator-free at import time.
    """
    keys = ("GDP", "interest_rate", "price_level", "debt_outstanding", "money_stock")
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.array(
            [[m[k] for m in t.macros] for t in ens.trajectories], dtype=np.float64
        )
    return out


def load_targets(
    path: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict]:
    """Read the vendored targets JSON -> (moments, weights, scales, provenance)."""
    with open(path) as f:
        doc = json.load(f)
    moments = {k: float(v) for k, v in doc.get("moments", {}).items()}
    weights = {k: float(v) for k, v in doc.get("weights", {}).items()}
    scales = {k: float(v) for k, v in doc.get("scales", {}).items()}
    provenance = doc.get("provenance", {})
    return moments, weights, scales, provenance
