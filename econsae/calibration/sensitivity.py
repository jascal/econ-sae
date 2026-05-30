"""Morris elementary-effects screening of the calibration objective.

Identifiability (`multistart_calibrate`) answers "where do independent fits
land?" -- a property of the *inverse* problem. This module answers the
complementary *forward* question: "how much does each knob move the
moments?" A parameter the moments barely respond to is, by construction,
unidentifiable -- so the Morris ranking explains *why* a knob is loose.

The Morris method (Morris 1991; mu* refinement, Campolongo 2007) is a cheap
global screening: for each parameter it averages many one-at-a-time finite
differences ("elementary effects", EE) taken from random base points across
the parameter hypercube. Per parameter we report:

  mu      mean EE            -- signed average influence
  mu_star mean |EE|          -- magnitude of influence (the ranking metric)
  sigma   std of EE          -- variation across the space => nonlinearity
                                and/or interaction with other parameters

Cost is r * (k+1) objective evaluations for k parameters and r trajectories
(e.g. k=8, r=10 -> 88 evals) -- trivial given the fast simulator. The same
EE bookkeeping also yields a (param x moment) mu* matrix, which shows which
knob drives which moment -- directly useful for choosing an *identifying*
moment when a parameter is loose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from econsae.calibration.config import SimConfig
from econsae.calibration.moments import load_targets, moment_distance
from econsae.calibration.optimize import DEFAULT_PARAM_BOUNDS, _evaluate_moments


def _morris_trajectory(k: int, p: int, delta: float, rng) -> np.ndarray:
    """One Morris trajectory: a (k+1, k) matrix in the unit hypercube where
    consecutive rows differ in exactly one coordinate by +/- delta.

    Canonical Morris/Saltelli construction:
        B*  =  ( 1 . x*  +  (delta/2)[(2B - 1) D* + 1] ) P*
    with B strictly-lower-triangular ones, D* a random +/-1 diagonal, P* a
    random permutation, and x* a random base point on the level grid chosen
    so that every coordinate stays in [0, 1] after its +/- delta step.
    """
    levels = np.linspace(0.0, 1.0, p)
    allowed = levels[levels <= 1.0 - delta + 1e-9]      # keep x*+delta in [0,1]
    xstar = rng.choice(allowed, size=k)
    B = np.tril(np.ones((k + 1, k)), -1)                # strictly lower triangular
    Dstar = np.diag(rng.choice([-1.0, 1.0], size=k))
    Jk = np.ones((k + 1, k))
    one = np.ones((k + 1, 1))
    Pstar = np.eye(k)[:, rng.permutation(k)]
    Bstar = (one @ xstar[None, :] + (delta / 2.0) * ((2 * B - Jk) @ Dstar + Jk)) @ Pstar
    return Bstar


@dataclass
class MorrisResult:
    param_names: list[str]
    bounds: dict[str, tuple[float, float]]
    moment_keys: list[str]
    r: int
    p: int
    delta: float
    seeds: tuple[int, ...]
    n_traj: int
    n_periods: int
    n_evals: int
    mu: dict[str, float] = field(default_factory=dict)
    mu_star: dict[str, float] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)
    mu_star_by_moment: dict[str, dict[str, float]] = field(default_factory=dict)

    def ranking_table(self) -> list[dict]:
        """Params sorted by mu* (objective influence), descending."""
        rows = [{
            "param": n,
            "mu": self.mu[n],
            "mu_star": self.mu_star[n],
            "sigma": self.sigma[n],
        } for n in self.param_names]
        rows.sort(key=lambda r: r["mu_star"], reverse=True)
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return rows

    def top_driver_per_moment(self) -> dict[str, str]:
        """For each moment, the parameter with the largest mu* on it."""
        out: dict[str, str] = {}
        for m in self.moment_keys:
            out[m] = max(self.param_names, key=lambda n: self.mu_star_by_moment[n][m])
        return out

    def to_report(self) -> dict:
        return {
            "method": "morris_elementary_effects",
            "r_trajectories": self.r,
            "p_levels": self.p,
            "delta": self.delta,
            "seeds": list(self.seeds),
            "n_traj": self.n_traj,
            "n_periods": self.n_periods,
            "n_evals": self.n_evals,
            "param_names": self.param_names,
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "ranking": self.ranking_table(),
            "mu_star_by_moment": self.mu_star_by_moment,
            "top_driver_per_moment": self.top_driver_per_moment(),
            "notes": (
                "mu_star ranks objective influence; low mu_star => the moments "
                "barely respond => the knob is unidentifiable. sigma high => "
                "nonlinear/interacting. mu_star_by_moment shows which knob drives "
                "which moment (pick an identifying moment for a loose knob from its "
                "column)."
            ),
        }


def morris_screening(
    targets_path: str,
    *,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    r: int = 10,
    p: int = 4,
    n_traj: int = 8,
    n_periods: int = 60,
    seeds: tuple[int, ...] = (0, 1, 2),
    seed: int = 0,
) -> MorrisResult:
    """Run Morris EE screening of the moment-distance objective.

    Elementary effects are computed in the [0,1]-scaled parameter space (so
    knobs of different bound ranges are directly comparable). The objective is
    seed-averaged over `seeds` (held fixed), matching `calibrate`, so the
    surface is deterministic.
    """
    target, weights, scales, _ = load_targets(targets_path)
    bounds = dict(param_bounds or DEFAULT_PARAM_BOUNDS)
    names = list(bounds)
    k = len(names)
    lo = np.array([bounds[n][0] for n in names], dtype=np.float64)
    hi = np.array([bounds[n][1] for n in names], dtype=np.float64)
    moment_keys = list(target)

    if p % 2 != 0:
        raise ValueError("Morris p (levels) must be even")
    delta = p / (2.0 * (p - 1))
    rng = np.random.default_rng(seed)
    base = SimConfig.default()

    ee_obj: dict[str, list[float]] = {n: [] for n in names}
    ee_mom: dict[str, dict[str, list[float]]] = {n: {m: [] for m in moment_keys}
                                                 for n in names}
    n_evals = 0

    for _ in range(r):
        Bstar = _morris_trajectory(k, p, delta, rng)            # (k+1, k) scaled
        f_obj: list[float] = []
        f_mom: list[dict] = []
        for row in Bstar:
            actual = lo + row * (hi - lo)
            cfg = base.with_overrides(**dict(zip(names, actual)))
            moments = _evaluate_moments(cfg, n_traj, n_periods, seeds)
            n_evals += 1
            f_obj.append(moment_distance(moments, target, weights, scales))
            f_mom.append(moments)
        for j in range(k):
            diff = Bstar[j + 1] - Bstar[j]
            c = int(np.argmax(np.abs(diff)))                    # the coord that moved
            step = float(diff[c])                               # +/- delta (scaled)
            nm = names[c]
            ee_obj[nm].append((f_obj[j + 1] - f_obj[j]) / step)
            for m in moment_keys:
                ee_mom[nm][m].append((f_mom[j + 1][m] - f_mom[j][m]) / step)

    res = MorrisResult(
        param_names=names, bounds=bounds, moment_keys=moment_keys,
        r=r, p=p, delta=delta, seeds=tuple(seeds),
        n_traj=n_traj, n_periods=n_periods, n_evals=n_evals,
    )
    for n in names:
        arr = np.array(ee_obj[n], dtype=np.float64)
        res.mu[n] = float(arr.mean()) if arr.size else 0.0
        res.mu_star[n] = float(np.abs(arr).mean()) if arr.size else 0.0
        res.sigma[n] = float(arr.std()) if arr.size else 0.0
        res.mu_star_by_moment[n] = {
            m: float(np.abs(ee_mom[n][m]).mean()) if ee_mom[n][m] else 0.0
            for m in moment_keys
        }
    return res
