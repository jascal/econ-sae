"""Derivative-free calibration of `SimConfig` to historical macro moments.

The objective runs the simulator (non-differentiable, stochastic) and scores
its moments against vendored targets. Stochasticity is tamed by evaluating
each candidate over a *fixed* set of seeds and averaging the moment vector,
so the objective is deterministic in the parameters -- a hard requirement for
the local optimizers to converge.

Default optimizer is `scipy.optimize.differential_evolution` (global,
bounded, derivative-free). A numpy-only `random` search + coordinate refine
is provided as a zero-extra-dependency fallback; scipy is imported lazily so
this module loads even if scipy is absent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from econsae.calibration.config import SimConfig
from econsae.calibration.moments import (
    compute_moments,
    macros_from_ensemble,
    moment_distance,
)

# The calibratable subset and their bounds. Structural flags (taylor_rule,
# io_network) are intentionally excluded, as are the rare aggregate-TFP
# shocks (agg_tfp_*), which are weakly identified by these moments. This set
# favors the most identifiable knobs: volatilities (drive growth/inflation
# vol), persistence (autocorr), impulse probabilities, and the rate level.
DEFAULT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "tfp_ar": (0.0, 0.95),
    "tfp_factor_vol": (0.005, 0.15),
    "tfp_idio_vol": (0.002, 0.08),
    "sentiment_factor_vol": (0.005, 0.12),
    "monetary_prob": (0.02, 0.40),
    "monetary_step": (0.002, 0.03),
    "base_interest_rate": (0.005, 0.08),
    "fiscal_prob": (0.02, 0.30),
}


@dataclass
class CalibrationResult:
    config: SimConfig
    objective: float
    baseline_objective: float          # objective at SimConfig.default()
    sim_moments: dict[str, float]
    target_moments: dict[str, float]
    weights: dict[str, float]
    scales: dict[str, float]
    param_names: list[str]
    bounds: dict[str, tuple[float, float]]
    seeds: tuple[int, ...]
    method: str
    n_traj: int
    n_periods: int
    n_evals: int
    trace: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    def moment_table(self) -> list[dict]:
        """Rows of {moment, target, sim, weight, scale, z} for reporting."""
        rows = []
        for k, tgt in self.target_moments.items():
            sim = self.sim_moments.get(k, float("nan"))
            scale = float(self.scales.get(k, max(abs(tgt), 1e-6)))
            rows.append({
                "moment": k,
                "target": tgt,
                "sim": sim,
                "weight": self.weights.get(k, 1.0),
                "scale": scale,
                "z": (sim - tgt) / (scale if abs(scale) > 1e-12 else 1e-6),
            })
        return rows

    def to_report(self) -> dict:
        return {
            "objective": self.objective,
            "baseline_objective": self.baseline_objective,
            "improvement": self.baseline_objective - self.objective,
            "method": self.method,
            "seeds": list(self.seeds),
            "n_traj": self.n_traj,
            "n_periods": self.n_periods,
            "n_evals": self.n_evals,
            "wall_seconds": self.wall_seconds,
            "param_names": self.param_names,
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "fitted_config": self.config.to_dict(),
            "moment_table": self.moment_table(),
            "trace": self.trace,
        }


def _evaluate_moments(cfg: SimConfig, n_traj: int, n_periods: int,
                      seeds: tuple[int, ...]) -> dict[str, float]:
    """Average the moment vector over a fixed seed set (variance reduction)."""
    # Local import: avoids any chance of an import cycle at package load.
    from econsae.simulator.ensemble import generate_ensemble

    per_seed = []
    for s in seeds:
        ens = generate_ensemble(
            n_trajectories=n_traj, n_periods=n_periods, seed=s, sim_config=cfg,
        )
        per_seed.append(compute_moments(macros_from_ensemble(ens)))
    return {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}


def calibrate(
    targets_path: str,
    *,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    n_traj: int = 16,
    n_periods: int = 80,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    method: str = "differential_evolution",
    maxiter: int = 40,
    popsize: int = 12,
    de_seed: int = 0,
) -> CalibrationResult:
    """Fit the calibratable `SimConfig` params to the vendored target moments.

    `method`: "differential_evolution" (scipy, default), "nelder-mead"
    (scipy local polish from the bound midpoint), or "random" (numpy-only
    random search -- no scipy needed).
    """
    from econsae.calibration.moments import load_targets

    target, weights, scales, _ = load_targets(targets_path)
    bounds = dict(param_bounds or DEFAULT_PARAM_BOUNDS)
    names = list(bounds.keys())
    lo = np.array([bounds[n][0] for n in names], dtype=np.float64)
    hi = np.array([bounds[n][1] for n in names], dtype=np.float64)

    base = SimConfig.default()
    baseline_moments = _evaluate_moments(base, n_traj, n_periods, seeds)
    baseline_obj = moment_distance(baseline_moments, target, weights, scales)

    trace: list[float] = []
    best = {"obj": float("inf"), "theta": None, "moments": None}
    n_evals = 0

    def score(theta: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        cfg = base.with_overrides(**dict(zip(names, theta)))
        moments = _evaluate_moments(cfg, n_traj, n_periods, seeds)
        obj = moment_distance(moments, target, weights, scales)
        if obj < best["obj"]:
            best.update(obj=obj, theta=np.array(theta, dtype=np.float64), moments=moments)
        trace.append(best["obj"])
        return obj

    # Seed the search with the default params (which lie within bounds), so
    # the returned best is guaranteed no worse than the baseline even for a
    # short/unlucky search.
    default_flat = base.flat()
    score(np.array([default_flat[n] for n in names], dtype=np.float64))

    t0 = time.time()
    if method == "random":
        rng = np.random.default_rng(de_seed)
        n_draws = maxiter * popsize
        for _ in range(n_draws):
            theta = lo + rng.random(len(names)) * (hi - lo)
            score(theta)
        # coordinate-descent refine around the best draw
        if best["theta"] is not None:
            for _ in range(2):
                for j in range(len(names)):
                    for frac in (-0.1, 0.1):
                        cand = best["theta"].copy()
                        cand[j] = float(np.clip(cand[j] + frac * (hi[j] - lo[j]), lo[j], hi[j]))
                        score(cand)
    else:
        from scipy.optimize import differential_evolution, minimize

        scipy_bounds = list(zip(lo.tolist(), hi.tolist()))
        if method == "differential_evolution":
            differential_evolution(
                score, scipy_bounds, maxiter=maxiter, popsize=popsize,
                seed=de_seed, polish=False, tol=1e-4, mutation=(0.5, 1.0),
                recombination=0.7, init="latinhypercube",
            )
        elif method == "nelder-mead":
            x0 = 0.5 * (lo + hi)
            minimize(score, x0, method="Nelder-Mead",
                     options={"maxiter": maxiter * popsize, "xatol": 1e-4, "fatol": 1e-6})
        else:
            raise ValueError(f"unknown method: {method!r}")
    wall = time.time() - t0

    best_theta = best["theta"] if best["theta"] is not None else 0.5 * (lo + hi)
    fitted = base.with_overrides(**dict(zip(names, best_theta)))
    return CalibrationResult(
        config=fitted,
        objective=float(best["obj"]),
        baseline_objective=float(baseline_obj),
        sim_moments=best["moments"] or baseline_moments,
        target_moments=target,
        weights=weights,
        scales=scales,
        param_names=names,
        bounds=bounds,
        seeds=tuple(seeds),
        method=method,
        n_traj=n_traj,
        n_periods=n_periods,
        n_evals=n_evals,
        trace=trace,
        wall_seconds=wall,
    )


@dataclass
class MultiStartResult:
    """Spread of independent calibration fits -> parameter identifiability.

    The moment-matching problem is underdetermined (more params than moments),
    so several parameter vectors hit nearly the same objective. Running the
    fit from independent optimizer seeds and looking at where each param lands
    reveals which knobs are well-determined (tight spread) and which are not
    (spread fills their bound range).
    """
    param_names: list[str]
    bounds: dict[str, tuple[float, float]]
    starts: list[dict]                       # per start: seed, objective, params
    target_moments: dict[str, float]

    def identifiability_table(self) -> list[dict]:
        """Per-param stats. `spread_frac` = std / bound-range; high => weak."""
        default_flat = SimConfig.default().flat()
        rows = []
        for name in self.param_names:
            vals = np.array([s["params"][name] for s in self.starts], dtype=np.float64)
            lo, hi = self.bounds[name]
            rng = max(hi - lo, 1e-12)
            std = float(vals.std())
            mean = float(vals.mean())
            spread_frac = std / rng
            rows.append({
                "param": name,
                "default": float(default_flat[name]),
                "mean": mean,
                "std": std,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "bound_lo": lo,
                "bound_hi": hi,
                "spread_frac": spread_frac,                 # std as fraction of range
                # crude identifiability label for quick scanning
                "identifiability": ("well" if spread_frac < 0.10
                                    else "weak" if spread_frac > 0.25
                                    else "moderate"),
            })
        return rows

    def to_report(self) -> dict:
        objs = [s["objective"] for s in self.starts]
        return {
            "n_starts": len(self.starts),
            "objective": {
                "min": float(min(objs)), "max": float(max(objs)),
                "mean": float(np.mean(objs)), "std": float(np.std(objs)),
            },
            "param_names": self.param_names,
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "identifiability": self.identifiability_table(),
            "starts": self.starts,
        }


def multistart_calibrate(
    targets_path: str,
    *,
    n_starts: int = 6,
    start_seed0: int = 0,
    **calibrate_kwargs,
) -> MultiStartResult:
    """Run `calibrate` from `n_starts` independent optimizer seeds.

    Extra kwargs (n_traj, n_periods, seeds, method, maxiter, popsize, ...) are
    forwarded to `calibrate`. The ensemble eval seeds stay fixed across starts
    (so the objective surface is identical); only the optimizer's own RNG
    (`de_seed`) varies, isolating optimizer-induced spread from sampling noise.
    """
    starts: list[dict] = []
    param_names: list[str] = []
    bounds: dict = {}
    target: dict = {}
    for i in range(n_starts):
        r = calibrate(targets_path, de_seed=start_seed0 + i, **calibrate_kwargs)
        param_names = r.param_names
        bounds = r.bounds
        target = r.target_moments
        flat = r.config.flat()
        starts.append({
            "de_seed": start_seed0 + i,
            "objective": r.objective,
            "params": {k: float(flat[k]) for k in r.param_names},
        })
    return MultiStartResult(param_names=param_names, bounds=bounds,
                            starts=starts, target_moments=target)
