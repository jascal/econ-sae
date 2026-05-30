"""Phase 10: calibrate the simulator to historical US macro moments.

Fits the calibratable `SimConfig` parameters (shock-schedule volatilities,
impulse probabilities, policy-rate level, plus a few behavioral knobs) so
the ensemble's scale-invariant macro moments approach the vendored targets
in `data/macro_targets_us.json`. Writes the fitted config and a calibration
report.

Outputs:
    configs/calibrated_macro.json                 -- fitted SimConfig
    runs/calibration/calibration_summary.json     -- moment table + metadata
    runs/calibration/trace.json                   -- best objective per eval

Usage:
    python scripts/calibrate.py                    # 'standard' budget
    python scripts/calibrate.py --budget fast      # quick (~minutes)
    python scripts/calibrate.py --budget thorough  # best fit (slow)
    python scripts/calibrate.py --method random    # numpy-only, no scipy
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from econsae.calibration import calibrate

# (n_traj, n_periods, seeds, maxiter, popsize)
BUDGETS = {
    "fast":     dict(n_traj=6,  n_periods=48, seeds=(0, 1),       maxiter=8,  popsize=6),
    "standard": dict(n_traj=10, n_periods=64, seeds=(0, 1, 2),    maxiter=15, popsize=8),
    "thorough": dict(n_traj=16, n_periods=80, seeds=(0, 1, 2, 3), maxiter=30, popsize=12),
}

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
REPORT_DIR = os.path.join(REPO_ROOT, "runs", "calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS), default="standard")
    ap.add_argument("--method", default="differential_evolution",
                    choices=["differential_evolution", "nelder-mead", "random"])
    ap.add_argument("--targets", default=os.path.join("data", "macro_targets_us.json"))
    ap.add_argument("--out", default=os.path.join("configs", "calibrated_macro.json"))
    args = ap.parse_args()

    budget = BUDGETS[args.budget]
    print("=" * 78)
    print(f"Phase 10 calibration | budget={args.budget} method={args.method}")
    print(f"  ensemble per eval: {budget['n_traj']} traj x {budget['n_periods']} periods "
          f"x {len(budget['seeds'])} seeds | DE maxiter={budget['maxiter']} popsize={budget['popsize']}")
    print("=" * 78)

    result = calibrate(args.targets, method=args.method, **budget)

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Fitted config (the durable artifact Phase 10 consumes)
    result.config.to_json(args.out, extra={
        "source": "scripts/calibrate.py",
        "targets": args.targets,
        "budget": args.budget,
        "method": args.method,
        "objective": result.objective,
        "baseline_objective": result.baseline_objective,
    })

    # Report + trace
    with open(os.path.join(REPORT_DIR, "calibration_summary.json"), "w") as f:
        json.dump(result.to_report(), f, indent=2)
        f.write("\n")
    with open(os.path.join(REPORT_DIR, "trace.json"), "w") as f:
        json.dump({"trace": result.trace}, f)
        f.write("\n")

    # Console summary
    print(f"\nobjective: {result.baseline_objective:.3f} (baseline) "
          f"-> {result.objective:.3f} (fitted)   "
          f"[{result.n_evals} evals, {result.wall_seconds:.0f}s]")
    print(f"\n{'moment':18s} {'target':>9s} {'sim':>9s} {'z':>7s} {'w':>4s}")
    print("-" * 52)
    for row in result.moment_table():
        print(f"{row['moment']:18s} {row['target']:9.4f} {row['sim']:9.4f} "
              f"{row['z']:7.2f} {row['weight']:4.1f}")
    print(f"\nfitted params (changed from default):")
    base = result.config.default().flat()
    fit = result.config.flat()
    for k in result.param_names:
        if abs(fit[k] - base[k]) > 1e-9:
            print(f"  {k:22s} {base[k]:8.4f} -> {fit[k]:8.4f}")
    print(f"\nwrote:\n  {args.out}\n  {os.path.join('runs', 'calibration', 'calibration_summary.json')}")


if __name__ == "__main__":
    main()
