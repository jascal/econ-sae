"""Phase 10 follow-up: parameter identifiability via multi-start calibration.

The moment-matching fit is underdetermined (more free params than moments),
so the point estimate from a single run can hide which knobs are actually
pinned down by the data and which are free to wander. This runs the
calibration from several independent optimizer seeds and reports, per
parameter, how widely the fits spread relative to the parameter's bound
range -- a cheap stand-in for a posterior, flagging weakly-identified knobs.

The ensemble evaluation seeds are held fixed across starts, so the spread
reflects optimizer/identifiability, not sampling noise.

Output:
    runs/calibration/identifiability_summary.json

Usage:
    python scripts/calibration_identifiability.py                 # default
    python scripts/calibration_identifiability.py --quick         # fast
    python scripts/calibration_identifiability.py --starts 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from econsae.calibration import multistart_calibrate

# per-start (n_traj, n_periods, seeds, maxiter, popsize)
BUDGETS = {
    "quick":    dict(n_traj=5, n_periods=40, seeds=(0, 1),    maxiter=5, popsize=4),
    "standard": dict(n_traj=8, n_periods=60, seeds=(0, 1, 2), maxiter=8, popsize=5),
}

REPORT_DIR = os.path.join(REPO_ROOT, "runs", "calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS), default="standard")
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--method", default="differential_evolution",
                    choices=["differential_evolution", "nelder-mead", "random"])
    ap.add_argument("--targets", default=os.path.join("data", "macro_targets_us.json"))
    args = ap.parse_args()

    budget = BUDGETS[args.budget]
    print("=" * 78)
    print(f"Calibration identifiability | {args.starts} starts | "
          f"budget={args.budget} method={args.method}")
    print("=" * 78)

    res = multistart_calibrate(args.targets, n_starts=args.starts,
                               method=args.method, **budget)
    report = res.to_report()

    objs = report["objective"]
    print(f"\nobjective across starts: min={objs['min']:.2f} "
          f"mean={objs['mean']:.2f} max={objs['max']:.2f} std={objs['std']:.2f}")
    print(f"\n{'param':<22s} {'default':>9s} {'mean fit':>9s} {'std':>8s} "
          f"{'[min':>8s} {'max]':>8s} {'spread%':>8s}  {'id':>8s}")
    print("-" * 90)
    for row in res.identifiability_table():
        print(f"{row['param']:<22s} {row['default']:>9.4f} {row['mean']:>9.4f} "
              f"{row['std']:>8.4f} {row['min']:>8.4f} {row['max']:>8.4f} "
              f"{row['spread_frac']*100:>7.1f}%  {row['identifiability']:>8s}")

    weak = [r["param"] for r in res.identifiability_table()
            if r["identifiability"] == "weak"]
    print(f"\nweakly-identified (spread > 25% of range): "
          f"{', '.join(weak) if weak else '(none)'}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "identifiability_summary.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
