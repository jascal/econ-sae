"""Phase 10.2: Morris elementary-effects sensitivity screening.

Complements the identifiability diagnostic. Identifiability asks "where do
fits land?"; this asks "how much does each knob move the moments?" -- so it
explains *why* a parameter is loosely identified (low influence => the
moments don't constrain it). The per-(param, moment) mu* matrix also shows
which knob drives which moment, which is how you pick an identifying moment
for a loose parameter.

Output:
    runs/calibration/sensitivity_summary.json

Usage:
    python scripts/calibration_sensitivity.py                # standard
    python scripts/calibration_sensitivity.py --quick
    python scripts/calibration_sensitivity.py -r 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from econsae.calibration import morris_screening

# (n_traj, n_periods, seeds, r)  -- p (levels) fixed at 4
BUDGETS = {
    "quick":    dict(n_traj=5, n_periods=40, seeds=(0, 1),    r=6),
    "standard": dict(n_traj=8, n_periods=60, seeds=(0, 1, 2), r=12),
}

REPORT_DIR = os.path.join(REPO_ROOT, "runs", "calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS), default="standard")
    ap.add_argument("-r", "--trajectories", type=int, default=None,
                    help="override r (Morris trajectories)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--targets", default=os.path.join("data", "macro_targets_us.json"))
    args = ap.parse_args()

    budget = dict(BUDGETS[args.budget])
    if args.trajectories is not None:
        budget["r"] = args.trajectories

    print("=" * 78)
    print(f"Morris EE sensitivity | budget={args.budget} r={budget['r']} seed={args.seed}")
    print(f"  {budget['n_traj']} traj x {budget['n_periods']} periods x "
          f"{len(budget['seeds'])} seeds per eval")
    print("=" * 78)

    res = morris_screening(args.targets, seed=args.seed, **budget)
    print(f"\n{res.n_evals} objective evaluations\n")

    print(f"{'param':<22s} {'mu*':>9s} {'mu':>9s} {'sigma':>9s}  {'rank':>4s}")
    print("-" * 58)
    for row in res.ranking_table():
        print(f"{row['param']:<22s} {row['mu_star']:>9.3f} {row['mu']:>9.3f} "
              f"{row['sigma']:>9.3f}  {row['rank']:>4d}")

    print(f"\ntop driver per moment (largest mu* on that moment):")
    for m, drv in res.top_driver_per_moment().items():
        print(f"  {m:<18s} <- {drv}")

    least = res.ranking_table()[-1]
    print(f"\nleast influential knob: {least['param']} (mu*={least['mu_star']:.3f}) "
          f"-- consistent with weak identifiability if also loose")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "sensitivity_summary.json")
    with open(out_path, "w") as f:
        json.dump(res.to_report(), f, indent=2)
        f.write("\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
