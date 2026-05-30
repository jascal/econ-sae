"""Offline regeneration of data/macro_targets_us.json from local FRED CSVs.

The calibration targets are vendored (committed values, no live API) for
reproducibility. This helper documents exactly how they were derived and
recomputes them from FRED CSV downloads you provide on disk -- it does NOT
hit the network. Re-run it when you want to refresh the vintage.

The moments are computed with the SAME functions the simulator side uses
(`econsae.calibration.compute_moments`), so the real-data targets and the
simulated moments are guaranteed comparable. The one exception is
`recession_freq`: the simulator's `phase:contraction` rule (GDP < 0.90 x
trailing-5 mean) essentially never fires on smooth real GDP, so that target
is taken from a recession *indicator* series (FRED USREC) -- the fraction of
quarters flagged a recession -- which is what the contraction-label
prevalence is meant to approximate.

Download these single-series CSVs from https://fred.stlouisfed.org (each as
`observation_date,<SERIES>`):
    GDPC1     Real GDP, quarterly                    (required)
    FEDFUNDS  Effective Federal Funds Rate, monthly  (required, percent)
    CPIAUCSL  CPI All Urban Consumers, monthly       (required, index)
    USREC     NBER Recession Indicator, monthly 0/1  (optional -> recession_freq)
    GFDEBTN   Federal Debt: Total Public Debt        (optional -> debt_to_money)
    M2SL      M2 Money Stock, monthly                (optional -> debt_to_money)

scales / weights / provenance structure are preserved from the existing
targets file; only the `moments` block + provenance vintage are updated.

Usage:
    python scripts/refresh_macro_targets.py \
        --gdp GDPC1.csv --fedfunds FEDFUNDS.csv --cpi CPIAUCSL.csv \
        [--usrec USREC.csv --debt GFDEBTN.csv --m2 M2SL.csv] \
        [--start 1990 --end 2019] [--out data/macro_targets_us.json]

By default writes a *_refreshed.json sibling so the committed file is not
clobbered until you explicitly point --out at it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np

from econsae.calibration.moments import MOMENT_KEYS, compute_moments

DEFAULT_TARGETS = os.path.join("data", "macro_targets_us.json")


def _read_fred_csv(path: str) -> dict[tuple[int, int], float]:
    """Parse a FRED single-series CSV -> {(year, month): value}.

    Accepts the modern `observation_date,SERIES` header or the older
    `DATE,VALUE`. Missing observations (FRED writes ".") are skipped.
    """
    out: dict[tuple[int, int], float] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                         # header
        for row in reader:
            if len(row) < 2:
                continue
            date, raw = row[0].strip(), row[1].strip()
            if raw in ("", ".", "NaN"):
                continue
            try:
                y, m = int(date[:4]), int(date[5:7])
                out[(y, m)] = float(raw)
            except ValueError:
                continue
    return out


def _to_quarterly(monthly: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Average monthly observations into (year, quarter) -> mean value."""
    buckets: dict[tuple[int, int], list[float]] = {}
    for (y, m), v in monthly.items():
        q = (m - 1) // 3 + 1
        buckets.setdefault((y, q), []).append(v)
    return {k: float(np.mean(vs)) for k, vs in buckets.items()}


def _quarter_key(series: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Coerce a series to (year, quarter) keys. Quarterly series (months
    1/4/7/10) and already-quarterly maps both collapse cleanly."""
    return _to_quarterly(series)


def _aligned(arrays: dict[str, dict], keys: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Build (1, T) arrays over a shared, time-sorted set of quarter keys."""
    return {name: np.array([[series[k] for k in keys]], dtype=np.float64)
            for name, series in arrays.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdp", required=True, help="GDPC1 CSV (real GDP, quarterly)")
    ap.add_argument("--fedfunds", required=True, help="FEDFUNDS CSV (monthly, percent)")
    ap.add_argument("--cpi", required=True, help="CPIAUCSL CSV (monthly index)")
    ap.add_argument("--usrec", help="USREC CSV (monthly 0/1) -> recession_freq")
    ap.add_argument("--debt", help="GFDEBTN CSV -> debt_to_money numerator")
    ap.add_argument("--m2", help="M2SL CSV -> debt_to_money denominator")
    ap.add_argument("--start", type=int, default=1990, help="first year (inclusive)")
    ap.add_argument("--end", type=int, default=2019, help="last year (inclusive)")
    ap.add_argument("--targets", default=DEFAULT_TARGETS,
                    help="existing targets file (scales/weights/provenance preserved)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <targets>_refreshed.json)")
    args = ap.parse_args()

    gdp = _quarter_key(_read_fred_csv(args.gdp))
    rate = _quarter_key(_read_fred_csv(args.fedfunds))           # percent
    cpi = _quarter_key(_read_fred_csv(args.cpi))

    # Common quarters within the window, time-sorted (diff(log) needs order).
    def in_window(k):
        return args.start <= k[0] <= args.end
    common = sorted(set(gdp) & set(rate) & set(cpi))
    common = [k for k in common if in_window(k)]
    if len(common) < 8:
        sys.exit(f"only {len(common)} aligned quarters in [{args.start},{args.end}]; "
                 f"need >= 8. Check the CSVs / window.")

    arrays = {
        "GDP": gdp,
        "interest_rate": {k: rate[k] / 100.0 for k in rate},     # percent -> decimal
        "price_level": cpi,
        "debt_outstanding": {k: 1.0 for k in common},            # placeholder
        "money_stock": {k: 1.0 for k in common},                 # placeholder
    }
    if args.debt and args.m2:
        debt = _quarter_key(_read_fred_csv(args.debt))
        m2 = _quarter_key(_read_fred_csv(args.m2))
        dm_keys = [k for k in common if k in debt and k in m2]
        if dm_keys:
            arrays["debt_outstanding"] = {k: debt.get(k, 1.0) for k in common}
            arrays["money_stock"] = {k: m2.get(k, 1.0) for k in common}

    aligned = _aligned(arrays, common)
    mom = compute_moments(aligned)

    # recession_freq: real GDP almost never trips the contraction rule, so
    # take it from the recession indicator instead (fraction of quarters).
    rec_source = "contraction-rule on real GDP (degenerate; see notes)"
    if args.usrec:
        usrec = _quarter_key(_read_fred_csv(args.usrec))
        flags = [usrec[k] for k in common if k in usrec]
        if flags:
            mom["recession_freq"] = float(np.mean([1.0 if v > 0.5 else 0.0 for v in flags]))
            rec_source = "FRED USREC quarterly fraction"
    if not (args.debt and args.m2):
        mom["debt_to_money"] = None  # not derivable; keep existing below

    # Merge with existing file: preserve scales / weights / provenance shape.
    with open(args.targets) as f:
        doc = json.load(f)
    existing = doc.get("moments", {})
    new_moments = {}
    for k in MOMENT_KEYS:
        v = mom.get(k)
        new_moments[k] = float(existing[k]) if v is None else float(v)
    doc["moments"] = new_moments
    prov = doc.setdefault("provenance", {})
    prov["date_range"] = f"{args.start}-01-01 to {args.end}-12-31"
    prov["vintage"] = f"refreshed by scripts/refresh_macro_targets.py over {len(common)} quarters"
    prov["recession_freq_source"] = rec_source

    out_path = args.out or args.targets.replace(".json", "_refreshed.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    # Comparison report
    print("=" * 64)
    print(f"refreshed macro targets over {len(common)} quarters "
          f"[{common[0][0]}Q{common[0][1]} .. {common[-1][0]}Q{common[-1][1]}]")
    print("=" * 64)
    print(f"{'moment':18s} {'committed':>11s} {'refreshed':>11s}")
    print("-" * 44)
    for k in MOMENT_KEYS:
        old = existing.get(k, float('nan'))
        print(f"{k:18s} {old:>11.4f} {new_moments[k]:>11.4f}")
    print(f"\nrecession_freq source: {rec_source}")
    print(f"wrote {out_path}"
          + ("" if args.out else "  (rename/point --out to overwrite the committed file)"))


if __name__ == "__main__":
    main()
