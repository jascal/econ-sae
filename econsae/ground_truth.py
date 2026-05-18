"""Ground-truth feature vocabulary for SAE alignment scoring.

PURPOSE
-------
A trained SAE produces N learned features. To grade it, we need a fixed
vocabulary of *known* features and a binary label per (training-sample,
ground-truth-feature) pair. Then for each (SAE feature, GT feature) pair
we compute the AUC of the SAE feature's activation against the GT label,
take the max-over-SAE-features per GT feature, and report coverage,
mean-best-AUC, and monosemanticity. (Same recipe as sm-sae.)

FEATURE TYPES
-------------
The vocabulary is deliberately mixed in difficulty:

  1. HARD CATEGORICAL  -- one-hot-ish, easy targets:
       sector:household, sector:firm, sector:government, sector:bank
       cohort:young, cohort:prime, cohort:retiree
       firm_sector:food, firm_sector:services, firm_sector:durables
       txn_period_has:<kind>  for kind in TXN_KINDS
       shock_period_has:<kind>  for kind in shock kinds

  2. CONTINUOUS BUCKETED  -- continuous coords binned by population quantile:
       debt_bucket:high           top quartile of debt_liab among debtors
       cash_bucket:poor           below 0.5 * expectation
       mpc_bucket:high            top quartile of mpc
       leverage:high              debt_liab / expectation > 1.0
       inventory_bucket:stockout  any sector inventory < threshold

  3. CONJUNCTIVE  -- compositions that have no clean tensor-product decomp:
       young_AND_credit_constrained   cohort=young AND debt_liab >= 0.9*credit_limit
       prime_AND_high_cash            cohort=prime AND money > 2*expectation
       retiree_AND_decumulating       cohort=retiree AND money declining over window
       food_firm_stockout             firm in food sector AND inv_food < 5
       services_firm_high_demand      services firm AND goods_out_services > median
       durables_firm_excess_inv       durables firm AND inv_durables > 50

  4. REGIME  -- macro-derived phase labels (same label for every agent in period t):
       phase:expansion                period GDP > 1.10 * 5-period trailing mean
       phase:contraction              period GDP < 0.90 * 5-period trailing mean
       phase:high_leverage            aggregate debt / aggregate money > 0.10
       phase:high_rate                interest_rate > base + 1.5*step
       phase:fiscal_active            fiscal shock was applied this period
       phase:monetary_active          monetary shock was applied this period

The CONJUNCTIVE block is the "Polygram trap" --- no clean Kronecker
factorization of the SM-style {cohort, txn_kind, sector, bucket} basis
recovers these features. An SAE has to either find one feature per
conjunction (clean monosemantic) or compose them via multiple features
(failure mode the alignment score penalizes).

SAMPLE STRUCTURE
----------------
For Phase 0.5 we materialize SAE samples as per-(trajectory, period,
agent) rows of the state tensor. The total sample count is
n_trajectories * T * n_agents. Each sample carries the same conserved
state coords (R^14) and side coords (R^9); only the labels depend on
the row of the (T, n_agents) grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from econsae.embeddings import (
    COORD_IDX, DIM, INVENTORY_COORDS, GOODS_OUT_COORDS,
    SECTOR_HH, SECTOR_FIRM, SECTOR_GOV, SECTOR_BANK,
)
from econsae.sectors import GOODS_SECTORS, HH_COHORTS
from econsae.simulator.core import Trajectory, TXN_KINDS
from econsae.simulator.shocks import ShockSchedule


# ---------------------------------------------------------------------------
# Static portion of the vocabulary (always present)
# ---------------------------------------------------------------------------
SECTOR_FEATURES = [
    "sector:household", "sector:firm", "sector:government", "sector:bank",
]
COHORT_FEATURES = [f"cohort:{c}" for c in HH_COHORTS]
FIRM_SECTOR_FEATURES = [f"firm_sector:{s}" for s in GOODS_SECTORS]

BUCKETED_FEATURES = [
    "debt_bucket:high",
    "cash_bucket:poor",
    "cash_bucket:rich",
    "mpc_bucket:high",
    "leverage:high",
    "inventory_bucket:stockout",
    "productivity:high",
]

CONJUNCTIVE_FEATURES = [
    "young_AND_indebted",
    "prime_AND_high_cash",
    "retiree_AND_decumulating",
    "food_firm_low_inv",
    "services_firm_high_output",
    "durables_firm_high_inv",
    "young_AND_high_mpc_AND_expansion",
    "firm_AND_indebted_AND_high_inventory",
]

REGIME_FEATURES = [
    "phase:expansion", "phase:contraction",
    "phase:high_leverage", "phase:high_rate",
    "phase:fiscal_active", "phase:monetary_active",
]

# Per-period transaction-presence features (kind, plus sector for purchases)
TXN_PRESENCE_FEATURES = (
    [f"txn_period_has:{k}" for k in TXN_KINDS if k != "purchase"]
    + [f"txn_period_has:purchase_{s}" for s in GOODS_SECTORS]
)

# Per-period shock-presence features (full enumeration)
SHOCK_KIND_VOCAB = (
    [f"shock_period_has:tfp_{s}" for s in GOODS_SECTORS]
    + [f"shock_period_has:sentiment_{c}" for c in HH_COHORTS]
    + ["shock_period_has:monetary", "shock_period_has:fiscal",
       "shock_period_has:productivity_aggregate"]
)


def all_features() -> list[str]:
    """Return the full sorted feature vocabulary."""
    return sorted(
        set(SECTOR_FEATURES) | set(COHORT_FEATURES) | set(FIRM_SECTOR_FEATURES)
        | set(BUCKETED_FEATURES) | set(CONJUNCTIVE_FEATURES) | set(REGIME_FEATURES)
        | set(TXN_PRESENCE_FEATURES) | set(SHOCK_KIND_VOCAB)
    )


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
@dataclass
class PeriodContext:
    """Features that apply uniformly to every agent in one period."""
    txn_kinds_present: set[str]              # already-namespaced (purchase_food etc.)
    shock_kinds: set[str]
    phase_features: set[str]


def _significant_shock_kinds(shock_dict: dict, kinds: set[str],
                             threshold: float = 0.02) -> set[str]:
    """Return only the shock kinds whose magnitude exceeds a threshold.

    The AR(1) TFP / sentiment processes produce small nonzero values every
    period; without thresholding the corresponding ground-truth features
    are 100% prevalent and have no negative samples to score AUC against.
    """
    out: set[str] = set()
    for k in kinds:
        if k.startswith("tfp_"):
            sec = k[len("tfp_"):]
            key = f"productivity_mult_{sec}"
            if key in shock_dict and abs(np.log(shock_dict[key])) > threshold:
                out.add(k)
        elif k.startswith("sentiment_"):
            cohort = k[len("sentiment_"):]
            key = f"mpc_add_{cohort}"
            if key in shock_dict and abs(shock_dict[key]) > threshold:
                out.add(k)
        elif k == "productivity_aggregate":
            key = "productivity_mult"
            if key in shock_dict and abs(np.log(shock_dict[key])) > 0.5 * threshold:
                out.add(k)
        else:
            # monetary, fiscal: keep as-is (sparse impulses already)
            out.add(k)
    return out


def _txn_kinds_present_in_period(txns) -> set[str]:
    out: set[str] = set()
    for t in txns:
        if t.kind == "purchase" and t.sector:
            out.add(f"purchase_{t.sector}")
        else:
            out.add(t.kind)
    return out


def _phase_features_for_period(
    macros: list[dict], t: int, base_rate: float, monetary_step: float,
    shock_kinds: set[str],
) -> set[str]:
    feats: set[str] = set()
    # Trailing mean GDP (5-period window, t-5..t-1)
    lo = max(0, t - 5)
    window = macros[lo:t]
    cur_gdp = float(macros[t]["GDP"])
    if window:
        ma = float(np.mean([m["GDP"] for m in window]))
        if cur_gdp > 1.10 * ma:
            feats.add("phase:expansion")
        elif cur_gdp < 0.90 * ma:
            feats.add("phase:contraction")

    # Leverage and rate regimes
    debt = float(macros[t]["debt_outstanding"])
    money = float(macros[t]["money_stock"])
    if money > 1e-9 and debt / money > 0.10:
        feats.add("phase:high_leverage")
    rate = float(macros[t]["interest_rate"])
    if rate > base_rate + 1.5 * monetary_step:
        feats.add("phase:high_rate")
    if "fiscal" in shock_kinds:
        feats.add("phase:fiscal_active")
    if "monetary" in shock_kinds:
        feats.add("phase:monetary_active")
    return feats


def agent_features(
    vec: np.ndarray,
    agent_kind: str,
    cohort: str | None,
    firm_sector: str | None,
    period_ctx: PeriodContext,
    debt_quantile_high: float,
    mpc_quantile_high: float,
    productivity_quantile_high: float,
    inventory_stockout_thresh: dict[str, float],
    sector_inv_low: dict[str, float],
    sector_inv_high: dict[str, float],
    sector_out_high: dict[str, float],
) -> set[str]:
    """Return the ground-truth feature label set for one (agent, period) sample."""
    feats: set[str] = set()

    feats.add(f"sector:{agent_kind}")
    if cohort:
        feats.add(f"cohort:{cohort}")
    if firm_sector:
        feats.add(f"firm_sector:{firm_sector}")

    # --- continuous bucketed -----------------------------------------------
    money = vec[COORD_IDX["money"]]
    debt = vec[COORD_IDX["debt_liab"]]
    mpc = vec[COORD_IDX["mpc"]]
    prod = vec[COORD_IDX["productivity"]]
    expect = vec[COORD_IDX["expectation"]]
    credit_lim = vec[COORD_IDX["credit_limit"]]

    if debt > debt_quantile_high and debt > 1e-3:
        feats.add("debt_bucket:high")
    if expect > 1e-6 and money < 0.5 * expect:
        feats.add("cash_bucket:poor")
    if expect > 1e-6 and money > 3.0 * expect:
        feats.add("cash_bucket:rich")
    if mpc > mpc_quantile_high:
        feats.add("mpc_bucket:high")
    if expect > 1e-6 and debt / expect > 1.0:
        feats.add("leverage:high")
    if prod > productivity_quantile_high:
        feats.add("productivity:high")
    # Stockout only meaningful for firms (HHs don't hold inventory).
    if agent_kind == "firm" and firm_sector is not None:
        if vec[COORD_IDX[INVENTORY_COORDS[firm_sector]]] < inventory_stockout_thresh[firm_sector]:
            feats.add("inventory_bucket:stockout")

    # --- conjunctive --------------------------------------------------------
    if cohort == "young" and debt > 0.05:
        feats.add("young_AND_indebted")
    if cohort == "prime" and expect > 1e-6 and money > 2.0 * expect:
        feats.add("prime_AND_high_cash")
    # retiree_AND_decumulating is judged at trajectory build time (window check).

    if firm_sector == "food" and vec[COORD_IDX["inv_food"]] < sector_inv_low["food"]:
        feats.add("food_firm_low_inv")
    if (firm_sector == "services"
            and vec[COORD_IDX[GOODS_OUT_COORDS["services"]]] > sector_out_high["services"]):
        feats.add("services_firm_high_output")
    if firm_sector == "durables" and vec[COORD_IDX["inv_durables"]] > sector_inv_high["durables"]:
        feats.add("durables_firm_high_inv")
    if (cohort == "young" and mpc > mpc_quantile_high
            and "phase:expansion" in period_ctx.phase_features):
        feats.add("young_AND_high_mpc_AND_expansion")
    if agent_kind == "firm" and debt > 0.05 and firm_sector is not None and (
            vec[COORD_IDX[INVENTORY_COORDS[firm_sector]]] > sector_inv_high[firm_sector]):
        feats.add("firm_AND_indebted_AND_high_inventory")

    # --- period-level features ----------------------------------------------
    for k in period_ctx.txn_kinds_present:
        feats.add(f"txn_period_has:{k}")
    for k in period_ctx.shock_kinds:
        feats.add(f"shock_period_has:{k}")
    feats |= period_ctx.phase_features

    return feats


# ---------------------------------------------------------------------------
# Vectorized matrix builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureMatrix:
    X: np.ndarray                     # (N_samples, DIM)
    Y: np.ndarray                     # (N_samples, n_features) {0,1}
    sample_index: list[tuple[int, int, int]]   # (traj_idx, period, agent_idx)
    feature_vocab: list[str]


def _quantile_thresholds(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.quantile(values[values > 0], q)) if np.any(values > 0) else float("inf")


def build_feature_matrix(
    trajectories: list[Trajectory],
    shock_schedules: list[ShockSchedule],
    base_rate: float = 0.02,
    monetary_step: float = 0.01,
) -> FeatureMatrix:
    """Materialize the (X, Y) matrices for SAE alignment scoring.

    X stacks every (trajectory, period, agent) state row. Y is the binary
    label matrix over the full feature vocabulary. Quantile thresholds for
    bucketed features are computed once over the entire ensemble so that
    'debt_bucket:high' has a fixed meaning across trajectories.
    """
    assert len(trajectories) == len(shock_schedules), \
        "trajectories and shock_schedules must align"
    vocab = all_features()
    vocab_idx = {f: i for i, f in enumerate(vocab)}

    # Stack everything for quantile computation
    all_states = []
    for traj in trajectories:
        all_states.append(traj.stack_states().reshape(-1, DIM))
    big = np.concatenate(all_states, axis=0)

    debt_thr = _quantile_thresholds(big[:, COORD_IDX["debt_liab"]], 0.75)
    mpc_thr = _quantile_thresholds(big[:, COORD_IDX["mpc"]], 0.75)
    prod_thr = _quantile_thresholds(big[:, COORD_IDX["productivity"]], 0.75)
    # Per-sector inventory + output quantile thresholds computed across the
    # ensemble. All percentiles taken over nonzero rows (i.e., over firms in
    # that sector, since HHs don't carry inventory):
    #   inv_thrs       (p25)  -- stockout threshold for inventory_bucket
    #   sector_inv_low (p50)  -- "low inventory" cutoff for conjunctive features
    #   sector_inv_high(p75)  -- "high inventory" cutoff (used to fire the two
    #                            previously-dead conjunctive features)
    #   sector_out_high(p75)  -- high-output threshold
    inv_thrs: dict[str, float] = {}
    sector_inv_low: dict[str, float] = {}
    sector_inv_high: dict[str, float] = {}
    sector_out_high: dict[str, float] = {}
    for sec in GOODS_SECTORS:
        inv_col = big[:, COORD_IDX[INVENTORY_COORDS[sec]]]
        out_col = big[:, COORD_IDX[GOODS_OUT_COORDS[sec]]]
        nz_inv = inv_col[inv_col > 1e-6]
        nz_out = out_col[out_col > 1e-9]
        inv_thrs[sec] = float(np.quantile(nz_inv, 0.25)) if nz_inv.size else 1.0
        sector_inv_low[sec] = float(np.quantile(nz_inv, 0.50)) if nz_inv.size else 1.0
        sector_inv_high[sec] = float(np.quantile(nz_inv, 0.75)) if nz_inv.size else 1.0
        sector_out_high[sec] = float(np.quantile(nz_out, 0.75)) if nz_out.size else 1.0

    rows_X: list[np.ndarray] = []
    rows_Y: list[np.ndarray] = []
    sample_idx: list[tuple[int, int, int]] = []

    for ti, (traj, sched) in enumerate(zip(trajectories, shock_schedules)):
        states = traj.stack_states()    # (T, n_agents, DIM)
        T, N, _ = states.shape

        # Per-period contexts
        ctxs: list[PeriodContext] = []
        for t in range(T):
            txn_kinds = _txn_kinds_present_in_period(traj.txn_log[t])
            raw_kinds = sched.kinds[t] if t < len(sched.kinds) else set()
            shock_dict = sched.shocks[t] if t < len(sched.shocks) else {}
            shock_kinds = _significant_shock_kinds(shock_dict, raw_kinds)
            phase = _phase_features_for_period(
                traj.macros, t, base_rate, monetary_step, shock_kinds,
            )
            ctxs.append(PeriodContext(
                txn_kinds_present=txn_kinds,
                shock_kinds=shock_kinds,
                phase_features=phase,
            ))

        # Decumulation detection per HH per period (5-period window)
        money_traj = states[:, :, COORD_IDX["money"]]  # (T, N)
        for t in range(T):
            for ai in range(N):
                vec = states[t, ai]
                kind = traj.agent_kinds[ai]
                cohort = traj.agent_cohorts[ai]
                firm_sector = traj.agent_firm_sectors[ai]
                feats = agent_features(
                    vec, kind, cohort, firm_sector, ctxs[t],
                    debt_quantile_high=debt_thr,
                    mpc_quantile_high=mpc_thr,
                    productivity_quantile_high=prod_thr,
                    inventory_stockout_thresh=inv_thrs,
                    sector_inv_low=sector_inv_low,
                    sector_inv_high=sector_inv_high,
                    sector_out_high=sector_out_high,
                )

                # Decumulation: retiree whose money has fallen over the 5-period window
                if cohort == "retiree" and t >= 5:
                    drop = money_traj[t - 5, ai] - money_traj[t, ai]
                    if drop > 0.5:
                        feats.add("retiree_AND_decumulating")

                row = np.zeros(len(vocab), dtype=np.float32)
                for f in feats:
                    j = vocab_idx.get(f)
                    if j is not None:
                        row[j] = 1.0
                rows_X.append(vec.astype(np.float32))
                rows_Y.append(row)
                sample_idx.append((ti, t, ai))

    X = np.stack(rows_X, axis=0)
    Y = np.stack(rows_Y, axis=0)
    return FeatureMatrix(X=X, Y=Y, sample_index=sample_idx, feature_vocab=vocab)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from econsae.simulator.ensemble import generate_ensemble
    ens = generate_ensemble(n_trajectories=4, n_periods=20, seed=0)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    print(f"feature matrix: X={fm.X.shape}, Y={fm.Y.shape}, "
          f"vocab={len(fm.feature_vocab)} features")
    # Prevalence of each feature (fraction of samples that have it)
    prev = fm.Y.mean(axis=0)
    rows = sorted(zip(fm.feature_vocab, prev), key=lambda x: -x[1])
    print("\nfeature prevalence (top 20):")
    for f, p in rows[:20]:
        print(f"  {p:>6.1%}  {f}")
    print("\nfeature prevalence (rarest 10):")
    for f, p in rows[-10:]:
        print(f"  {p:>6.1%}  {f}")
    never_active = [f for f, p in rows if p == 0]
    print(f"\nfeatures never observed: {len(never_active)}/{len(rows)}")
    if never_active:
        print("  e.g.:", never_active[:5])
