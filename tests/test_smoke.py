"""Smoke tests for Phase 0.5: multi-good, cohort-aware, credit-enabled SFC economy.

Verifies that:
  - economy builds with the expected agent topology
  - rollouts produce trajectories of the right shape
  - all 9 transaction kinds appear under default conditions
  - conservation residuals are at machine precision under baseline and under shocks
  - stochastic shock generator is reproducible and labels are recorded
  - ensemble runner produces conservation-clean trajectories
  - ground-truth feature matrix has correctly aligned rows
"""

from __future__ import annotations

import numpy as np

from econsae.embeddings import COORDS, COORD_IDX, DIM, build_economy
from econsae.sectors import GOODS_SECTORS, HH_COHORTS
from econsae.simulator.core import Economy, check_conservation, TXN_KINDS
from econsae.simulator.shocks import draw_shock_schedule
from econsae.simulator.ensemble import generate_ensemble
from econsae.ground_truth import build_feature_matrix, all_features


# ---- economy topology -----------------------------------------------------
def test_economy_topology():
    econ = build_economy(households_per_cohort=4, firms_per_sector=1)
    # 12 HH + 3 firms + 1 gov + 1 bank
    assert len(econ) == 12 + 3 + 1 + 1
    X = econ.stack()
    assert X.shape == (17, DIM)
    assert DIM == 23
    assert len(COORDS) == DIM
    # Each cohort and firm sector represented
    for c in HH_COHORTS:
        assert len(econ.by_cohort(c)) == 4
    for s in GOODS_SECTORS:
        assert len(econ.by_firm_sector(s)) == 1


# ---- rollout + conservation -----------------------------------------------
def test_rollout_shape():
    econ = Economy.small()
    traj = econ.rollout(n_periods=12)
    assert traj.T == 12
    assert traj.stack_states().shape == (12, 17, DIM)
    assert all("GDP" in m for m in traj.macros)


def test_conservation_baseline():
    econ = Economy.small()
    traj = econ.rollout(n_periods=40)
    res = check_conservation(traj)
    assert res["money_drift"] < 1e-6
    assert res["debt_net"] < 1e-6
    assert res["labor_balance"] < 1e-6
    for sec in GOODS_SECTORS:
        assert res[f"goods_balance_{sec}"] < 1e-6


def test_conservation_with_shocks():
    econ = Economy.small()
    sched = draw_shock_schedule(n_periods=40, seed=7)
    traj = econ.rollout(40, shocks=sched.shocks)
    res = check_conservation(traj)
    for k, v in res.items():
        assert v < 1e-6, f"{k} = {v} exceeds tol"


def test_phase3_features_optional_and_conserve():
    """Sentiment-driven MPC, Taylor rule, and I-O network are all optional;
    each must (a) leave conservation untouched and (b) actually change the
    macro dynamics relative to the off-baseline."""
    import numpy as np
    base = Economy.small()
    base_traj = base.rollout(40)
    base_gdps = np.array([m["GDP"] for m in base_traj.macros])

    for name, kwargs in [
        ("sentiment", dict(sentiment_strength=0.20)),
        ("taylor",    dict(taylor_rule=True)),
        ("io_network", dict(io_network=True)),
    ]:
        econ = Economy.small()
        for k, v in kwargs.items():
            setattr(econ, k, v)
        traj = econ.rollout(40)
        res = check_conservation(traj)
        for rk, rv in res.items():
            assert rv < 1e-6, f"{name}: {rk}={rv}"
        # Dynamics should differ from baseline (at least one period)
        gdps = np.array([m["GDP"] for m in traj.macros])
        assert not np.allclose(gdps, base_gdps), f"{name}: dynamics identical to baseline"


def test_all_txn_kinds_fire_under_diverse_shocks():
    """With stochastic shocks across enough periods, every txn kind should appear.

    `b2b_purchase` is conditional on the I-O network being enabled, so we
    expect it only when `io_network=True`.
    """
    econ = Economy.small()
    econ.io_network = True
    sched = draw_shock_schedule(n_periods=80, seed=0)
    traj = econ.rollout(80, shocks=sched.shocks)
    kinds = {t.kind for p in traj.txn_log for t in p}
    for k in TXN_KINDS:
        assert k in kinds, f"missing txn kind: {k}"
    sectors = {t.sector for p in traj.txn_log for t in p if t.sector}
    for sec in GOODS_SECTORS:
        assert sec in sectors


# ---- shock generator ------------------------------------------------------
def test_shock_schedule_reproducible():
    a = draw_shock_schedule(n_periods=30, seed=123)
    b = draw_shock_schedule(n_periods=30, seed=123)
    assert a.shocks == b.shocks
    assert a.kinds == b.kinds


def test_shock_kinds_recorded():
    sched = draw_shock_schedule(n_periods=80, seed=0)
    # union of all observed shock kinds includes at least the always-on AR ones
    all_kinds = set().union(*sched.kinds)
    for sec in GOODS_SECTORS:
        assert f"tfp_{sec}" in all_kinds
    for c in HH_COHORTS:
        assert f"sentiment_{c}" in all_kinds


# ---- ensemble -------------------------------------------------------------
def test_ensemble_conservation():
    ens = generate_ensemble(n_trajectories=4, n_periods=20, seed=0)
    cons = ens.conservation_summary()
    for k, v in cons.items():
        assert v < 1e-6, f"{k} = {v}"


# ---- ground-truth labeling ------------------------------------------------
def test_feature_matrix_shape_and_alignment():
    ens = generate_ensemble(n_trajectories=4, n_periods=20, seed=0)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    n_samples = 4 * 20 * 17
    assert fm.X.shape == (n_samples, DIM)
    assert fm.Y.shape == (n_samples, len(fm.feature_vocab))
    assert len(fm.sample_index) == n_samples
    assert fm.feature_vocab == all_features()
    # row 0 -> traj 0, period 0, agent 0
    assert fm.sample_index[0] == (0, 0, 0)


def test_feature_vocab_has_each_class():
    vocab = set(all_features())
    # categorical
    assert "sector:household" in vocab
    assert "cohort:young" in vocab
    assert "firm_sector:food" in vocab
    # bucketed
    assert "debt_bucket:high" in vocab
    assert "mpc_bucket:high" in vocab
    # conjunctive
    assert any(v.startswith("young_AND_") for v in vocab)
    # regime
    assert "phase:expansion" in vocab
    # shocks
    assert "shock_period_has:monetary" in vocab


# ---- SAE model + training -------------------------------------------------
def test_sae_variants_forward_and_train():
    import torch
    from econsae.sae.models import make_sae
    from econsae.sae.train import TrainConfig, train

    torch.manual_seed(0)
    X = torch.randn(256, 24)
    for kind in ("topk", "l1", "jumprelu"):
        kw = ({"k": 4} if kind == "topk"
              else {"l1_coeff": 1e-2} if kind == "l1"
              else {"l0_coeff": 5e-3})
        sae = make_sae(kind, input_dim=24, n_features=32, **kw)
        out = sae(X)
        assert out.x_hat.shape == X.shape
        assert out.z.shape == (256, 32)
        # short training run shouldn't crash
        hist = train(sae, X, TrainConfig(epochs=2, batch_size=64, log_every=10**6),
                     verbose=False)
        assert len(hist.recon_loss) > 0


# ---- World-model classes --------------------------------------------------
def test_world_model_variants_instantiate_and_forward():
    import torch
    from econsae.sae.world_model import (
        WorldModel, DeepWorldModel, AttnWorldModel, TemporalWorldModel,
        MACRO_DIM, SHOCK_DIM,
    )
    from econsae.embeddings import DIM

    # Per-agent input: state + macro + shock
    per_agent = torch.randn(8, DIM + MACRO_DIM + SHOCK_DIM)
    # Per-period input: (B, N, in_dim)
    per_period = torch.randn(2, 17, DIM + MACRO_DIM + SHOCK_DIM)
    # Per-trajectory input: (B, T, N, in_dim)
    per_traj = torch.randn(2, 5, 17, DIM + MACRO_DIM + SHOCK_DIM)

    wm = WorldModel(); out = wm(per_agent)
    assert out.shape == (8, DIM)

    deep = DeepWorldModel(); out = deep(per_agent)
    assert out.shape == (8, DIM)

    attn = AttnWorldModel(); out = attn(per_period)
    assert out.shape == (2, 17, DIM)

    tem = TemporalWorldModel(); out = tem(per_traj)
    assert out.shape == (2, 5, 17, DIM)
    _, h1 = tem(per_traj, return_h1=True)
    assert h1.shape[-1] == tem.h1_dim


# ---- SAE feeds -----------------------------------------------------------
def test_feed_builders_align_with_gt_matrix():
    from econsae.simulator.ensemble import generate_ensemble
    from econsae.sae.data import feed_raw, feed_embedded
    ens = generate_ensemble(n_trajectories=2, n_periods=10, seed=0)
    fr = feed_raw(ens)
    fe = feed_embedded(ens, embed_dim=12, seed=0)
    assert fr.Y.shape == fe.Y.shape  # same vocab dimension
    assert fr.X.shape[0] == fe.X.shape[0]  # same sample count
    assert fr.feature_vocab == fe.feature_vocab
    # Y is binary
    import numpy as np
    assert ((fr.Y == 0) | (fr.Y == 1)).all()


# ---- Phase 10: calibration ------------------------------------------------
import os

_TARGETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "macro_targets_us.json",
)


def test_price_level_macro_present():
    """The Phase 10 inflation series: mean firm price, additive macro key."""
    econ = Economy.small()
    traj = econ.rollout(n_periods=12)
    assert all("price_level" in m for m in traj.macros)
    assert all(m["price_level"] > 0 for m in traj.macros)


def test_compute_moments_on_known_series():
    from econsae.calibration import compute_moments
    T = 10
    gdp = np.full((2, T), 100.0)                       # constant -> zero growth
    price = np.tile(1.01 ** np.arange(T), (2, 1))      # +1% per period
    rate = np.full((2, T), 0.03)
    debt = np.full((2, T), 20.0)
    money = np.full((2, T), 100.0)
    m = compute_moments({
        "GDP": gdp, "price_level": price, "interest_rate": rate,
        "debt_outstanding": debt, "money_stock": money,
    })
    assert abs(m["gdp_growth_mean"]) < 1e-9
    assert m["gdp_growth_vol"] < 1e-9
    assert m["recession_freq"] == 0.0                  # constant GDP never contracts
    assert abs(m["inflation_mean"] - np.log(1.01)) < 1e-6
    assert abs(m["fedfunds_mean"] - 0.03) < 1e-9
    assert m["fedfunds_vol"] < 1e-9
    assert abs(m["debt_to_money"] - 0.2) < 1e-9


def test_moment_distance_zero_at_target():
    from econsae.calibration import moment_distance
    tgt = {"a": 0.5, "b": 0.02}
    assert moment_distance(tgt, tgt) == 0.0
    # weight-0 moments are ignored even when far off
    assert moment_distance({"a": 999.0}, {"a": 0.0}, weights={"a": 0.0}) == 0.0


def test_sim_config_defaults_match_draw_shock_and_roundtrip():
    import inspect
    from econsae.calibration import SimConfig
    from econsae.simulator import shocks as S
    defaults = {
        k: v.default
        for k, v in inspect.signature(S.draw_shock_schedule).parameters.items()
        if v.default is not inspect.Parameter.empty and k not in ("n_periods", "seed")
    }
    assert SimConfig.default().shock_kwargs() == defaults
    # nested dict round-trip
    cfg = SimConfig.default()
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


def test_generate_ensemble_backcompat_byte_identical():
    """Default path must be byte-identical to passing SimConfig.default()."""
    from econsae.calibration import SimConfig
    a = generate_ensemble(n_trajectories=4, n_periods=20, seed=1)
    b = generate_ensemble(n_trajectories=4, n_periods=20, seed=1,
                          sim_config=SimConfig.default())
    assert np.array_equal(a.stack_all_states(), b.stack_all_states())
    assert [s.kinds for s in a.shock_schedules] == [s.kinds for s in b.shock_schedules]


def test_calibration_smoke_improves_objective(tmp_path):
    """A tiny calibration run returns a valid config no worse than baseline."""
    from econsae.calibration import calibrate, SimConfig
    r = calibrate(_TARGETS_PATH, n_traj=4, n_periods=30, seeds=(0, 1),
                  method="random", maxiter=2, popsize=4)
    assert isinstance(r.config, SimConfig)
    assert r.n_evals > 0
    assert r.objective <= r.baseline_objective + 1e-9
    # fitted config serializes and round-trips
    out = tmp_path / "fit.json"
    r.config.to_json(str(out))
    assert SimConfig.from_json(str(out)) == r.config


def test_multistart_identifiability_structure():
    """Multi-start fit returns per-param spread over the calibrated knobs."""
    from econsae.calibration import multistart_calibrate
    res = multistart_calibrate(_TARGETS_PATH, n_starts=2, n_traj=4, n_periods=30,
                               seeds=(0, 1), method="random", maxiter=2, popsize=3)
    assert len(res.starts) == 2
    table = res.identifiability_table()
    assert {r["param"] for r in table} == set(res.param_names)
    for r in table:
        assert r["bound_lo"] <= r["mean"] <= r["bound_hi"]
        assert r["spread_frac"] >= 0.0
        assert r["identifiability"] in ("well", "moderate", "weak")
    report = res.to_report()
    assert "identifiability" in report
    # thresholds are emitted explicitly, and the correlation matrix is square
    assert {"well_below", "weak_above"} <= set(report["thresholds"])
    corr = report["param_correlation"]
    assert set(corr) == set(res.param_names)
    for a in res.param_names:
        assert abs(corr[a][a] - 1.0) < 1e-9 or corr[a][a] == 0.0   # 0 if constant col


def test_refresh_macro_targets_parsing(tmp_path):
    """FRED CSV parsing: missing values skipped, monthly->quarterly mean."""
    from scripts.refresh_macro_targets import _read_fred_csv, _to_quarterly
    p = tmp_path / "s.csv"
    p.write_text("observation_date,X\n1990-01-01,1.0\n1990-02-01,.\n"
                 "1990-03-01,3.0\n1990-04-01,5.0\n")
    rec = _read_fred_csv(str(p))
    assert (1990, 2) not in rec                          # missing "." skipped
    assert rec[(1990, 1)] == 1.0 and rec[(1990, 3)] == 3.0
    q = _to_quarterly(rec)
    assert abs(q[(1990, 1)] - 2.0) < 1e-9                # mean(1.0, 3.0)
    assert abs(q[(1990, 2)] - 5.0) < 1e-9
    # all-missing series -> empty dict (graceful)
    p2 = tmp_path / "empty.csv"
    p2.write_text("observation_date,X\n1990-01-01,.\n1990-02-01,NaN\n")
    assert _read_fred_csv(str(p2)) == {}


def test_morris_trajectory_properties():
    """Each Morris trajectory: consecutive rows differ in exactly one coord
    by +/- delta, and all coordinates stay in [0, 1]."""
    from econsae.calibration.sensitivity import _morris_trajectory
    k, p = 6, 4
    delta = p / (2.0 * (p - 1))
    rng = np.random.default_rng(0)
    for _ in range(20):
        B = _morris_trajectory(k, p, delta, rng)
        assert B.shape == (k + 1, k)
        assert B.min() >= -1e-9 and B.max() <= 1 + 1e-9
        for j in range(k):
            d = np.abs(B[j + 1] - B[j])
            moved = d > 1e-9
            assert moved.sum() == 1                       # exactly one coord moves
            assert abs(d[moved][0] - delta) < 1e-9        # by exactly delta
        # every coordinate is perturbed exactly once across the trajectory
        moved_counts = (np.abs(np.diff(B, axis=0)) > 1e-9).sum(axis=0)
        assert (moved_counts == 1).all()


def test_feed_threshold_threading():
    """feed_* thread base_rate/monetary_step into the phase:high_rate label."""
    from econsae.sae.data import feed_raw
    ens = generate_ensemble(n_trajectories=2, n_periods=12, seed=0)
    j = feed_raw(ens).feature_vocab.index("phase:high_rate")
    hot = feed_raw(ens, base_rate=0.0, monetary_step=0.0).Y[:, j]    # threshold 0 -> always
    cold = feed_raw(ens, base_rate=10.0, monetary_step=0.0).Y[:, j]  # threshold 10 -> never
    assert hot.sum() > 0
    assert cold.sum() == 0


def test_resolve_calibration_arm(tmp_path):
    from scripts._calibration_arm import resolve_arm
    from econsae.calibration import SimConfig
    base = resolve_arm(None)
    assert base.label == "baseline" and base.suffix == "" and base.sim_config is None
    assert (base.base_rate, base.monetary_step) == (0.02, 0.01)
    p = tmp_path / "c.json"
    SimConfig.default().with_overrides(base_interest_rate=0.05, monetary_step=0.02).to_json(str(p))
    cal = resolve_arm(str(p))
    assert cal.label == "calibrated" and cal.suffix == "__calibrated"
    assert abs(cal.base_rate - 0.05) < 1e-9 and abs(cal.monetary_step - 0.02) < 1e-9


def test_morris_screening_structure():
    from econsae.calibration import morris_screening
    res = morris_screening(_TARGETS_PATH, r=2, p=4, n_traj=4, n_periods=30,
                           seeds=(0, 1), seed=0)
    k = len(res.param_names)
    assert res.n_evals == 2 * (k + 1)                     # r * (k+1)
    table = res.ranking_table()
    assert {r["param"] for r in table} == set(res.param_names)
    assert [r["rank"] for r in table] == list(range(1, k + 1))
    assert all(r["mu_star"] >= 0.0 for r in table)
    # per-moment mu* matrix is complete
    for n in res.param_names:
        assert set(res.mu_star_by_moment[n]) == set(res.moment_keys)
    assert set(res.top_driver_per_moment()) == set(res.moment_keys)
    assert "ranking" in res.to_report()
