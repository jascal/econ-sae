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
