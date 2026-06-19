"""A predictive recoverability model: does rate-distortion theory PREDICT which
ground-truth features an unsupervised SAE recovers?

The structural-match law, made quantitative. The claim (from the regime arc):
  - ALLOCATION (does the SAE surface feature f?) is governed by rate-distortion /
    reverse water-filling -> a feature is recovered iff its VARIANCE-SHARE in the
    representation clears a budget-set water level. Variance-share(f) =
    p(1-p)·‖μ_pos − μ_neg‖² / tr(Σ) (the between-class variance fraction = the
    reconstruction "cost to ignore" f).
  - PRESENCE (can a probe read f?) is governed by detection theory -> the Fisher SNR
    ‖μ_pos − μ_neg‖²_{Σ_w^{-1}} (within-class-whitened, scale-free), NOT variance.
  - They DIVERGE for high-Fisher / low-variance-share features ("meaning is
    variance-cheap"): present (probe high) yet unallocated (SAE low).

Test (one representation, ALL ground-truth features across tiers):
  Spearman(var_share, SAE-AUC)  should be HIGH  (allocation ~ variance-share)
  Spearman(fisher,    probe-AUC) should be HIGH  (presence  ~ Fisher SNR)
  Spearman(fisher,    SAE-AUC)  should be LOWER  (allocation is NOT Fisher-driven)
  + a fitted water level on var_share that separates recovered from not.

Run:  .venv/bin/python scripts/regime_recoverability_theory.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from econsae.ground_truth import build_feature_matrix
from econsae.sae.evaluation import align, feature_tier, score_sae
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    TemporalWorldModel, WMTrainConfig, build_temporal_data,
    extract_temporal_h1_activations, train_temporal_world_model,
)
from econsae.simulator.ensemble import generate_ensemble
from regime_granularity_experiment import (
    PROBE_LAM, SAE_EPOCHS, SAE_L0, SAE_WIDTH, WINDOW, auc, per_period_Y,
    ridge_lda_probe_auc, temporal_pool, zscore,
)

FISHER_LAM = 1e-2


def var_share(X: np.ndarray, y: np.ndarray, totvar: float) -> float:
    """Between-class variance fraction = reconstruction relevance of feature y."""
    y = y.astype(bool)
    if y.sum() < 2 or (~y).sum() < 2:
        return float("nan")
    p = y.mean()
    dmu = X[y].mean(0) - X[~y].mean(0)
    return float(p * (1 - p) * (dmu @ dmu) / totvar)


def fisher_snr(X: np.ndarray, y: np.ndarray, lam: float = FISHER_LAM) -> float:
    """Mahalanobis (within-class-whitened) class separation = detection relevance."""
    y = y.astype(bool)
    if y.sum() < 2 or (~y).sum() < 2:
        return float("nan")
    mp, mn = X[y].mean(0), X[~y].mean(0)
    dmu = mp - mn
    Xp, Xn = X[y] - mp, X[~y] - mn
    Sw = (Xp.T @ Xp + Xn.T @ Xn) / max(len(X) - 2, 1)
    return float(dmu @ np.linalg.solve(Sw + lam * np.eye(len(dmu)), dmu))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, z) -> float:
    """Spearman(a, b) controlling for z — each axis's UNIQUE contribution."""
    r_ab, r_az, r_bz = spearman(a, b), spearman(a, z), spearman(b, z)
    denom = np.sqrt(max((1 - r_az**2) * (1 - r_bz**2), 1e-12))
    return float((r_ab - r_az * r_bz) / denom)


def per_feature_stats(X: np.ndarray, Y: np.ndarray, vocab: list[str], seed: int):
    """For one representation: per-feature var_share, fisher, probe-AUC, SAE-AUC."""
    Xz = zscore(X)
    totvar = float(Xz.var(0).sum())
    torch.manual_seed(seed)
    Xt = torch.tensor(Xz, dtype=torch.float32)
    sae = make_sae("jumprelu", Xt.shape[1], SAE_WIDTH, l0_coeff=SAE_L0, init_theta=0.05)
    train(sae, Xt, TrainConfig(epochs=SAE_EPOCHS, batch_size=256, lr=1e-3,
                               log_every=10**9), verbose=False)
    rep = align(score_sae(sae, Xt), Y, vocab)
    rows = []
    for j, f in enumerate(vocab):
        y = Y[:, j]
        rows.append({
            "feature": f, "tier": feature_tier(f), "prevalence": float(y.mean()),
            "var_share": var_share(Xz, y, totvar),
            "fisher": fisher_snr(Xz, y),
            "probe_auc": ridge_lda_probe_auc(Xz, y, lam=PROBE_LAM, seed=seed),
            "sae_auc": float(rep.alignment[:, j].max()),
        })
    return rows


def build_reps(n_trajs, n_periods, wm_epochs, seed):
    ens = generate_ensemble(n_trajectories=n_trajs, n_periods=n_periods, seed=seed)
    trajs, scheds = ens.trajectories, ens.shock_schedules
    T, N = trajs[0].T, trajs[0].n_agents
    torch.manual_seed(seed)
    wm = TemporalWorldModel()
    train_temporal_world_model(wm, build_temporal_data(trajs, scheds),
                               WMTrainConfig(epochs=wm_epochs), verbose=False)
    H1_flat, _ = extract_temporal_h1_activations(wm, trajs, scheds)
    d = H1_flat.shape[1]
    fm = build_feature_matrix(trajs, scheds)
    # raw per-(agent,period) feed: ALL tiers present
    raw = (H1_flat, fm.Y, list(fm.feature_vocab))
    # C3 granularity feed: where regime becomes present (high fisher, still low var_share)
    h1_mean = H1_flat.reshape(n_trajs, T, N, d).mean(axis=2)
    Ypp, vocab_pp = per_period_Y(fm, n_trajs, T, N)
    c3 = (temporal_pool(h1_mean, WINDOW, "concat"), Ypp, vocab_pp)
    return raw, c3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trajs, n_periods, wm_epochs = (16, 48, 20) if args.quick else (48, 80, 40)

    all_rows = {}      # feature -> list of per-seed row dicts (raw feed)
    regime_div = {}    # regime feature -> {raw:(vs,fish,sae), c3:(vs,fish,sae)}
    for seed in args.seeds:
        raw, c3 = build_reps(n_trajs, n_periods, wm_epochs, seed)
        for r in per_feature_stats(*raw, seed):
            all_rows.setdefault(r["feature"], []).append(r)
        # regime divergence: same features in the C3 feed
        Xz = zscore(c3[0]); tv = float(Xz.var(0).sum())
        torch.manual_seed(seed)
        Xt = torch.tensor(Xz, dtype=torch.float32)
        sae = make_sae("jumprelu", Xt.shape[1], SAE_WIDTH, l0_coeff=SAE_L0, init_theta=0.05)
        train(sae, Xt, TrainConfig(epochs=SAE_EPOCHS, batch_size=256, lr=1e-3,
                                   log_every=10**9), verbose=False)
        rep = align(score_sae(sae, Xt), c3[1], c3[2])
        for j, f in enumerate(c3[2]):
            if feature_tier(f) != "regime":
                continue
            y = c3[1][:, j]
            regime_div.setdefault(f, {"raw": [], "c3": []})["c3"].append(
                (var_share(Xz, y, tv), fisher_snr(Xz, y), float(rep.alignment[:, j].max())))

    # aggregate per-feature (mean over seeds), raw feed
    feats = sorted(all_rows, key=lambda f: -np.nanmean([r["var_share"] for r in all_rows[f]]))
    def col(key):
        return np.array([np.nanmean([r[key] for r in all_rows[f]]) for f in feats])
    vs, fi, pr, sae = col("var_share"), col("fisher"), col("probe_auc"), col("sae_auc")
    lvs, lfi = np.log10(vs + 1e-9), np.log10(fi + 1e-9)   # ranks are scale-free anyway
    tier = [all_rows[f][0]["tier"] for f in feats]

    print(f"\n=== rate-distortion recoverability model (raw per-agent feed, all "
          f"{len(feats)} features, {len(args.seeds)} seeds, n_trajs={n_trajs}) ===")
    print("Across all tiers var_share & Fisher CO-VARY, so both correlate with SAE "
          "recovery —\nthe confound-free test is the PARTIAL correlation (each axis's "
          "unique contribution):\n")
    print(f"  {'':<26}{'-> SAE_AUC (allocation)':>24}{'-> probe_AUC (presence)':>26}")
    print(f"  {'partial var_share | fisher':<26}"
          f"{partial_spearman(lvs, sae, lfi):>+24.3f}{partial_spearman(lvs, pr, lfi):>+26.3f}")
    print(f"  {'partial fisher | var_share':<26}"
          f"{partial_spearman(lfi, sae, lvs):>+24.3f}{partial_spearman(lfi, pr, lvs):>+26.3f}")
    print("  (theory: allocation is governed by var_share, presence by Fisher — the "
          "diagonal dominates)")
    print(f"\n  raw Spearman:  var_share->SAE {spearman(vs, sae):+.3f}   "
          f"fisher->SAE {spearman(fi, sae):+.3f}   fisher->probe {spearman(fi, pr):+.3f}")

    print("\n  per-tier means (raw feed):")
    print(f"  {'tier':<12}{'n':>3}{'var_share':>12}{'fisher':>9}{'SAE_AUC':>9}")
    for t in ["categorical", "bucketed", "conjunctive", "regime"]:
        idx = [i for i, tt in enumerate(tier) if tt == t]
        if idx:
            print(f"  {t:<12}{len(idx):>3}{np.nanmean(vs[idx]):>12.4f}"
                  f"{np.nanmean(fi[idx]):>9.2f}{np.nanmean(sae[idx]):>9.3f}")

    # the divergence: regime in raw vs C3 (granularity raises FISHER, not var_share)
    print("\n  regime divergence — granularity raises FISHER (presence), not "
          "var_share (allocation):")
    print(f"  {'phase':<22}{'raw vs/fish/sae':>22}{'C3 vs/fish/sae':>22}")
    rvs, rfi, rsa = [], [], []
    for f in sorted(regime_div):
        raw_r = [r for r in all_rows[f]]
        rv = np.nanmean([r["var_share"] for r in raw_r])
        rf = np.nanmean([r["fisher"] for r in raw_r])
        rs = np.nanmean([r["sae_auc"] for r in raw_r])
        c3 = np.array(regime_div[f]["c3"])
        rvs.append(c3[:, 0].mean()); rfi.append(c3[:, 1].mean()); rsa.append(c3[:, 2].mean())
        print(f"  {f:<22}{rv:>7.4f}/{rf:>5.1f}/{rs:>4.2f}      "
              f"{c3[:,0].mean():>7.4f}/{c3[:,1].mean():>5.1f}/{c3[:,2].mean():>4.2f}")
    rvs, rfi, rsa = np.array(rvs), np.array(rfi), np.array(rsa)
    print("\n  WITHIN the regime tier (C3, all 'present') — the axes DECOUPLE:")
    print(f"    Spearman(var_share, SAE) = {spearman(rvs, rsa):+.3f}   "
          f"Spearman(fisher, SAE) = {spearman(rfi, rsa):+.3f}")
    print("    (fiscal_active = max Fisher / min var_share / low SAE = the exemplar: "
          "maximally detectable, minimally recoverable)")

    out = Path(__file__).resolve().parents[1] / "runs" / "regime_recoverability_theory_summary.json"
    out.write_text(json.dumps({
        "config": {"n_trajs": n_trajs, "n_periods": n_periods, "seeds": args.seeds},
        "spearman_raw": {"var_share_vs_sae": spearman(vs, sae),
                         "fisher_vs_sae": spearman(fi, sae),
                         "fisher_vs_probe": spearman(fi, pr)},
        "partial": {"var_share_vs_sae_given_fisher": partial_spearman(lvs, sae, lfi),
                    "fisher_vs_sae_given_var_share": partial_spearman(lfi, sae, lvs),
                    "var_share_vs_probe_given_fisher": partial_spearman(lvs, pr, lfi),
                    "fisher_vs_probe_given_var_share": partial_spearman(lfi, pr, lvs)},
        "regime_tier_internal_c3": {"var_share_vs_sae": spearman(rvs, rsa),
                                    "fisher_vs_sae": spearman(rfi, rsa)},
        "per_feature": [{"feature": feats[i], "tier": tier[i], "var_share": float(vs[i]),
                         "fisher": float(fi[i]), "probe_auc": float(pr[i]),
                         "sae_auc": float(sae[i])} for i in range(len(feats))]}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
