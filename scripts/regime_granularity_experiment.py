"""Held-out label-free recovery test: is the econ-regime unsupervised ceiling a
GRANULARITY artifact (the structural-match law) or a genuine LABEL dependence?

This is the experiment SUPERVISION_DEPENDENCE.md / RESEARCH_MANIFESTO.md Reckoning #1
calls for and leaves out of scope: "does an SAE trained WITHOUT the regime labels
recover them when scored at matched granularity?"

We isolate the *granularity* knob on the SAME unsupervised TemporalWorldModel h1
(next-state objective only — NO regime labels anywhere in WM or SAE training; labels
are used ONLY to SCORE). Conditions, increasingly granularity-matched to the
window-aggregated, agent-invariant regime tier:

  C0_raw       raw per-(agent, period) h1            -> the published floor (~0.605/0.00)
  C1_agentpool mean over agents, per period          (fix: regime is agent-invariant)
  C2_timemean  C1 + mean over a trailing window W     (fix: regime is window-aggregated)
  C3_timecat   C1 + concat of the trailing window W   (richer temporal granularity)
  Cref_macro   macro_feed_v3 (engineered, label-free) -> reference (~0.885/3-of-6)

For each condition we report the UNSUPERVISED JumpReLU SAE regime recovery
(mAUC / cov95 / per-feature) AND a ridge-LDA linear-PROBE ceiling (is regime even
linearly PRESENT in that representation?). The probe decomposes the law's components:

  probe high & SAE low  -> regime is THERE but unsupervised discovery doesn't surface it
                           (an SAE-objective/sparsity limit, not a substrate absence)
  probe low  & SAE low  -> regime is NOT in unsupervised h1 at that granularity
                           (the WM objective didn't reward encoding it -> needs the signal
                            injected, by engineering OR supervision)

Run:  .venv/bin/python scripts/regime_granularity_experiment.py [--seeds 0 1 2] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for macro_feed_v3

from econsae.ground_truth import build_feature_matrix
from econsae.sae.evaluation import align, feature_tier, score_sae
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, train
from econsae.sae.world_model import (
    TemporalWorldModel,
    WMTrainConfig,
    build_temporal_data,
    extract_temporal_h1_activations,
    train_temporal_world_model,
)
from econsae.simulator.ensemble import generate_ensemble
from macro_feed_v3_experiment import PER_PERIOD_PREFIXES, build_macro_feed_v3

REGIME = "regime"
WINDOW = 5  # matches the regime labels' 5-period trailing GDP window


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (rank-based)."""
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def ridge_lda_probe_auc(X: np.ndarray, y: np.ndarray, lam: float = 1.0,
                        seed: int = 0) -> float:
    """Closed-form ridge-LDA linear probe; train/test split; held-out ROC-AUC.
    A label-using diagnostic of how much of `y` is *linearly present* in X."""
    y = y.astype(bool)
    if y.sum() < 3 or (~y).sum() < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr = X[tr], y[tr]
    if ytr.sum() < 2 or (~ytr).sum() < 2 or y[te].sum() < 1 or (~y[te]).sum() < 1:
        return float("nan")
    mu_p = Xtr[ytr].mean(0)
    mu_n = Xtr[~ytr].mean(0)
    Xc = Xtr - Xtr.mean(0)
    cov = (Xc.T @ Xc) / max(len(Xtr) - 1, 1)
    d = X.shape[1]
    w = np.linalg.solve(cov + lam * np.eye(d), mu_p - mu_n)
    return auc(X[te] @ w, y[te])


def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0)
    sd = X.std(0).clip(min=1e-6)
    return ((X - mu) / sd).astype(np.float32)


def per_period_Y(fm, n_trajs: int, T: int, N: int):
    """Per-period label matrix (agent 0 representative) for the per-period tiers."""
    pp = [f for f in fm.feature_vocab if any(f.startswith(p) for p in PER_PERIOD_PREFIXES)]
    idx = {f: j for j, f in enumerate(fm.feature_vocab)}
    Y = np.zeros((n_trajs * T, len(pp)), dtype=np.uint8)
    for ti in range(n_trajs):
        for t in range(T):
            row = (ti * T + t) * N + 0
            for j, f in enumerate(pp):
                if fm.Y[row, idx[f]] > 0:
                    Y[ti * T + t, j] = 1
    return Y, pp


def temporal_pool(h1_mean: np.ndarray, W: int, mode: str) -> np.ndarray:
    """h1_mean: (n_trajs, T, d). Trailing window of W (causal), mean or concat."""
    n_trajs, T, d = h1_mean.shape
    if mode == "mean":
        out = np.zeros((n_trajs, T, d), dtype=np.float32)
        for t in range(T):
            lo = max(0, t - W + 1)
            out[:, t] = h1_mean[:, lo:t + 1].mean(axis=1)
        return out.reshape(n_trajs * T, d)
    # concat: pad the start by repeating the earliest available period
    out = np.zeros((n_trajs, T, d * W), dtype=np.float32)
    for t in range(T):
        cols = []
        for k in range(W - 1, -1, -1):
            cols.append(h1_mean[:, max(0, t - k)])
        out[:, t] = np.concatenate(cols, axis=-1)
    return out.reshape(n_trajs * T, d * W)


def regime_recovery(X: np.ndarray, Y: np.ndarray, vocab: list[str], seed: int):
    """Train an unsupervised JumpReLU SAE on X, score regime tier + per-feature;
    plus the ridge-LDA probe ceiling per regime feature."""
    torch.manual_seed(seed)
    Xt = torch.tensor(zscore(X), dtype=torch.float32)
    sae = make_sae("jumprelu", Xt.shape[1], 256, l0_coeff=1e-3, init_theta=0.05)
    train(sae, Xt, TrainConfig(epochs=200, batch_size=256, lr=1e-3, log_every=10**9),
          verbose=False)
    Z = score_sae(sae, Xt)
    rep = align(Z, Y, vocab)
    pt = rep.per_tier[REGIME]
    reg_idx = [j for j, f in enumerate(vocab) if feature_tier(f) == REGIME]
    sae_pf = {vocab[j]: float(rep.alignment[:, j].max()) for j in reg_idx}
    Xz = zscore(X)
    probe_pf = {vocab[j]: ridge_lda_probe_auc(Xz, Y[:, j], seed=seed) for j in reg_idx}
    return {
        "dim": int(X.shape[1]),
        "sae_regime_mauc": float(pt["mean_best_auc"]),
        "sae_regime_cov95": float(pt["coverage_0.95"]),
        "sae_regime_per_feature": sae_pf,
        "probe_regime_mean": float(np.nanmean(list(probe_pf.values()))),
        "probe_regime_cov95": float(np.mean([v >= 0.95 for v in probe_pf.values()
                                             if not np.isnan(v)])),
        "probe_regime_per_feature": probe_pf,
    }


def run_seed(n_trajs: int, n_periods: int, wm_epochs: int, seed: int) -> dict:
    ens = generate_ensemble(n_trajectories=n_trajs, n_periods=n_periods, seed=seed)
    trajs, scheds = ens.trajectories, ens.shock_schedules
    T, N = trajs[0].T, trajs[0].n_agents

    # ---- unsupervised TemporalWorldModel (next-state objective ONLY) ----
    torch.manual_seed(seed)
    wm = TemporalWorldModel()
    train_temporal_world_model(
        wm, build_temporal_data(trajs, scheds),
        WMTrainConfig(epochs=wm_epochs), verbose=False)

    H1_flat, _ = extract_temporal_h1_activations(wm, trajs, scheds)
    d = H1_flat.shape[1]
    H1 = H1_flat.reshape(n_trajs, T, N, d)
    h1_mean = H1.mean(axis=2)  # (n_trajs, T, d), agent-pooled per period

    fm = build_feature_matrix(trajs, scheds)
    Ypp, vocab_pp = per_period_Y(fm, n_trajs, T, N)

    conds = {}
    # C0: raw per-(agent, period) — the published floor; per-agent GT.
    conds["C0_raw"] = regime_recovery(H1_flat, fm.Y, list(fm.feature_vocab), seed)
    # C1: agent-pooled, per period.
    conds["C1_agentpool"] = regime_recovery(
        h1_mean.reshape(n_trajs * T, d), Ypp, vocab_pp, seed)
    # C2: agent-pooled + trailing-window MEAN.
    conds["C2_timemean"] = regime_recovery(
        temporal_pool(h1_mean, WINDOW, "mean"), Ypp, vocab_pp, seed)
    # C3: agent-pooled + trailing-window CONCAT.
    conds["C3_timecat"] = regime_recovery(
        temporal_pool(h1_mean, WINDOW, "concat"), Ypp, vocab_pp, seed)
    # Cref: macro_feed_v3 (engineered, label-free) — the known reference.
    feed = build_macro_feed_v3(trajs, scheds, wm)
    conds["Cref_macro"] = regime_recovery(
        feed.X.numpy(), feed.Y, feed.feature_vocab, seed)

    return {"seed": seed, "n_trajs": n_trajs, "n_periods": n_periods,
            "window": WINDOW, "conditions": conds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="small/fast dev run")
    args = ap.parse_args()
    n_trajs, n_periods, wm_epochs = (16, 48, 20) if args.quick else (48, 80, 40)

    results = [run_seed(n_trajs, n_periods, wm_epochs, s) for s in args.seeds]

    # aggregate across seeds
    order = ["C0_raw", "C1_agentpool", "C2_timemean", "C3_timecat", "Cref_macro"]
    print(f"\n=== regime recovery vs granularity (mean ± sd over {len(args.seeds)} "
          f"seeds; n_trajs={n_trajs}, n_periods={n_periods}) ===")
    print(f"{'condition':<14}{'dim':>6}  {'SAE mAUC':>10}  {'SAE cov95':>10}  "
          f"{'probe mAUC':>11}  {'probe cov95':>12}")
    agg = {}
    for c in order:
        def col(key):
            return np.array([r["conditions"][c][key] for r in results], dtype=float)
        sm, sc = col("sae_regime_mauc"), col("sae_regime_cov95")
        pm, pc = col("probe_regime_mean"), col("probe_regime_cov95")
        dim = results[0]["conditions"][c]["dim"]
        agg[c] = {"sae_mauc": [float(sm.mean()), float(sm.std())],
                  "sae_cov95": [float(sc.mean()), float(sc.std())],
                  "probe_mauc": [float(pm.mean()), float(pm.std())],
                  "probe_cov95": [float(pc.mean()), float(pc.std())], "dim": int(dim)}
        print(f"{c:<14}{dim:>6}  {sm.mean():>6.3f}±{sm.std():<3.2f}  "
              f"{sc.mean():>6.1%}±{sc.std():<4.1%}  {pm.mean():>6.3f}±{pm.std():<3.2f}  "
              f"{pc.mean():>6.1%}±{pc.std():<4.1%}")

    # per-feature (seed 0) to see WHICH phases recover where
    print("\n=== regime per-feature SAE AUC (seed 0) ===")
    feats = list(results[0]["conditions"]["C0_raw"]["sae_regime_per_feature"].keys())
    print(f"{'phase':<22}" + "".join(f"{c.split('_')[0]:>9}" for c in order))
    for f in feats:
        row = "".join(
            f"{results[0]['conditions'][c]['sae_regime_per_feature'].get(f, float('nan')):>9.3f}"
            for c in order)
        print(f"{f:<22}{row}")

    out = Path(__file__).resolve().parents[1] / "runs" / "regime_granularity_summary.json"
    out.write_text(json.dumps({"config": {"n_trajs": n_trajs, "n_periods": n_periods,
                                          "wm_epochs": wm_epochs, "seeds": args.seeds,
                                          "window": WINDOW},
                               "aggregate": agg, "per_seed": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
