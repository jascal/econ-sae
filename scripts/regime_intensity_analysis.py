"""Diagnostic: do the SAEs encode smooth regime signals?

The temporal_sentiment_experiment showed that binary phase labels stay
at AUC ~0.55-0.61 even when the simulator's dynamics depend on regime.
The hypothesis in the README: the SAE substrate encodes regime intensity
as a smooth continuous signal, but binary threshold labels (`GDP[t] >
1.10 * trailing_mean`) can't be cleanly recovered from a smooth feature.

This script tests that hypothesis directly:

  1. Regenerate the sentiment-driven 128 x 100 ensemble.
  2. Load the trained TemporalWorldModel + SAE checkpoint
     (runs/temporal_sentiment_experiment/jr_w1024_ep200.pt).
  3. Score the SAE on the acts feed -> (N_samples, n_sae_features)
     activations Z.
  4. Compute several CONTINUOUS regime intensities per (traj, t):
       gdp_dev               (GDP[t] - trailing_mean) / trailing_mean
       leverage              debt_outstanding / money_stock
       rate                  interest_rate
       gdp_trend             OLS slope of GDP over the last 5 periods
       gdp_volatility        std of GDP over the last 5 periods
       cons_share_food       C_food / GDP
       cons_share_durables   C_durables / GDP
  5. For each (continuous label, SAE feature) pair compute Pearson
     correlation; report max |corr| per label and which feature scored.
  6. Cross-reference against the binary `phase:*` AUCs from the same
     SAE. If smooth correlation is high (>=0.8) while binary AUC is
     low (~0.6), the diagnosis is confirmed.

Output:
  prints a per-label comparison table; saves
  runs/regime_intensity_summary.json
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch

from econsae.ground_truth import build_feature_matrix
from econsae.sae.data import Feed
from econsae.sae.evaluation import align, score_sae, feature_tier
from econsae.sae.models import make_sae
from econsae.sae.world_model import (
    TemporalWorldModel, extract_temporal_h1_activations,
)
from econsae.simulator.ensemble import generate_ensemble


RUNS_DIR = os.path.join(REPO_ROOT, "runs")
SAE_CKPT = os.path.join(RUNS_DIR, "temporal_sentiment_experiment", "jr_w1024_ep200.pt")
WM_CKPT = os.path.join(RUNS_DIR, "world_model_temporal_sentiment.pt")


CONTINUOUS_LABELS = [
    "gdp_dev",
    "leverage",
    "rate",
    "gdp_trend",
    "gdp_volatility",
    "cons_share_food",
    "cons_share_durables",
]


def compute_continuous_labels(trajectories, window: int = 5) -> dict[str, np.ndarray]:
    """Per-(traj, t, agent) continuous regime intensities, packed as flat arrays.

    Most labels are period-level (same value across all 17 agents within a
    period). The flat ordering matches build_feature_matrix's sample_index
    (traj-major, period-major, agent-major).
    """
    n_trajs = len(trajectories)
    T = trajectories[0].T
    N = trajectories[0].n_agents
    n_samples = n_trajs * T * N

    out = {name: np.zeros(n_samples, dtype=np.float32) for name in CONTINUOUS_LABELS}
    for ti, traj in enumerate(trajectories):
        macros = traj.macros
        for t in range(T):
            lo = max(0, t - window)
            window_data = [m["GDP"] for m in macros[lo:t]]
            cur_gdp = macros[t]["GDP"]
            trailing = float(np.mean(window_data)) if window_data else float(cur_gdp)
            gdp_dev = (cur_gdp - trailing) / max(trailing, 1e-9)
            leverage = macros[t]["debt_outstanding"] / max(macros[t]["money_stock"], 1e-9)
            rate = macros[t]["interest_rate"]

            if len(window_data) >= 2:
                vals = np.array(window_data + [cur_gdp], dtype=np.float64)
                xs = np.arange(len(vals), dtype=np.float64)
                gdp_trend = float(np.polyfit(xs, vals, 1)[0])
                gdp_vol = float(np.std(vals))
            else:
                gdp_trend = 0.0
                gdp_vol = 0.0

            denom = max(cur_gdp, 1e-9)
            cs_food = macros[t]["C_food"] / denom
            cs_dur = macros[t]["C_durables"] / denom

            base = (ti * T + t) * N
            for ai in range(N):
                idx = base + ai
                out["gdp_dev"][idx] = gdp_dev
                out["leverage"][idx] = leverage
                out["rate"][idx] = rate
                out["gdp_trend"][idx] = gdp_trend
                out["gdp_volatility"][idx] = gdp_vol
                out["cons_share_food"][idx] = cs_food
                out["cons_share_durables"][idx] = cs_dur
    return out


def max_abs_correlation(Z: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Best SAE feature index and signed correlation against scalar label y."""
    # Center
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    Zn = Zc.std(axis=0).clip(min=1e-9)
    yn = float(yc.std()) if float(yc.std()) > 0 else 1e-9
    # Correlation per feature
    corr = (Zc * yc[:, None]).mean(axis=0) / (Zn * yn)
    # Mask any constant columns
    active = (Z.max(axis=0) - Z.min(axis=0)) > 1e-9
    corr = np.where(active, corr, 0.0)
    best = int(np.argmax(np.abs(corr)))
    return best, float(corr[best])


def load_world_model() -> TemporalWorldModel:
    ckpt = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = TemporalWorldModel(**cfg)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def load_sae():
    obj = torch.load(SAE_CKPT, map_location="cpu", weights_only=False)
    cfg = obj.get("feed_config", {})
    sae = make_sae("jumprelu", obj["input_dim"], obj["n_features"],
                   l0_coeff=cfg.get("l0_coeff", 1.5e-3),
                   init_theta=cfg.get("init_theta", 0.05))
    sae.load_state_dict(obj["state_dict"])
    sae.eval()
    return sae


def main(seed: int = 0, n_trajectories: int = 128, n_periods: int = 100,
         sentiment_strength: float = 0.20):
    print("=" * 78)
    print("Regime intensity analysis (continuous labels via correlation)")
    print("=" * 78)

    ens = generate_ensemble(n_trajectories=n_trajectories,
                            n_periods=n_periods, seed=seed,
                            sentiment_strength=sentiment_strength)
    print(f"\nensemble: {len(ens)} trajectories x {n_periods} periods")

    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)

    wm = load_world_model()
    H1, idx = extract_temporal_h1_activations(wm, ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index, "activation index mismatch"
    feed = Feed(
        name="acts_temporal_sentiment_d192",
        X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
    )

    sae = load_sae()
    Z = score_sae(sae, feed.X)
    print(f"SAE activations: {Z.shape}  active features: "
          f"{int((Z.max(axis=0) > 1e-9).sum())}/{Z.shape[1]}")

    # Continuous labels
    labels = compute_continuous_labels(ens.trajectories)

    # Binary AUC for comparison (uses align())
    rep = align(Z, fm.Y, fm.feature_vocab)

    # --- per-label correlation report ---
    print("\nContinuous regime-intensity correlations (max |Pearson| over SAE features):")
    print(f"  {'label':<30s} {'corr':>7s} {'best#':>6s}")
    print("  " + "-" * 50)
    cont_results: list[dict] = []
    for name in CONTINUOUS_LABELS:
        best_idx, corr = max_abs_correlation(Z, labels[name])
        print(f"  {name:<30s} {corr:>7.3f} {best_idx:>6d}")
        cont_results.append({"label": name, "best_corr": corr,
                              "best_sae_feature": best_idx})

    # --- cross-reference: binary phase AUCs (same SAE) ---
    print("\nBinary phase AUCs from the same SAE (for comparison):")
    print(f"  {'label':<30s} {'AUC':>7s}")
    print("  " + "-" * 50)
    bin_results: list[dict] = []
    for j, name in enumerate(fm.feature_vocab):
        if feature_tier(name) == "regime":
            auc = float(rep.alignment[:, j].max())
            print(f"  {name:<30s} {auc:>7.3f}")
            bin_results.append({"label": name, "max_auc": auc})

    # --- side-by-side ---
    print("\nDiagnosis: SMOOTH (corr) vs THRESHOLDED (AUC) recovery:")
    print(f"  {'continuous label':<30s} {'corr':>7s} | "
          f"{'binary label':<28s} {'AUC':>7s}")
    print("  " + "-" * 80)
    pairs = [
        ("gdp_dev",                ("phase:expansion", "phase:contraction")),
        ("leverage",               ("phase:high_leverage",)),
        ("rate",                   ("phase:high_rate",)),
        ("gdp_trend",              ("phase:expansion", "phase:contraction")),
        ("cons_share_durables",   ()),
    ]
    for cont_name, bin_names in pairs:
        c = next(r for r in cont_results if r["label"] == cont_name)
        if bin_names:
            for bn in bin_names:
                b = next((r for r in bin_results if r["label"] == bn), None)
                if b:
                    print(f"  {cont_name:<30s} {c['best_corr']:>7.3f} | "
                          f"{bn:<28s} {b['max_auc']:>7.3f}")
        else:
            print(f"  {cont_name:<30s} {c['best_corr']:>7.3f} | "
                  f"{'—':<28s} {'—':>7s}")

    out = {
        "continuous": cont_results,
        "binary_regime_aucs": bin_results,
        "sae_ckpt": SAE_CKPT,
        "wm_ckpt": WM_CKPT,
        "n_samples": int(Z.shape[0]),
        "n_sae_features": int(Z.shape[1]),
    }
    out_path = os.path.join(RUNS_DIR, "regime_intensity_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
