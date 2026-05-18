"""AUC alignment scoring with per-tier breakdown.

The core question: a trained SAE has produced N learned features; the
ground-truth vocabulary has M known features. For each (sae_feature f,
gt_feature g) pair we compute the AUC of "sample i has feature g" vs
"sae feature f activates on sample i". AUC is chance-corrected and
threshold-free, working uniformly for binary (TopK) and continuous (L1,
JumpReLU) SAE activations.

Per-tier breakdown (NEW for econ-sae)
-------------------------------------
The vocabulary spans four tiers of difficulty:

  CATEGORICAL  -- sector / cohort / firm_sector / txn-kind-present / shock-kind-present
                  ("easy"; should be recovered cleanly by any decent SAE)
  BUCKETED     -- continuous coords binned by quantile
                  ("medium"; requires the SAE to find an axis-aligned activation pattern)
  CONJUNCTIVE  -- 2- or 3-condition compositions
                  ("hard"; the polysemantic-trap target)
  REGIME       -- macro-derived phase labels
                  ("medium-hard"; requires aggregating over a window)

The headline metric for econ-sae is *per-tier coverage and mean-best
AUC*. A score of 0.95 coverage on categorical with 0.4 coverage on
conjunctive means the SAE recovered the easy structure but missed the
hard structure --- exactly the failure mode econ-sae was designed to
expose. The opposite (high conjunctive recovery) would indicate the SAE
has internalized the non-trivial joint statistics; that's the
interpretability win.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Single-pair ROC-AUC (kept for documentation; align() uses the
    vectorized variant below which is orders of magnitude faster)."""
    return float(_auc_one_vs_many(scores, labels.reshape(-1, 1))[0])


def _auc_one_vs_many(scores: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """AUC of `scores` against EACH column of binary Y in one shot.

    Vectorized via the Mann-Whitney identity:
        AUC = (sum_of_ascending_ranks_of_positives - n_pos*(n_pos-1)/2) / (n_pos*n_neg)

    scores: (N,)              -- continuous SAE activations for one feature
    Y     : (N, M) {0,1}      -- binary labels for M ground-truth features

    Returns: (M,) array of AUCs. Returns 0.5 for any column with no positives
    or no negatives (chance / undefined).
    """
    N = scores.shape[0]
    M = Y.shape[1]
    # Ascending rank of each sample in score order. Ties broken stably.
    order = np.argsort(scores, kind="stable")
    ranks_asc = np.empty(N, dtype=np.float64)
    ranks_asc[order] = np.arange(N, dtype=np.float64)
    n_pos = Y.sum(axis=0)                                # (M,)
    sum_ranks = ranks_asc @ Y                            # (M,)
    n_neg = N - n_pos
    aucs = np.full(M, 0.5, dtype=np.float64)
    valid = (n_pos > 0) & (n_neg > 0)
    aucs[valid] = (
        (sum_ranks[valid] - n_pos[valid] * (n_pos[valid] - 1.0) / 2.0)
        / (n_pos[valid] * n_neg[valid])
    )
    return aucs.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature tiering
# ---------------------------------------------------------------------------
def feature_tier(name: str) -> str:
    """Return the difficulty tier of a ground-truth feature name."""
    if name.startswith(("sector:", "cohort:", "firm_sector:",
                        "txn_period_has:", "shock_period_has:")):
        return "categorical"
    if name.startswith("phase:"):
        return "regime"
    if name.startswith(("debt_bucket:", "cash_bucket:", "mpc_bucket:",
                        "leverage:", "inventory_bucket:", "productivity:")):
        return "bucketed"
    if "AND" in name or name in (
        "food_firm_low_inv", "services_firm_high_output",
        "durables_firm_high_inv",
    ):
        return "conjunctive"
    return "other"


TIERS = ("categorical", "bucketed", "conjunctive", "regime")


# ---------------------------------------------------------------------------
@dataclass
class AlignmentReport:
    run_id: str = ""
    feed_name: str = ""
    variant: str = ""
    sae_feature_count: int = 0
    gt_feature_count: int = 0
    alignment: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    gt_vocab: list[str] = field(default_factory=list)
    sae_features_active: int = 0
    coverage_at_0_95: float = 0.0
    coverage_at_0_90: float = 0.0
    mean_best_auc: float = 0.0
    monosemanticity: float = 0.0
    top_matches: list[dict] = field(default_factory=list)
    # Per-tier coverage and mean-best AUC.
    per_tier: dict[str, dict[str, float]] = field(default_factory=dict)


def score_sae(sae, X: torch.Tensor) -> np.ndarray:
    """(N, n_features) activation magnitudes."""
    sae.eval()
    with torch.no_grad():
        out = sae(X)
    return out.z.detach().cpu().numpy()


def align(Z: np.ndarray, Y: np.ndarray, gt_vocab: list[str]) -> AlignmentReport:
    """Compute the full alignment matrix and per-tier aggregates."""
    N, n_sae = Z.shape
    M = Y.shape[1]
    active_mask = (Z.max(axis=0) > 1e-9)
    A = np.full((n_sae, M), 0.5, dtype=np.float32)
    Yf = Y.astype(np.float64)
    for f in range(n_sae):
        if not active_mask[f]:
            continue
        A[f] = _auc_one_vs_many(Z[:, f].astype(np.float64), Yf)

    if A.shape[0] > 0:
        best_auc_per_gt = A.max(axis=0)
    else:
        best_auc_per_gt = np.full(M, 0.5, dtype=np.float32)
    coverage_95 = float((best_auc_per_gt >= 0.95).mean())
    coverage_90 = float((best_auc_per_gt >= 0.90).mean())
    mean_best = float(best_auc_per_gt.mean())

    # monosemanticity: fraction of active SAE features with a clean GT match
    n_active = int(active_mask.sum())
    monosem_count = 0
    if n_active > 0:
        for f in range(n_sae):
            if active_mask[f] and A[f].max() >= 0.95:
                monosem_count += 1
    monosem = monosem_count / max(n_active, 1)

    # Per-tier breakdown
    per_tier: dict[str, dict[str, float]] = {}
    for tier in TIERS:
        idxs = [j for j, name in enumerate(gt_vocab) if feature_tier(name) == tier]
        if not idxs:
            per_tier[tier] = {"n_features": 0, "coverage_0.95": 0.0,
                              "coverage_0.90": 0.0, "mean_best_auc": 0.0}
            continue
        sub = best_auc_per_gt[idxs]
        per_tier[tier] = {
            "n_features": len(idxs),
            "coverage_0.95": float((sub >= 0.95).mean()),
            "coverage_0.90": float((sub >= 0.90).mean()),
            "mean_best_auc": float(sub.mean()),
        }

    # top matches above 0.95
    top_matches = []
    for j in range(M):
        if best_auc_per_gt[j] >= 0.95:
            f_best = int(A[:, j].argmax())
            top_matches.append({
                "gt_feature": gt_vocab[j],
                "tier": feature_tier(gt_vocab[j]),
                "sae_feature_idx": f_best,
                "auc": float(A[f_best, j]),
            })
    top_matches.sort(key=lambda m: -m["auc"])

    return AlignmentReport(
        sae_feature_count=n_sae,
        gt_feature_count=M,
        alignment=A, gt_vocab=gt_vocab,
        sae_features_active=n_active,
        coverage_at_0_95=coverage_95,
        coverage_at_0_90=coverage_90,
        mean_best_auc=mean_best,
        monosemanticity=monosem,
        top_matches=top_matches,
        per_tier=per_tier,
    )


# ---------------------------------------------------------------------------
def format_report_table(reports: list[AlignmentReport]) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("ALIGNMENT REPORT  (per-tier mean-best AUC and coverage@0.95)")
    lines.append("=" * 100)
    hdr = (f"{'feed':<14s} {'variant':<10s} {'n_sae':>5s} {'act':>4s} "
           f"{'cov95':>6s} {'mAUC':>6s} {'mono':>6s}  "
           + "  ".join(f"{t[:4]+'/95':>8s}" for t in TIERS) + "  "
           + "  ".join(f"{t[:4]+'/A':>7s}" for t in TIERS))
    lines.append(hdr)
    lines.append("-" * 100)
    for r in reports:
        row = (f"{r.feed_name:<14s} {r.variant:<10s} "
               f"{r.sae_feature_count:>5d} {r.sae_features_active:>4d} "
               f"{r.coverage_at_0_95:>6.1%} {r.mean_best_auc:>6.3f} "
               f"{r.monosemanticity:>6.1%}  "
               + "  ".join(f"{r.per_tier[t]['coverage_0.95']:>8.1%}" for t in TIERS) + "  "
               + "  ".join(f"{r.per_tier[t]['mean_best_auc']:>7.3f}" for t in TIERS))
        lines.append(row)
    return "\n".join(lines)


def report_to_dict(r: AlignmentReport) -> dict:
    return {
        "run_id": r.run_id,
        "feed": r.feed_name, "variant": r.variant,
        "n_sae_features": r.sae_feature_count,
        "n_active": r.sae_features_active,
        "n_gt_features": r.gt_feature_count,
        "coverage_0.95": r.coverage_at_0_95,
        "coverage_0.90": r.coverage_at_0_90,
        "mean_best_auc": r.mean_best_auc,
        "monosemanticity": r.monosemanticity,
        "n_top_matches": len(r.top_matches),
        "per_tier": r.per_tier,
    }
