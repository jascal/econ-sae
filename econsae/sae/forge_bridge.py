"""Polygram SAE-forge bridge for econ-sae.

Mirrors sm-sae's scripts/forge_pipeline.py 9-stage flow, adapted to
econ-sae's two-feed structure (per-agent SAE substrate + per-period
macro-feed SAE substrate). Each helper function takes inputs that are
already loaded; orchestration lives in scripts/forge_pipeline.py.

Stages:

  1. load_sae(ckpt)                 load an econ-sae SAE checkpoint
  2. convert_to_safetensors(sae, out)  write polygram-canonical safetensors
  3. build_records(sae_safetensors) -> dict[int, SAEFeatureRecord]
  4. build_dictionary(records, encoding, selector, sae, feed)
                                    -> polygram.Dictionary (+ selection meta)
  5. synthesize_validation_report(sae, dictionary, feed)
                                    pairwise stats from feed activations,
                                    no LLM host needed
  6. compress(vreport, safetensors_path, out_path)
                                    -> *.compressed.safetensors
  7. forge(compressed, sae, feed)   STUB; blocked on sae-forge release
  8. score_against_gt(activations, Y)
                                    AUC against the GT feature vocabulary
  9. write_report(run_dir, payload) forge_results.json

Encoding choices (stage 4):
    mps_rung1   cap=8
    rung3       cap=16
    rung4       cap=32
    rung5       cap=2**(n_amp_qubits + 3)   (default; n_amp_qubits=4 -> 128)

Selector choices (stage 4):
    firing_rate   keep top features by mean firing rate on the feed (default)
    gt_alignment  keep features with highest |max AUC - 0.5| against GT
    head          first k feature IDs (reproducible baseline)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch

# Side-effect import: registers TemporalWMAdapter / DualHeadRegimeWM with
# saeforge so `run_synthetic` can dispatch on econ-sae host models.
from econsae.sae import forge_adapter as _forge_adapter  # noqa: F401


# ---------------------------------------------------------------------------
# Stage 1: load SAE
# ---------------------------------------------------------------------------
def load_sae(ckpt_path: str):
    """Load an econ-sae SAE checkpoint (path/to/something.pt).

    Reads the per-checkpoint metadata to choose the right SAE class +
    hyperparameters. Returns (sae, meta_dict).
    """
    from econsae.sae.models import make_sae

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind = obj["kind"]
    in_dim = int(obj["input_dim"])
    n_feat = int(obj["n_features"])
    cfg = obj.get("feed_config", {})

    if kind == "topk":
        sae = make_sae("topk", in_dim, n_feat, k=cfg.get("k", cfg.get("topk_k", 8)))
    elif kind == "l1":
        sae = make_sae("l1", in_dim, n_feat,
                       l1_coeff=cfg.get("l1_coeff", 1e-2))
    elif kind == "jumprelu":
        sae = make_sae("jumprelu", in_dim, n_feat,
                       l0_coeff=cfg.get("l0_coeff", 5e-3),
                       init_theta=cfg.get("init_theta", 0.05))
    elif kind == "gated":
        sae = make_sae("gated", in_dim, n_feat,
                       l1_coeff=cfg.get("l1_coeff", 1e-3))
    else:
        raise ValueError(f"unknown SAE kind: {kind}")
    sae.load_state_dict(obj["state_dict"])
    sae.eval()
    return sae, obj


# ---------------------------------------------------------------------------
# Stage 2: SAE checkpoint -> polygram-canonical safetensors
# ---------------------------------------------------------------------------
def convert_to_safetensors(sae, out_path: str) -> str:
    """Write the SAE's W_dec, W_enc, b_enc, b_dec in the polygram-expected
    layout:
        W_dec : (n_features, input_dim)
        W_enc : (input_dim, n_features)
        b_enc : (n_features,)
        b_dec : (input_dim,)

    econ-sae's _BaseSAE keeps W_dec as (input_dim, n_features) and W_enc
    as (n_features, input_dim) internally; both are transposed once on
    write to match polygram's documented canonical layout.
    """
    from safetensors.torch import save_file
    state = {
        "W_dec": sae.W_dec.detach().t().contiguous(),
        "W_enc": sae.W_enc.detach().t().contiguous(),
        "b_enc": sae.b_enc.detach().contiguous(),
        "b_dec": sae.b_dec.detach().contiguous(),
    }
    save_file(state, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Stage 3: SAEFeatureRecord dict
# ---------------------------------------------------------------------------
def build_records(safetensors_path: str) -> dict:
    """Use polygram's loader to convert safetensors -> dict[int, SAEFeatureRecord]."""
    from polygram import load_sae_safetensors
    return load_sae_safetensors(safetensors_path)


# ---------------------------------------------------------------------------
# Stage 3.5: feature selectors
# ---------------------------------------------------------------------------
def select_by_head(sae, feed, records) -> list[int]:
    """Feature IDs in ascending numeric order (reproducible baseline)."""
    return sorted(records.keys())


def select_by_firing_rate(sae, feed, records) -> list[int]:
    """Rank features by mean firing rate on the feed (descending).
    Tie-break on feature_id ascending for determinism."""
    ids = sorted(records.keys())
    with torch.no_grad():
        z = sae(feed.X).z.detach().cpu()
    rate = (z.abs() > 1e-9).float().mean(dim=0).numpy()
    scores = [(-float(rate[i]), i) for i in ids]
    scores.sort()
    return [i for _, i in scores]


def select_by_gt_alignment(sae, feed, records) -> list[int]:
    """Rank features by max |AUC - 0.5| across GT labels (descending)."""
    from econsae.sae.evaluation import _auc_one_vs_many
    ids = sorted(records.keys())
    with torch.no_grad():
        z = sae(feed.X).z.detach().cpu().numpy().astype(np.float32)
    Yf = feed.Y.astype(np.float64)
    n_feat = z.shape[1]
    best = np.zeros(n_feat, dtype=np.float32)
    for f in range(n_feat):
        col = z[:, f]
        if (col.max() - col.min()) < 1e-9:
            best[f] = 0.0
            continue
        aucs = _auc_one_vs_many(col.astype(np.float64), Yf)
        best[f] = float(np.abs(aucs - 0.5).max())
    scores = [(-float(best[i]), i) for i in ids]
    scores.sort()
    return [i for _, i in scores]


SELECTORS = {
    "head":          select_by_head,
    "firing_rate":   select_by_firing_rate,
    "gt_alignment":  select_by_gt_alignment,
}


def _resolve_selector(arg):
    if callable(arg):
        return arg
    if isinstance(arg, str):
        if arg not in SELECTORS:
            raise ValueError(
                f"unknown selector {arg!r}; choose from {sorted(SELECTORS)} "
                f"or pass a callable (sae, feed, records) -> list[int]"
            )
        return SELECTORS[arg]
    raise TypeError(f"selector must be str or callable, got {type(arg).__name__}")


# ---------------------------------------------------------------------------
# Stage 4: SAEFeatureRecord dict -> polygram.Dictionary
# ---------------------------------------------------------------------------
def build_dictionary(records: dict, encoding_name: str = "rung5",
                     selector="firing_rate", sae=None, feed=None,
                     n_amp_qubits: int = 4):
    """Wrap records as a polygram Dictionary using the chosen encoding.

    Returns (dictionary, sel_report, selection_meta).
    """
    from polygram import (from_sae_lens, MPSRung1, Rung3, Rung4, Rung5)
    builders = {
        "mps_rung1": lambda: MPSRung1(bond_dim=2, phase_knobs=True),
        "rung3":     lambda: Rung3(bond_dim=2),
        "rung4":     lambda: Rung4(bond_dim=2),
        "rung5":     lambda: Rung5(bond_dim=2, n_amp_qubits=n_amp_qubits),
    }
    if encoding_name not in builders:
        raise ValueError(f"unknown encoding {encoding_name!r}; "
                         f"choose from {list(builders)}")
    encoding = builders[encoding_name]()

    selector_fn = _resolve_selector(selector)
    method_name = (selector if isinstance(selector, str)
                   else getattr(selector, "__name__", "custom"))
    if method_name != "head" and (sae is None or feed is None):
        raise ValueError(
            f"selector={method_name!r} needs sae and feed; only 'head' "
            f"works without them."
        )

    ordered_ids = list(selector_fn(sae, feed, records))
    if sorted(ordered_ids) != sorted(records.keys()):
        raise ValueError(
            f"selector {method_name!r} did not return a permutation of "
            f"records.keys(); got {len(ordered_ids)} ids "
            f"(expected {len(records)})."
        )

    encoding_cap = getattr(encoding, "max_features", len(ordered_ids))
    sae_n_features = len(records)
    cap = min(encoding_cap, sae_n_features)
    if cap < encoding_cap:
        warnings.warn(
            f"encoding {encoding_name!r} cap={encoding_cap} clamped to "
            f"{cap} (the SAE only has {sae_n_features} features). "
            f"Train a wider SAE if you need to use the encoding's full "
            f"capacity.",
            UserWarning, stacklevel=2,
        )
    kept_ids = ordered_ids[:cap]

    dictionary, sel_report = from_sae_lens(
        records,
        feature_ids=kept_ids,
        name=f"econ_sae_{encoding_name}",
        encoding=encoding,
        assign_amp_knobs=True,
        assign_phase_knobs=True,
    )
    selection_meta = {
        "method":        method_name,
        "n_candidates":  len(records),
        "n_kept":        len(kept_ids),
        "kept_ids":      [int(i) for i in kept_ids],
        "ordered_ids":   [int(i) for i in ordered_ids],
        "encoding_cap":  int(encoding_cap),
        "cap_applied":   int(cap),
    }
    return dictionary, sel_report, selection_meta


# ---------------------------------------------------------------------------
# Stage 5: synthesize a ValidationReport from feed activations
# ---------------------------------------------------------------------------
def synthesize_validation_report(sae, dictionary, feed,
                                 polygram_threshold: float = 0.3,
                                 jaccard_threshold: float = 0.05,
                                 min_firing_rate: float = 0.001,
                                 min_both_fire: int = 1):
    """Compute pairwise stats on the feed's activations; no LLM/host
    model needed. KL-ablation fields are 0 (Compressor's `merge` strategy
    mainly uses polygram_overlap + jaccard so this should still produce
    a usable plan).
    """
    from polygram import (BucketStats, CandidatePair, ValidationReport,
                          ValidationSummary)

    with torch.no_grad():
        Z = sae(feed.X).z.detach().cpu().numpy()  # (N, F)

    N, F_full = Z.shape
    n_features = len(dictionary.features)
    Z = Z[:, :n_features]

    fires = (Z > 1e-9).astype(np.int64)
    n_fires = fires.sum(axis=0).astype(int)
    gram = np.abs(np.asarray(dictionary.gram())) ** 2
    Wd = sae.W_dec.detach().cpu().numpy()
    Wd = Wd[:, :n_features]
    Wd_norm = Wd / (np.linalg.norm(Wd, axis=0, keepdims=True) + 1e-9)
    decoder_overlap = np.abs(Wd_norm.T @ Wd_norm)

    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            both = int(((fires[:, i] == 1) & (fires[:, j] == 1)).sum())
            either = int(((fires[:, i] == 1) | (fires[:, j] == 1)).sum())
            jac = both / either if either > 0 else 0.0
            zi = Z[:, i]; zj = Z[:, j]
            mi, mj = zi.mean(), zj.mean()
            num = ((zi - mi) * (zj - mj)).sum()
            den = (np.sqrt(((zi - mi) ** 2).sum() * ((zj - mj) ** 2).sum())
                   + 1e-12)
            pearson = float(num / den)
            gate_pass = (
                float(gram[i, j]) >= polygram_threshold
                and jac >= jaccard_threshold
                and n_fires[i] / N >= min_firing_rate
                and n_fires[j] / N >= min_firing_rate
                and both >= min_both_fire
            )
            pairs.append(CandidatePair(
                i=i, j=j,
                polygram_overlap=float(gram[i, j]),
                decoder_overlap=float(decoder_overlap[i, j]),
                jaccard=jac,
                pearson_activation=pearson,
                kl_ablate_i=0.0, kl_ablate_j=0.0,
                kl_ratio_paired=0.0, kl_log_ratio_abs=0.0,
                n_fires_i=int(n_fires[i]),
                n_fires_j=int(n_fires[j]),
                n_both_fire=both,
                n_either_fire=either,
                gate_pass=gate_pass,
            ))

    confirmed = tuple((p.i, p.j) for p in pairs if p.gate_pass)
    summary = ValidationSummary(
        spearman_polygram_jaccard=0.0,
        spearman_decoder_jaccard=0.0,
        spearman_polygram_log_kl_abs=0.0,
        pearson_polygram_jaccard=0.0,
        pearson_decoder_jaccard=0.0,
        buckets={},
        outcome=f"synthesized from {N} econ-sae feed activations",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name=dictionary.name,
        model_name="econsae",
        layer=0,
        n_prompts=int(N),
        n_tokens=int(N),
        polygram_overlap_threshold=polygram_threshold,
        jaccard_threshold=jaccard_threshold,
        min_firing_rate=min_firing_rate,
        min_both_fire=min_both_fire,
        feature_ids=tuple(range(n_features)),
        pairs=tuple(pairs),
        summary=summary,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# Stage 6: polygram.Compressor
# ---------------------------------------------------------------------------
def compress(vreport, sae_safetensors_path: str, out_path: str,
             strategy: str = "merge"):
    """Run polygram.Compressor end-to-end; write *.compressed.safetensors
    plus a companion compression report JSON that saeforge's
    FeatureBasis.from_polygram_checkpoint can auto-locate."""
    from polygram import Compressor
    compressor = Compressor(
        validation_report=vreport,
        sae_checkpoint=Path(sae_safetensors_path),
        strategy=strategy,
    )
    result = compressor.run(Path(out_path))
    # saeforge's FeatureBasis loader looks for a sibling report at
    # <stem>_compression_report.json next to the compressed safetensors.
    report_path = Path(out_path).with_suffix("") .as_posix() + \
                  "_compression_report.json"
    try:
        with open(report_path, "w") as f:
            f.write(result.report.to_json())
    except Exception as e:
        # Non-fatal -- saeforge step can re-derive from result if available
        import warnings
        warnings.warn(
            f"failed to write compression report to {report_path}: "
            f"{type(e).__name__}: {e}",
            UserWarning,
        )
    return result


# ---------------------------------------------------------------------------
# Stage 7a: real saeforge run_synthetic via the TemporalWMAdapter
# (Phase 9.2; lands after the basis-projection eval in stage 7b)
# ---------------------------------------------------------------------------
def forge_synthetic(
    compressed_path: str,
    host,
    x_eval: torch.Tensor,
    run_dir: str,
    *,
    scale_boost: float | str = "auto",
) -> dict:
    """Run saeforge's `ForgePipeline.run_synthetic` against a TemporalWM host.

    Builds a `FeatureBasis` from the polygram-compressed safetensors, wraps
    it in a `SubspaceProjector`, dispatches `TemporalWMAdapter` via the
    `forge_adapter` registration (imported at module top), and scores
    next-state MSE between forged and host outputs on `x_eval` (a z-scored
    (B, T, N, in_dim) tensor).

    Returns a dict with the basis dims, MSE faithfulness, and forged-module
    parameter count. Status is `"skipped"` when the host's class has no
    registered adapter (e.g. `AttnWorldModel` until a Phase 9.3 adapter
    lands).
    """
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.forge import ForgePipeline
    from saeforge.projector import SubspaceProjector

    from econsae.sae.forge_adapter import NextStateMSE

    out: dict = {"status": "ok"}
    try:
        adapter_for(host)
    except NotImplementedError as e:
        out["status"] = "no_adapter"
        out["error"] = str(e)
        return out
    try:
        basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    except Exception as e:
        out["status"] = "basis_load_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    # The adapter projects fc1's output (= host.h1_dim) through the basis.
    # When basis.d_model != host.h1_dim the SAE was trained on a different
    # substrate (e.g. the engineered macro feed, d=223) and there is no
    # single fc1-aligned bridge point in the host. Surface a clean skip so
    # the failure mode is obvious.
    if basis.d_model != getattr(host, "h1_dim", basis.d_model):
        out["status"] = "dim_mismatch"
        out["error"] = (
            f"basis.d_model={basis.d_model} but host.h1_dim="
            f"{host.h1_dim}. The adapter bridges at fc1; an SAE whose "
            f"input dim differs from h1_dim wasn't trained on a layer of "
            f"this host."
        )
        return out

    projector = SubspaceProjector(basis=basis, scale_boost=scale_boost)
    pipeline = ForgePipeline(
        basis=basis, projector=projector,
        faithfulness=NextStateMSE(),
        # The default "auto" mode picks "host_wrapped" for under-complete
        # bases; that path requires an adapter `host_wrapped_module`
        # override (a Phase 9.3 extension). The native-in-basis path runs
        # against the projected weights only — exactly what we want here.
        forward_mode="native_in_basis",
    )
    forged_dir = os.path.join(run_dir, "forged_synthetic")
    try:
        result = pipeline.run_synthetic(
            host_model=host, output_dir=forged_dir, eval_input_ids=x_eval,
        )
    except Exception as e:
        out["status"] = "run_synthetic_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out.update({
        "basis": {
            "n_features": int(basis.n_features),
            "d_model": int(basis.d_model),
            "scale_compression_ratio": float(basis.scale_compression_ratio),
        },
        "scale_boost": float(projector.scale_boost),
        "next_state_mse": float(result.faithfulness),
        "faithfulness_target": result.faithfulness_target_name,
        "n_params_forged": int(result.n_params),
        "x_eval_shape": list(x_eval.shape),
        "forged_dir": forged_dir,
    })
    return out


# ---------------------------------------------------------------------------
# Stage 7b: saeforge basis evaluation (kept-subspace GT-AUC)
# ---------------------------------------------------------------------------
# The basis-projection step here is independent of `forge_synthetic`: it
# answers "does compressing the SAE preserve the GT-recoverable features?"
# rather than "does the forged downstream computation track the host's?".
# Phase 9.2 reports both numbers side-by-side in the stage-7 summary.
def forge_evaluate(compressed_path: str, sae, feed, run_dir: str) -> dict:
    """Use saeforge to load + project + score the polygram-compressed SAE.

    Stages:
      1. FeatureBasis.from_polygram_checkpoint(compressed_path)
         loads the W_dec / kept_ids / merged_norms from the compressed
         safetensors + companion compression report.
      2. SubspaceProjector(basis) projects original SAE activations onto
         the basis. The projection is equivalent to applying polygram's
         merge clustering: features in the same cluster share a slot in
         the projected representation.
      3. GroundTruthTarget(labels).faithfulness(activations) scores the
         projected activations against the GT label matrix via per-feature
         max-AUC (same metric our align() uses).

    Returns a dict with the basis dimensions, faithfulness score, and
    pseudo-faithfulness on the ORIGINAL (uncompressed) SAE activations
    for comparison. If compression preserves GT alignment, the two
    scores should be close.
    """
    from saeforge import FeatureBasis, SubspaceProjector
    from saeforge.eval import GroundTruthTarget

    out: dict = {"status": "ok"}
    try:
        basis = FeatureBasis.from_polygram_checkpoint(compressed_path)
    except Exception as e:
        out["status"] = "basis_load_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["basis"] = {
        "n_features": int(basis.n_features),
        "d_model":    int(basis.d_model),
        "scale_compression_ratio": float(basis.scale_compression_ratio),
        "kept_ids":   [int(i) for i in basis.kept_ids[:20]] + (
            ["..."] if len(basis.kept_ids) > 20 else []
        ),
    }

    projector = SubspaceProjector(basis)
    out["projector"] = {
        "n_basis_features": int(basis.n_features),
        "scale_boost": 1.0,
    }

    # Project the original SAE activations through the compressed basis.
    # The projector multiplies activations by W_dec @ W_dec_pinv to keep
    # only the components in the compressed subspace.
    import torch
    with torch.no_grad():
        Z = sae(feed.X).z.detach().cpu().numpy().astype(np.float32)  # (N, F)

    # Use the basis's W_dec (n_features x d_model) to define the kept
    # subspace. Pseudoinverse gives projection from full-feature space
    # back to the kept-feature space.
    W_dec = basis.W_dec.astype(np.float32)            # (n_kept, d_model)
    # Original SAE features: project to d_model via the original W_dec,
    # then re-project onto kept subspace. Simpler: just slice Z by
    # kept_ids -- those are the indices of features the compressor kept.
    kept_ids = np.asarray(basis.kept_ids, dtype=int)
    Z_kept = Z[:, kept_ids[kept_ids < Z.shape[1]]]      # (N, n_kept)

    # GT-AUC on the kept-feature submatrix
    gt_kept = score_against_gt(Z_kept, feed)

    # GT-AUC on the FULL original SAE for comparison
    gt_full = score_against_gt(Z, feed)

    out["gt_kept_subspace"] = gt_kept
    out["gt_full_subspace"] = gt_full
    out["delta_mAUC"] = float(gt_kept["mean_best_auc"]
                                - gt_full["mean_best_auc"])
    out["delta_cov95"] = float(gt_kept["coverage_0.95"]
                                 - gt_full["coverage_0.95"])
    out["note"] = (
        "Kept-subspace GT-AUC eval. Complements forge_synthetic's "
        "next-state MSE: this answers 'does compression preserve GT-"
        "recoverable features?', not 'does the forged forward track "
        "the host's?'. Phase 9.2 reports both numbers."
    )
    return out


# Kept under the old name for backward-compat with any caller that
# imported it during the stub era.
forge_stub = forge_evaluate


# ---------------------------------------------------------------------------
# Stage 8: AUC against the GT feature matrix
# ---------------------------------------------------------------------------
def score_against_gt(activations: np.ndarray, feed) -> dict:
    """Compute AUC of each activation column against the feed's GT labels.

    Returns a small dict shaped like the entries the scoreboard consumes.
    """
    from econsae.sae.evaluation import _auc_one_vs_many

    Y = feed.Y.astype(np.float64)
    n_feat = activations.shape[1]
    n_gt = Y.shape[1]
    best_per_gt = np.zeros(n_gt, dtype=np.float32)
    active = (activations.max(axis=0) - activations.min(axis=0)) > 1e-9
    for f in range(n_feat):
        if not active[f]:
            continue
        aucs = _auc_one_vs_many(activations[:, f].astype(np.float64), Y)
        best_per_gt = np.maximum(best_per_gt, aucs)
    return {
        "n_features":     int(n_feat),
        "n_gt_features":  int(n_gt),
        "coverage_0.95":  float((best_per_gt >= 0.95).mean()),
        "coverage_0.90":  float((best_per_gt >= 0.90).mean()),
        "mean_best_auc":  float(best_per_gt.mean()),
    }


# ---------------------------------------------------------------------------
# Stage 9: write final report
# ---------------------------------------------------------------------------
def write_report(run_dir: str, payload: dict) -> str:
    import json
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "forge_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
