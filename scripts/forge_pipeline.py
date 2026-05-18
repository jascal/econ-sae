"""End-to-end SAE-forge benchmark pipeline (econ-sae).

Mirrors sm-sae's scripts/forge_pipeline.py with two important
adaptations:

  - econ-sae has TWO feed types (per-agent and per-period macro-feed).
    The pipeline takes an explicit `--feed-type` argument so the same
    driver works for either.
  - econ-sae's SAE checkpoints aren't named `<feed>__<variant>.pt`;
    they live under `runs/{experiment}/...pt` with experiment-specific
    naming. The pipeline takes a direct `--sae-ckpt` path.

Stages 1-6 and 8-9 are implemented now. Stage 7 (forge into a host
transformer) is a stub gated on sae-forge's pluggable-Faithfulness
release, mirroring sm-sae.

Usage:
    python scripts/forge_pipeline.py \\
        --sae-ckpt runs/regime_dual_head_experiment/jr_w512_ep300.pt \\
        --feed-type macro \\
        [--encoding rung5] [--n-amp-qubits 4] \\
        [--select-by firing_rate|gt_alignment|head] \\
        [--out runs/forge]

Feed types:
    acts    per-(period, agent) world-model h1 substrate
            (uses runs/world_model_temporal_sentiment.pt by default)
    macro   per-period macro-feed v3 substrate
            (uses runs/world_model_regime_dual_head.pt by default)

Encodings (cap = max polygram-Dictionary feature count):
    mps_rung1   8       tightest
    rung3       16
    rung4       32
    rung5       2 ** (n_amp_qubits + 3)  -- default; n_amp_qubits=4 -> 128

Selectors:
    firing_rate  features that fire most often on the feed (default)
    gt_alignment features with highest |max AUC - 0.5| against GT
    head         first cap features by id (reproducible baseline)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import torch

from econsae.sae.forge_bridge import (
    SELECTORS, build_dictionary, build_records, compress, convert_to_safetensors,
    forge_stub, load_sae, score_against_gt, synthesize_validation_report,
    write_report,
)


# ---------------------------------------------------------------------------
# Feed builders -- pick the right feed type for the SAE under test.
# ---------------------------------------------------------------------------
def _build_acts_feed():
    """Per-(period, agent) feed using the temporal+sentiment world model."""
    from econsae.sae.data import Feed
    from econsae.sae.world_model import (
        TemporalWorldModel, extract_temporal_h1_activations,
    )
    from econsae.simulator.ensemble import generate_ensemble
    from econsae.ground_truth import build_feature_matrix

    ens = generate_ensemble(n_trajectories=128, n_periods=100, seed=0,
                            sentiment_strength=0.20)
    ckpt = torch.load(
        os.path.join(REPO_ROOT, "runs", "world_model_temporal_sentiment.pt"),
        map_location="cpu", weights_only=False,
    )
    wm = TemporalWorldModel(**ckpt["config"])
    wm.load_state_dict(ckpt["state_dict"])
    wm.eval()
    H1, idx = extract_temporal_h1_activations(wm, ens.trajectories,
                                               ens.shock_schedules)
    fm = build_feature_matrix(ens.trajectories, ens.shock_schedules)
    assert idx == fm.sample_index
    return Feed(
        name="acts_phase_for_forge",
        X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y, feature_vocab=fm.feature_vocab, sample_index=fm.sample_index,
        notes=f"TemporalWorldModel h1, sentiment={0.20}, used for forge stage.",
    )


def _build_macro_feed():
    """Per-period feed v3 using the dual-head supervised world model."""
    from scripts.regime_dual_head_experiment import DualHeadRegimeWM
    from scripts.macro_feed_v3_experiment import build_macro_feed_v3
    from econsae.simulator.ensemble import generate_ensemble

    ens = generate_ensemble(n_trajectories=128, n_periods=100, seed=0,
                            sentiment_strength=0.20)
    ckpt = torch.load(
        os.path.join(REPO_ROOT, "runs", "world_model_regime_dual_head.pt"),
        map_location="cpu", weights_only=False,
    )
    wm = DualHeadRegimeWM(**ckpt["config"])
    wm.load_state_dict(ckpt["state_dict"])
    wm.eval()
    return build_macro_feed_v3(ens.trajectories, ens.shock_schedules, wm)


FEEDS = {
    "acts":  _build_acts_feed,
    "macro": _build_macro_feed,
}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sae-ckpt", required=True, dest="sae_ckpt",
                    help="path to a SAE checkpoint .pt file")
    ap.add_argument("--feed-type", required=True, choices=sorted(FEEDS.keys()),
                    dest="feed_type",
                    help="which feed the SAE was trained on")
    ap.add_argument("--encoding", default="rung5",
                    choices=["mps_rung1", "rung3", "rung4", "rung5"])
    ap.add_argument("--n-amp-qubits", type=int, default=4, dest="n_amp_qubits")
    ap.add_argument("--select-by", default="firing_rate",
                    choices=sorted(SELECTORS.keys()), dest="select_by")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "forge"))
    args = ap.parse_args()

    run_name = os.path.splitext(os.path.basename(args.sae_ckpt))[0]
    parent = os.path.basename(os.path.dirname(args.sae_ckpt))
    if parent and parent != "runs":
        run_name = f"{parent}__{run_name}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"forge_pipeline: ckpt={args.sae_ckpt}  feed-type={args.feed_type}  "
          f"out={run_dir}")
    t_start = time.time()

    # Stage 1
    print(f"  [1] load SAE  <-  {args.sae_ckpt}")
    sae, meta = load_sae(args.sae_ckpt)
    print(f"      kind={meta['kind']}  input_dim={meta['input_dim']}  "
          f"n_features={meta['n_features']}")

    # Build feed
    t0 = time.time()
    print(f"  [feed] build {args.feed_type}-type feed...")
    feed = FEEDS[args.feed_type]()
    print(f"      X={tuple(feed.X.shape)}  Y={feed.Y.shape}  "
          f"vocab={len(feed.feature_vocab)}  ({time.time() - t0:.1f}s)")

    # Stage 2
    safetensors_path = os.path.join(run_dir, "sae.safetensors")
    print(f"  [2] write safetensors  ->  {safetensors_path}")
    convert_to_safetensors(sae, safetensors_path)

    # Stage 3
    print(f"  [3] build SAEFeatureRecord dict")
    records = build_records(safetensors_path)
    print(f"      {len(records)} features loaded")

    # Stage 4
    print(f"  [4] wrap as polygram.Dictionary "
          f"(encoding={args.encoding}, select-by={args.select_by})")
    dictionary, sel_report, selection_meta = build_dictionary(
        records, args.encoding,
        selector=args.select_by, sae=sae, feed=feed,
        n_amp_qubits=args.n_amp_qubits,
    )
    print(f"      {len(dictionary.features)} features kept "
          f"(cap={selection_meta['cap_applied']}, "
          f"selection={selection_meta['method']})")

    # Stage 5
    print(f"  [5] synthesize ValidationReport from {feed.N} feed activations")
    vreport = synthesize_validation_report(sae, dictionary, feed)
    n_pairs = len(vreport.pairs)
    n_confirmed = len(vreport.confirmed)
    print(f"      {n_pairs} candidate pairs, {n_confirmed} confirmed")
    vreport_path = os.path.join(run_dir, "validation_report.json")
    try:
        vreport.to_json(vreport_path)
        print(f"      saved -> {vreport_path}")
    except Exception as e:
        print(f"      (skipped writing vreport: {type(e).__name__}: {e})")

    # Stage 6
    compressed_path = os.path.join(run_dir, "sae.compressed.safetensors")
    print(f"  [6] run Compressor  ->  {compressed_path}")
    comp_summary = {}
    try:
        comp_result = compress(vreport, safetensors_path, compressed_path)
        comp_summary = {
            "n_clusters":       int(comp_result.report.n_clusters),
            "n_features_kept":  int(comp_result.report.n_features_kept),
            "n_features_zeroed": int(comp_result.report.n_features_zeroed),
        }
        print(f"      clusters={comp_summary['n_clusters']}  "
              f"kept={comp_summary['n_features_kept']}  "
              f"zeroed={comp_summary['n_features_zeroed']}")
    except Exception as e:
        comp_summary = {"error": f"{type(e).__name__}: {e}"}
        print(f"      Compressor failed: {comp_summary['error']}")

    # Stage 7 (stub)
    print(f"  [7] forge into host model  ...")
    forge_summary = forge_stub(compressed_path, sae, feed, run_dir)
    print(f"      status={forge_summary['status']}")

    # Stage 8
    print(f"  [8] score original SAE activations against GT")
    with torch.no_grad():
        Z = sae(feed.X).z.detach().cpu().numpy().astype(np.float32)
    gt_summary = score_against_gt(Z, feed)
    print(f"      cov95={gt_summary['coverage_0.95']:.1%}  "
          f"cov90={gt_summary['coverage_0.90']:.1%}  "
          f"mAUC={gt_summary['mean_best_auc']:.3f}")

    # Stage 9
    payload = {
        "ckpt":           args.sae_ckpt,
        "feed_type":      args.feed_type,
        "encoding":       args.encoding,
        "selection":      selection_meta,
        "n_records":      len(records),
        "n_dict_features": len(dictionary.features),
        "validation_report": {
            "n_pairs":     n_pairs,
            "n_confirmed": n_confirmed,
        },
        "compression":    comp_summary,
        "forge":          forge_summary,
        "gt_alignment":   gt_summary,
        "wall_time_s":    float(time.time() - t_start),
    }
    out_path = write_report(run_dir, payload)
    print(f"  [9] wrote {out_path}")
    print(f"\nTotal wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
