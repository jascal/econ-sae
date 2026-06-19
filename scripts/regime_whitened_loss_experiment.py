"""The decisive label-free allocation test: a VARIANCE-EQUALIZED SAE objective.

The granularity + allocation experiments showed regime is linearly PRESENT in a
label-free representation (probe cov95 up to 72% C3 / 100% Cref) yet no label-free
SAE recipe (width/TopK/whiten-input/l0) surfaces it — because the L2 reconstruction
objective is variance-greedy and regime is variance-cheap. The one untried route:
change the OBJECTIVE so every direction costs equally.

We reconstruct in WHITENED space: loss = ‖M·(x − x̂)‖² with M = (Σ + λI)^(−α), Σ the
z-scored data covariance. α=0 ⇒ M=I ⇒ standard L2 (the control); α=0.5 ⇒ shrinkage-
whitening (low-variance principal directions cost as much as high-variance ones).

CRUCIAL: this whitens the LOSS, not the INPUT. The encoder still sees the natural x
(input-whitening — feeding the encoder Σ^-1/2 x — was already tried and HURT, because
it amplifies noise). Here only the error metric is reweighted, so the SAE is rewarded
for allocating a latent to the (naturally present) low-variance regime direction.

  α>0 lifts regime cov95 above the α=0 control  -> the gap is a default-objective
      artifact; Reckoning #1 fully exculpated for econ (supervision incidental).
  cov95 flat across α                            -> supervision's allocation role is
      robust; it survives the most targeted label-free counter-move.

Run:  .venv/bin/python scripts/regime_whitened_loss_experiment.py [--seeds 0 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from econsae.sae.evaluation import align, feature_tier, score_sae
from econsae.sae.models import make_sae
from econsae.sae.train import TrainConfig, _lr_at
from regime_granularity_experiment import PROBE_LAM, ridge_lda_probe_auc, zscore
from regime_sae_allocation_experiment import build_feeds

REGIME = "regime"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]   # whitening strength (0 = standard L2 control)
SHRINK = 1e-2                          # λ as a fraction of the mean eigenvalue
SAE_WIDTH, SAE_L0, SAE_EPOCHS, SAE_BATCH = 512, 1e-3, 300, 256


def whitening_matrix(Xz: np.ndarray, alpha: float, shrink: float) -> np.ndarray:
    """M = (Σ + λI)^(−α) from the (already z-scored) data covariance, via eigh.
    λ = shrink · mean(eigenvalue) so near-zero (noise) directions aren't blown up."""
    if alpha == 0.0:
        return np.eye(Xz.shape[1], dtype=np.float32)
    Xc = Xz - Xz.mean(0)
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    d, V = np.linalg.eigh(cov)
    lam = shrink * float(d.mean())
    scale = np.power(np.clip(d, 0, None) + lam, -alpha)
    return (V @ np.diag(scale) @ V.T).astype(np.float32)


def train_whitened(sae, X: torch.Tensor, M: torch.Tensor, cfg: TrainConfig):
    """train() (train.py) with the recon term swapped for ‖M·(x − x̂)‖²; optimizer,
    warmup, decoder constraint, and dead-neuron resampling are identical."""
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    N = X.shape[0]
    steps_per_epoch = max(1, (N + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    step = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(N)
        for i in range(0, N, cfg.batch_size):
            x = X[perm[i:i + cfg.batch_size]]
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg, total_steps)
            z = sae.encode(x)
            x_hat = sae.decode(z)
            err = (x - x_hat) @ M.T                 # whitened residual
            recon = err.pow(2).sum(-1).mean()
            loss = recon + sae.sparsity_loss(z)
            opt.zero_grad(); loss.backward(); opt.step()
            if cfg.constrain_decoder:
                with torch.no_grad():
                    sae.W_dec.div_(sae.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9))
            sae.register_activation(z)
            if (step + 1) % cfg.resample_every == 0:
                sae.resample_dead(x, threshold=cfg.resample_threshold)
            step += 1
    return sae


def regime_recovery_whitened(X, Y, vocab, alpha, seed):
    torch.manual_seed(seed)
    Xz = zscore(X)
    M = torch.tensor(whitening_matrix(Xz, alpha, SHRINK), dtype=torch.float32)
    Xt = torch.tensor(Xz, dtype=torch.float32)
    sae = make_sae("jumprelu", Xt.shape[1], SAE_WIDTH, l0_coeff=SAE_L0, init_theta=0.05)
    train_whitened(sae, Xt, M,
                   TrainConfig(epochs=SAE_EPOCHS, batch_size=SAE_BATCH, lr=1e-3))
    rep = align(score_sae(sae, Xt), Y, vocab)
    pt = rep.per_tier[REGIME]
    return float(pt["mean_best_auc"]), float(pt["coverage_0.95"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trajs, n_periods, wm_epochs = (16, 48, 20) if args.quick else (48, 80, 40)

    feeds0 = build_feeds(n_trajs, n_periods, wm_epochs, args.seeds[0])
    ceiling = {}
    for fname, (X, Y, vocab) in feeds0.items():
        Xz = zscore(X)
        ri = [j for j, f in enumerate(vocab) if feature_tier(f) == REGIME]
        pf = [ridge_lda_probe_auc(Xz, Y[:, j], lam=PROBE_LAM, seed=args.seeds[0]) for j in ri]
        ceiling[fname] = float(np.mean([v >= 0.95 for v in pf if not np.isnan(v)]))

    out = {f: {f"a{a}": {"mauc": [], "cov95": []} for a in ALPHAS} for f in feeds0}
    for seed in args.seeds:
        feeds = feeds0 if seed == args.seeds[0] else build_feeds(n_trajs, n_periods, wm_epochs, seed)
        for fname, (X, Y, vocab) in feeds.items():
            for a in ALPHAS:
                m, c = regime_recovery_whitened(X, Y, vocab, a, seed)
                out[fname][f"a{a}"]["mauc"].append(m)
                out[fname][f"a{a}"]["cov95"].append(c)

    print(f"\n=== variance-equalized (whitened-LOSS) SAE: does it close the allocation "
          f"gap label-free? ({len(args.seeds)} seeds, n_trajs={n_trajs}) ===")
    for fname in feeds0:
        print(f"\n[{fname}]  probe cov95 ceiling = {ceiling[fname]:.1%}   (α=0 is the L2 control)")
        print(f"  {'α (whiten strength)':<22}{'mAUC':>14}{'cov95':>14}")
        for a in ALPHAS:
            m = np.array(out[fname][f"a{a}"]["mauc"]); c = np.array(out[fname][f"a{a}"]["cov95"])
            star = "  <-- beats control" if c.mean() > out[fname]["a0.0"]["cov95"][0] else ""
            print(f"  {a:<22}{m.mean():>6.3f}±{m.std():<4.2f}  "
                  f"{c.mean():>6.1%}±{c.std():<5.1%}{star}")

    p = Path(__file__).resolve().parents[1] / "runs" / "regime_whitened_loss_summary.json"
    p.write_text(json.dumps({"config": {"n_trajs": n_trajs, "n_periods": n_periods,
                                        "wm_epochs": wm_epochs, "seeds": args.seeds,
                                        "alphas": ALPHAS, "shrink": SHRINK},
                             "probe_ceiling": ceiling, "results": out}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
