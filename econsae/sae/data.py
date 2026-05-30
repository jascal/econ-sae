"""SAE feeds for econ-sae.

Three feeds, each carrying a common (X, Y) interface:

  X : torch.Tensor (N, D)             -- SAE training inputs
  Y : np.ndarray   (N, V) {0, 1}      -- ground-truth labels per sample
  vocab : list[str] of length V       -- ground-truth feature names

The vocabulary is shared across feeds; only the X tensor differs:

  feed_raw       -- agent state vectors directly (D = DIM = 23). Easiest:
                    sector and subsector coords are literally features.
  feed_embedded  -- agent vectors projected through a random linear map
                    into a lower dimension, forcing superposition.
  feed_acts      -- hidden activations of a trained world model
                    (D = h1_dim = 96). Closest to the real "interpret
                    an LLM" use case.

All three feeds share the SAME (traj, period, agent) sample-index ordering
returned by `econsae.ground_truth.build_feature_matrix`. That guarantees
the Y matrix from one feed builder can be re-used across feeds, and lets
downstream code compare an SAE's predictions across feeds row-for-row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from econsae.embeddings import DIM
from econsae.ground_truth import build_feature_matrix, FeatureMatrix
from econsae.simulator.ensemble import Ensemble


@dataclass
class Feed:
    name: str
    X: torch.Tensor                                   # (N, D)
    Y: np.ndarray                                     # (N, V) {0,1}
    feature_vocab: list[str]
    sample_index: list[tuple[int, int, int]]
    notes: str = ""

    @property
    def N(self) -> int: return self.X.shape[0]
    @property
    def D(self) -> int: return self.X.shape[1]


def feed_raw(ens: Ensemble, base_rate: float = 0.02,
             monetary_step: float = 0.01) -> Feed:
    """Per-(traj, period, agent) state vectors, R^23. Sanity-check feed.

    `base_rate` / `monetary_step` set the `phase:high_rate` /
    `phase:contraction` label thresholds (Phase 10); pass the calibrated
    values when the ensemble was built from a calibrated `SimConfig`.
    """
    fm: FeatureMatrix = build_feature_matrix(
        ens.trajectories, ens.shock_schedules,
        base_rate=base_rate, monetary_step=monetary_step)
    X = torch.tensor(fm.X, dtype=torch.float32)
    return Feed(
        name="raw",
        X=X,
        Y=fm.Y,
        feature_vocab=fm.feature_vocab,
        sample_index=fm.sample_index,
        notes=f"{X.shape[0]} agent-period vectors in R^{X.shape[1]}.",
    )


def feed_embedded(ens: Ensemble, embed_dim: int = 12, seed: int = 0,
                  base_rate: float = 0.02, monetary_step: float = 0.01) -> Feed:
    """Random-projection feed: simulates residual-stream superposition.

    Each sample x in R^DIM gets multiplied by a random linear map W in
    R^(DIM x embed_dim). With embed_dim < DIM the projection is lossy, and
    features that were one-hot-aligned to coords now superpose. This is the
    closest substrate to an LLM residual stream.
    """
    fm = build_feature_matrix(
        ens.trajectories, ens.shock_schedules,
        base_rate=base_rate, monetary_step=monetary_step)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((DIM, embed_dim)).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
    X_proj = (fm.X @ W).astype(np.float32)
    return Feed(
        name=f"embedded_d{embed_dim}",
        X=torch.tensor(X_proj, dtype=torch.float32),
        Y=fm.Y,
        feature_vocab=fm.feature_vocab,
        sample_index=fm.sample_index,
        notes=(f"Agent vectors projected through a random {DIM}x{embed_dim} embedding. "
               f"Forces superposition (D={embed_dim} < DIM={DIM})."),
    )


def feed_acts(ens: Ensemble, world_model, base_rate: float = 0.02,
              monetary_step: float = 0.01) -> Feed:
    """World-model hidden activations as SAE substrate.

    Pushes every (traj, period, agent) input through the trained world
    model's first hidden layer (with ReLU) and uses those activations
    as the SAE input. This is the canonical "interpret a neural net"
    setting --- the world model is the model under interpretation, and
    the SAE's job is to disentangle the latent features the world model
    encoded in its hidden state.
    """
    from econsae.sae.world_model import extract_h1_activations
    H1, idx = extract_h1_activations(world_model, ens.trajectories,
                                      ens.shock_schedules, base_rate=base_rate)
    fm = build_feature_matrix(
        ens.trajectories, ens.shock_schedules,
        base_rate=base_rate, monetary_step=monetary_step)
    # The indices should line up exactly with the GT matrix builder.
    assert idx == fm.sample_index, (
        "activation index does not match GT sample index -- ordering bug"
    )
    return Feed(
        name=f"acts_h1_d{H1.shape[1]}",
        X=torch.tensor(H1, dtype=torch.float32),
        Y=fm.Y,
        feature_vocab=fm.feature_vocab,
        sample_index=fm.sample_index,
        notes=(f"World-model h1 activations (D={H1.shape[1]}). "
               f"Mean sparsity: {(H1 == 0).mean():.1%}"),
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from econsae.simulator.ensemble import generate_ensemble
    ens = generate_ensemble(n_trajectories=4, n_periods=20, seed=0)

    fr = feed_raw(ens)
    print(f"feed_raw      : X={tuple(fr.X.shape)}  Y={fr.Y.shape}  vocab={len(fr.feature_vocab)}")
    print(f"  notes: {fr.notes}")

    fe = feed_embedded(ens, embed_dim=12)
    print(f"feed_embedded : X={tuple(fe.X.shape)}  Y={fe.Y.shape}  vocab={len(fe.feature_vocab)}")
    print(f"  notes: {fe.notes}")

    # acts feed needs a (trained) world model; skip in this smoke run
    print("feed_acts     : requires a trained world model -- see scripts/train_world_model.py")
