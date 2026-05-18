"""Tiny predictive world model for econ-sae.

The point of the world model in this project is NOT to be the best
possible forecaster --- it is to be a small, well-trained MLP whose
hidden-layer activations form an interesting, polysemantic substrate
for SAE training. The classical mechanistic-interpretability target
is "internal activations of a neural net trained on some structured
data"; here we manufacture exactly that.

Architecture (per-agent)
------------------------
Input  : [agent_state_t (R^DIM), macro_summary (R^M), shock_vec (R^S)]
         flattened to R^(DIM + M + S)
Hidden : H1 = ReLU(W1 x + b1)            (width = h1_dim, this is the SAE substrate)
         H2 = ReLU(W2 H1 + b2)
Output : agent_state_{t+1} = W3 H2 + b3  (R^DIM)

Training is plain MSE on next-period state, with one trajectory's
periods serving as a contiguous training set. With 32 trajectories x
59 transitions x 17 agents = 31,996 (input, target) pairs, this is
tiny but enough to learn the SFC dynamics well enough that the H1
activations encode rich agent-and-period context.

The same forward pass at inference time yields H1 activations for every
(trajectory, period, agent) row in the bundle, including period 0 (no
target needed for activation extraction). Those activations become the
`acts` feed in `econsae.sae.data`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from econsae.embeddings import DIM
from econsae.sectors import GOODS_SECTORS, HH_COHORTS


# Macro-summary keys (in this order) extracted per period
MACRO_KEYS = (
    "GDP", "C_food", "C_services", "C_durables",
    "wages", "tax_revenue", "interest_paid",
    "debt_outstanding", "money_stock", "interest_rate",
)
MACRO_DIM = len(MACRO_KEYS)

# Shock-vector encoding (canonical, fixed-dim)
SHOCK_DIM = 10


def encode_shock(shock: dict, base_rate: float = 0.02) -> np.ndarray:
    """Pack a shock dict into a fixed-dim R^10 vector.

    Layout:
      [0]  productivity_mult_food          (default 1.0)
      [1]  productivity_mult_services
      [2]  productivity_mult_durables
      [3]  mpc_add_young
      [4]  mpc_add_prime
      [5]  mpc_add_retiree
      [6]  productivity_mult                (aggregate)
      [7]  transfer_per_hh                  (absolute level)
      [8]  interest_rate - base_rate        (deviation; 0 if not set this period)
      [9]  tax_rate_add
    """
    v = np.zeros(SHOCK_DIM, dtype=np.float32)
    for i, sec in enumerate(GOODS_SECTORS):
        v[i] = float(shock.get(f"productivity_mult_{sec}", 1.0)) - 1.0
    for i, c in enumerate(HH_COHORTS):
        v[3 + i] = float(shock.get(f"mpc_add_{c}", 0.0))
    v[6] = float(shock.get("productivity_mult", 1.0)) - 1.0
    v[7] = float(shock.get("transfer_per_hh", 0.0))
    if "interest_rate" in shock:
        v[8] = float(shock["interest_rate"]) - base_rate
    v[9] = float(shock.get("tax_rate_add", 0.0))
    return v


def encode_macros(macro_row: dict) -> np.ndarray:
    """Pull a fixed-size summary vector from a single-period macros dict."""
    return np.array([float(macro_row[k]) for k in MACRO_KEYS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    """Per-agent two-hidden-layer MLP. Width h1_dim is the SAE substrate."""

    def __init__(self, agent_dim: int = DIM, macro_dim: int = MACRO_DIM,
                 shock_dim: int = SHOCK_DIM, h1_dim: int = 96, h2_dim: int = 48):
        super().__init__()
        self.agent_dim = agent_dim
        self.macro_dim = macro_dim
        self.shock_dim = shock_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        in_dim = agent_dim + macro_dim + shock_dim
        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, agent_dim)
        # Pre-register normalization buffers so load_state_dict succeeds whether
        # the checkpoint was saved before or after train_world_model() ran.
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(agent_dim))
        self.register_buffer("y_std", torch.ones(agent_dim))

    def forward(self, x: torch.Tensor, return_h1: bool = False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_h1:
            return out, h1
        return out

    def h1(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.fc1(x))


# ---------------------------------------------------------------------------
# DeepWorldModel: 3-hidden-layer variant.
# The SAE substrate is the FIRST hidden layer, deliberately the widest so it
# has room to encode many compositional features that subsequent layers
# project down to the output. With (192, 128, 64) hidden dims the world
# model has ~3x the parameters of the baseline (96, 48) but the SAE input
# dimension is only 2x wider (192 vs 96).
# ---------------------------------------------------------------------------
class DeepWorldModel(nn.Module):
    def __init__(self, agent_dim: int = DIM, macro_dim: int = MACRO_DIM,
                 shock_dim: int = SHOCK_DIM,
                 hidden_dims: tuple[int, ...] = (192, 128, 64)):
        super().__init__()
        self.agent_dim = agent_dim
        self.macro_dim = macro_dim
        self.shock_dim = shock_dim
        self.hidden_dims = tuple(hidden_dims)
        in_dim = agent_dim + macro_dim + shock_dim
        dims = [in_dim] + list(self.hidden_dims) + [agent_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        # Pre-register normalization buffers (same convention as WorldModel)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(agent_dim))
        self.register_buffer("y_std", torch.ones(agent_dim))

    @property
    def h1_dim(self) -> int:
        return self.hidden_dims[0]

    def forward(self, x: torch.Tensor, return_h1: bool = False):
        h = x
        h1_out = None
        # All but the final layer use ReLU; output is linear.
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
            if i == 0:
                h1_out = h
        out = self.layers[-1](h)
        if return_h1:
            return out, h1_out
        return out

    def h1(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.layers[0](x))


# Training config (shared across all world-model variants).
@dataclass
class WMTrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"


# ---------------------------------------------------------------------------
# AttnWorldModel: input is a full period (all N_agents at once); a
# MultiheadAttention block lets each agent's prediction depend on every
# other agent's state. The per-agent h1 (after attention + linear) is still
# the SAE substrate, so downstream code that flattens (period, agent) ->
# sample row works unchanged.
#
# Architecture:
#   per-agent input (R^in_dim)
#     -> input_proj (R^embed_dim)
#     -> {self-attention over agents + LayerNorm + residual}  x n_attn_layers
#     -> fc1 + ReLU  (h1, the SAE substrate, R^h1_dim)
#     -> fc2 + ReLU  (R^h2_dim)
#     -> fc3         (per-agent next state, R^agent_dim)
#
# The shock and macro vectors are constants across agents in one period but
# concatenated per-agent so each agent's input row knows the context. Mean /
# std normalization is over the in_dim coordinates (same convention as
# WorldModel / DeepWorldModel).
# ---------------------------------------------------------------------------
class AttnWorldModel(nn.Module):
    def __init__(self, agent_dim: int = DIM, macro_dim: int = MACRO_DIM,
                 shock_dim: int = SHOCK_DIM,
                 embed_dim: int = 64, n_heads: int = 4, n_attn_layers: int = 1,
                 h1_dim: int = 192, h2_dim: int = 128):
        super().__init__()
        self.agent_dim = agent_dim
        self.macro_dim = macro_dim
        self.shock_dim = shock_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_attn_layers = n_attn_layers
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        in_dim = agent_dim + macro_dim + shock_dim

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            for _ in range(n_attn_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_attn_layers)
        ])
        self.fc1 = nn.Linear(embed_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, agent_dim)

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(agent_dim))
        self.register_buffer("y_std", torch.ones(agent_dim))

    def forward(self, x: torch.Tensor, return_h1: bool = False):
        # x: (B, N_agents, in_dim) -- inputs assumed already z-scored
        h = self.input_proj(x)                                       # (B, N, embed_dim)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attended, _ = attn(h, h, h, need_weights=False)
            h = norm(h + attended)
        h1 = F.relu(self.fc1(h))                                     # (B, N, h1_dim)
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)                                           # (B, N, agent_dim)
        if return_h1:
            return out, h1
        return out

    def h1(self, x: torch.Tensor) -> torch.Tensor:
        """Per-period h1 activations of shape (B, N_agents, h1_dim)."""
        with torch.no_grad():
            h = self.input_proj(x)
            for attn, norm in zip(self.attn_layers, self.attn_norms):
                attended, _ = attn(h, h, h, need_weights=False)
                h = norm(h + attended)
            return F.relu(self.fc1(h))


# ---------------------------------------------------------------------------
# TemporalWorldModel: AttnWorldModel + cross-period GRU.
#
# Same per-period cross-agent attention block as AttnWorldModel, plus a
# per-agent GRU layer that carries hidden state across periods. The GRU's
# hidden state is exactly the temporal-context buffer needed to encode
# regime / phase features --- e.g. "is current GDP > 5-period trailing
# mean" is a function of the last several periods' macros, all of which
# the GRU sees.
#
# Architecture (one forward pass on a trajectory batch shape (B, T, N, in_dim)):
#   per-period cross-agent attention:
#       reshape -> (B*T, N, embed)  -> MultiheadAttn  -> (B*T, N, embed)
#   per-agent cross-period GRU:
#       reshape -> (B*N, T, embed)  -> GRU            -> (B*N, T, gru_hidden)
#   per-(period, agent) MLP head:
#       fc1 + ReLU -> (B, T, N, h1_dim)   [SAE substrate]
#       fc2 + ReLU + fc3 -> (B, T, N, agent_dim)
#
# Training loss: MSE between pred[:, :-1] and states[:, 1:] (next-period
# prediction; the model can't be graded on its prediction at the final
# period because there's no future state). At inference time we keep
# every period's h1 row, including the last, as the SAE substrate.
# ---------------------------------------------------------------------------
class TemporalWorldModel(nn.Module):
    def __init__(self, agent_dim: int = DIM, macro_dim: int = MACRO_DIM,
                 shock_dim: int = SHOCK_DIM,
                 embed_dim: int = 64, n_heads: int = 4, n_attn_layers: int = 1,
                 gru_hidden: int = 128, n_gru_layers: int = 1,
                 h1_dim: int = 192, h2_dim: int = 128):
        super().__init__()
        self.agent_dim = agent_dim
        self.macro_dim = macro_dim
        self.shock_dim = shock_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_attn_layers = n_attn_layers
        self.gru_hidden = gru_hidden
        self.n_gru_layers = n_gru_layers
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        in_dim = agent_dim + macro_dim + shock_dim

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            for _ in range(n_attn_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_attn_layers)
        ])
        self.gru = nn.GRU(embed_dim, gru_hidden,
                          num_layers=n_gru_layers, batch_first=True)
        self.fc1 = nn.Linear(gru_hidden, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, agent_dim)

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(agent_dim))
        self.register_buffer("y_std", torch.ones(agent_dim))

    def forward(self, x: torch.Tensor, return_h1: bool = False):
        # x: (B, T, N, in_dim) -- already z-scored
        B, T, N, _ = x.shape
        # Cross-agent attention applied per-period
        h = x.reshape(B * T, N, -1)
        h = self.input_proj(h)                                       # (B*T, N, embed)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attended, _ = attn(h, h, h, need_weights=False)
            h = norm(h + attended)
        # h: (B*T, N, embed) -> (B, T, N, embed)
        h = h.reshape(B, T, N, -1)
        # GRU across periods per-agent: (B, N, T, embed) -> (B*N, T, embed)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, -1)
        h, _ = self.gru(h)                                           # (B*N, T, gru_hidden)
        # Reshape back: (B, N, T, gru_hidden) -> (B, T, N, gru_hidden)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
        # MLP head
        h1 = F.relu(self.fc1(h))                                     # (B, T, N, h1_dim)
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_h1:
            return out, h1
        return out

    def h1(self, x: torch.Tensor) -> torch.Tensor:
        """Per-(B, T, N) h1 activations, shape (B, T, N, h1_dim)."""
        with torch.no_grad():
            _, h1 = self.forward(x, return_h1=True)
            return h1


@dataclass
class TemporalWorldModelData:
    X: torch.Tensor                                    # (n_trajs, T, N, in_dim)
    states: torch.Tensor                               # (n_trajs, T, N, agent_dim)
    index: list[tuple[int, int]]                       # placeholder; not used here


def build_temporal_data(trajectories, shock_schedules,
                         base_rate: float = 0.02) -> TemporalWorldModelData:
    """Per-trajectory sequences of full-period agent rows.

    Each trajectory becomes one (T, N, in_dim) input tensor (sequence
    along the period axis) plus the corresponding (T, N, agent_dim)
    states tensor that supplies both the model's per-period self-input
    AND the t+1 prediction targets for training.
    """
    Xs, Ss = [], []
    for traj, sched in zip(trajectories, shock_schedules):
        states = traj.stack_states()              # (T, N, agent_dim)
        T, N, agent_dim = states.shape
        macro_vecs = np.stack([encode_macros(m) for m in traj.macros], axis=0)
        shock_vecs = np.stack(
            [encode_shock(sh, base_rate=base_rate) for sh in sched.shocks], axis=0
        )
        in_dim = agent_dim + macro_vecs.shape[1] + shock_vecs.shape[1]
        x_traj = np.zeros((T, N, in_dim), dtype=np.float32)
        # Vectorized fill: same macro/shock for every agent in a given period
        for t in range(T):
            row = np.concatenate([macro_vecs[t], shock_vecs[t]])
            x_traj[t, :, :agent_dim] = states[t]
            x_traj[t, :, agent_dim:] = row
        Xs.append(x_traj)
        Ss.append(states.astype(np.float32))
    return TemporalWorldModelData(
        X=torch.tensor(np.stack(Xs, axis=0)),
        states=torch.tensor(np.stack(Ss, axis=0)),
        index=[],
    )


def train_temporal_world_model(model: TemporalWorldModel,
                                data: TemporalWorldModelData,
                                cfg: WMTrainConfig = WMTrainConfig(),
                                verbose: bool = True) -> dict:
    """Train on next-period MSE prediction across full trajectories.

    Each minibatch is a set of complete trajectories; loss is computed
    only on periods 0..T-2 (since the final period has no future).
    Inputs / states are z-scored using stats over (traj, period, agent)
    flattened; stats stored as model buffers for reuse at inference.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    X = data.X.to(device)             # (B_trajs, T, N, in_dim)
    S = data.states.to(device)        # (B_trajs, T, N, agent_dim)

    X_flat = X.reshape(-1, X.shape[-1])
    S_flat = S.reshape(-1, S.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = S_flat.mean(dim=0); y_std = S_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach())
    model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach())
    model.y_std.copy_(y_std.detach())

    Xn = (X - x_mean) / x_std
    Sn = (S - y_mean) / y_std

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)
    history: list[float] = []
    B_total = Xn.shape[0]
    batch_size = max(1, cfg.batch_size)
    for ep in range(cfg.epochs):
        perm = torch.randperm(B_total, device=device)
        losses = []
        for i in range(0, B_total, batch_size):
            bi = perm[i:i + batch_size]
            pred = model(Xn[bi])                       # (B, T, N, agent_dim)
            loss = F.mse_loss(pred[:, :-1], Sn[bi][:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  mse={epoch_loss:.4f}")
    return {"history": history}


def extract_temporal_h1_activations(model: TemporalWorldModel,
                                     trajectories, shock_schedules,
                                     base_rate: float = 0.02
                                     ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """For every (traj, t, agent), compute the temporal model's h1.

    Trajectories are processed one at a time to keep peak memory low.
    The flat output ordering matches build_feature_matrix's sample_index
    (traj-major, then period-major, then agent-major).
    """
    model.eval()
    data = build_temporal_data(trajectories, shock_schedules, base_rate=base_rate)
    rows = []
    idx = []
    with torch.no_grad():
        x_mean = model.x_mean
        x_std = model.x_std
        for ti in range(data.X.shape[0]):
            xt = data.X[ti:ti + 1]                       # (1, T, N, in_dim)
            xt = (xt - x_mean) / x_std
            h1 = model.h1(xt).squeeze(0).cpu().numpy()   # (T, N, h1_dim)
            T, N, _ = h1.shape
            for t in range(T):
                for a in range(N):
                    rows.append(h1[t, a])
                    idx.append((ti, t, a))
    return np.stack(rows, axis=0).astype(np.float32), idx


# Per-period training data for AttnWorldModel.
@dataclass
class AttnWorldModelData:
    X: torch.Tensor                                    # (N_periods, N_agents, in_dim)
    Y: torch.Tensor                                    # (N_periods, N_agents, agent_dim)
    index: list[tuple[int, int]]                       # (traj_idx, period_t)


def build_attn_data(trajectories, shock_schedules,
                     base_rate: float = 0.02) -> AttnWorldModelData:
    """Per-(traj, t) -> per-(traj, t+1) period pairs.

    Each "sample" is a full period of agent states (not per-agent). The
    period's macro and shock vectors are broadcast across the N_agents
    rows of the input.
    """
    Xs, Ys, idx = [], [], []
    for ti, (traj, sched) in enumerate(zip(trajectories, shock_schedules)):
        states = traj.stack_states()              # (T, N, DIM)
        T, N, _ = states.shape
        macro_vecs = np.stack([encode_macros(m) for m in traj.macros], axis=0)
        shock_vecs = np.stack(
            [encode_shock(sh, base_rate=base_rate) for sh in sched.shocks], axis=0
        )
        agent_dim = states.shape[2]
        in_dim = agent_dim + macro_vecs.shape[1] + shock_vecs.shape[1]
        for t in range(T - 1):
            xp = np.zeros((N, in_dim), dtype=np.float32)
            for ai in range(N):
                xp[ai] = np.concatenate([states[t, ai], macro_vecs[t], shock_vecs[t]])
            Xs.append(xp)
            Ys.append(states[t + 1].astype(np.float32))
            idx.append((ti, t))
    return AttnWorldModelData(
        X=torch.tensor(np.stack(Xs, axis=0)),
        Y=torch.tensor(np.stack(Ys, axis=0)),
        index=idx,
    )


def train_attn_world_model(model: AttnWorldModel, data: AttnWorldModelData,
                            cfg: WMTrainConfig = WMTrainConfig(),
                            verbose: bool = True) -> dict:
    """Train AttnWorldModel via MSE on per-(period, agent) next state.

    The (period, agent, coord) target tensor is z-scored coordinate-wise
    (same way as the per-agent models); inputs are z-scored over the
    in_dim coordinates as well. Normalization stats are stored as model
    buffers so extraction at inference time replays identical scaling.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    X = data.X.to(device)           # (P, N, in_dim)
    Y = data.Y.to(device)           # (P, N, agent_dim)

    # Flatten across the agent dim for global mean/std on in_dim coords.
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1, Y.shape[-1])
    x_mean = X_flat.mean(dim=0); x_std = X_flat.std(dim=0).clamp_min(1e-3)
    y_mean = Y_flat.mean(dim=0); y_std = Y_flat.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach())
    model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach())
    model.y_std.copy_(y_std.detach())

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[float] = []
    P = Xn.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(P, device=device)
        losses = []
        for i in range(0, P, cfg.batch_size):
            bi = perm[i:i + cfg.batch_size]
            pred = model(Xn[bi])
            loss = F.mse_loss(pred, Yn[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  mse={epoch_loss:.4f}")
    return {"history": history}


def extract_attn_h1_activations(model: AttnWorldModel, trajectories,
                                 shock_schedules,
                                 base_rate: float = 0.02
                                 ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """For every (traj, t, agent), compute the AttnWorldModel's h1 row.

    Returns the same flat (N_samples, h1_dim) layout as the per-agent
    extractor so downstream feed_acts / SAE training code works unchanged.
    The sample_index ordering exactly matches build_feature_matrix's
    (sorted by trajectory, then period, then agent index).
    """
    model.eval()
    rows = []
    idx = []
    in_dim = model.agent_dim + model.macro_dim + model.shock_dim
    with torch.no_grad():
        for ti, (traj, sched) in enumerate(zip(trajectories, shock_schedules)):
            states = traj.stack_states()
            T, N, _ = states.shape
            macro_vecs = np.stack([encode_macros(m) for m in traj.macros], axis=0)
            shock_vecs = np.stack(
                [encode_shock(sh, base_rate=base_rate) for sh in sched.shocks], axis=0
            )
            for t in range(T):
                xp = np.zeros((N, in_dim), dtype=np.float32)
                for ai in range(N):
                    xp[ai] = np.concatenate([states[t, ai], macro_vecs[t], shock_vecs[t]])
                xt = torch.tensor(xp, dtype=torch.float32).unsqueeze(0)   # (1, N, in_dim)
                xt = (xt - model.x_mean) / model.x_std
                h1 = model.h1(xt).squeeze(0).cpu().numpy()                # (N, h1_dim)
                for ai in range(N):
                    rows.append(h1[ai])
                    idx.append((ti, t, ai))
    return np.stack(rows, axis=0).astype(np.float32), idx


# ---------------------------------------------------------------------------
# Training dataset construction
# ---------------------------------------------------------------------------
@dataclass
class WorldModelData:
    """Stacks of (input, target) pairs for world-model training.

    X      : (N_pairs, DIM + MACRO_DIM + SHOCK_DIM)
    Y      : (N_pairs, DIM)
    index  : list of (traj_idx, period, agent_idx) for each row
    """
    X: torch.Tensor
    Y: torch.Tensor
    index: list[tuple[int, int, int]]


def build_world_model_data(trajectories, shock_schedules,
                            base_rate: float = 0.02) -> WorldModelData:
    """Build per-(traj, t, agent) -> per-(traj, t+1, agent) training pairs."""
    Xs, Ys, idx = [], [], []
    for ti, (traj, sched) in enumerate(zip(trajectories, shock_schedules)):
        states = traj.stack_states()              # (T, N, DIM)
        T, N, _ = states.shape
        # encode per-period context vectors
        macro_vecs = np.stack([encode_macros(m) for m in traj.macros], axis=0)  # (T, MACRO_DIM)
        shock_vecs = np.stack(
            [encode_shock(sh, base_rate=base_rate) for sh in sched.shocks], axis=0
        )                                                                       # (T, SHOCK_DIM)
        for t in range(T - 1):
            for ai in range(N):
                x = np.concatenate([states[t, ai], macro_vecs[t], shock_vecs[t]])
                y = states[t + 1, ai]
                Xs.append(x.astype(np.float32))
                Ys.append(y.astype(np.float32))
                idx.append((ti, t, ai))
    return WorldModelData(
        X=torch.tensor(np.stack(Xs, axis=0)),
        Y=torch.tensor(np.stack(Ys, axis=0)),
        index=idx,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_world_model(model: WorldModel, data: WorldModelData,
                       cfg: WMTrainConfig = WMTrainConfig(),
                       verbose: bool = True) -> dict:
    """Train the world model end-to-end on MSE next-state prediction.

    Returns a dict with loss history. Inputs and targets are z-scored
    coordinate-wise before training (and the model learns in z-scored
    space); the normalization is stored on the model for re-use at
    inference time.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    X = data.X.to(device)
    Y = data.Y.to(device)

    # z-score targets and inputs separately (overwrite the pre-registered buffers)
    x_mean = X.mean(dim=0); x_std = X.std(dim=0).clamp_min(1e-3)
    y_mean = Y.mean(dim=0); y_std = Y.std(dim=0).clamp_min(1e-3)
    model.x_mean.copy_(x_mean.detach())
    model.x_std.copy_(x_std.detach())
    model.y_mean.copy_(y_mean.detach())
    model.y_std.copy_(y_std.detach())
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: list[float] = []
    N = Xn.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        losses = []
        for i in range(0, N, cfg.batch_size):
            bi = perm[i:i + cfg.batch_size]
            pred = model(Xn[bi])
            loss = F.mse_loss(pred, Yn[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"  ep {ep + 1:>3d}/{cfg.epochs}  mse={epoch_loss:.4f}")
    return {"history": history}


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
def extract_h1_activations(model: WorldModel, trajectories, shock_schedules,
                           base_rate: float = 0.02) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """For every (traj, t, agent), push the world-model input through fc1+ReLU.

    Returns:
      H1    : (N_samples, h1_dim) numpy array of activations
      index : list of (traj_idx, period, agent_idx) per row
    """
    model.eval()
    rows = []
    idx = []
    with torch.no_grad():
        for ti, (traj, sched) in enumerate(zip(trajectories, shock_schedules)):
            states = traj.stack_states()
            T, N, _ = states.shape
            macro_vecs = np.stack([encode_macros(m) for m in traj.macros], axis=0)
            shock_vecs = np.stack(
                [encode_shock(sh, base_rate=base_rate) for sh in sched.shocks], axis=0
            )
            for t in range(T):
                for ai in range(N):
                    x = np.concatenate([states[t, ai], macro_vecs[t], shock_vecs[t]])
                    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    # normalize using stored stats
                    if hasattr(model, "x_mean"):
                        x_t = (x_t - model.x_mean) / model.x_std
                    h1 = model.h1(x_t).squeeze(0).cpu().numpy()
                    rows.append(h1)
                    idx.append((ti, t, ai))
    return np.stack(rows, axis=0).astype(np.float32), idx


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from econsae.simulator.ensemble import generate_ensemble
    ens = generate_ensemble(n_trajectories=4, n_periods=20, seed=0)
    data = build_world_model_data(ens.trajectories, ens.shock_schedules)
    print(f"world-model data: X={data.X.shape}, Y={data.Y.shape}")
    model = WorldModel()
    print(f"world model: in={model.agent_dim + model.macro_dim + model.shock_dim}  "
          f"h1={model.h1_dim}  h2={model.h2_dim}  out={model.agent_dim}")
    res = train_world_model(model, data, WMTrainConfig(epochs=10), verbose=True)
    H1, idx = extract_h1_activations(model, ens.trajectories, ens.shock_schedules)
    print(f"\nextracted h1: {H1.shape}, sparsity: {(H1 == 0).mean():.2%}")
