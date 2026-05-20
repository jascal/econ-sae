"""Phase 9.2: saeforge ArchitectureAdapter for TemporalWorldModel.

Lets `saeforge.ForgePipeline.run_synthetic` forge an econ-sae host model
end-to-end. The SAE substrate sits at a single point in the network
(`fc1` output, a.k.a. `h1`), so the projection algebra is one-sided and
much simpler than transformer forge.

Bridge points (in PyTorch (out, in) weight layout):
  fc1: (h1_dim, gru_hidden)   ->  (n_features, gru_hidden)
        W'  =  scale_boost * E^T @ W       b' = scale_boost * E^T @ b
  fc2: (h2_dim, h1_dim)       ->  (h2_dim, n_features)
        W'  =  W @ W_dec^T / scale_boost   b' = b

Everything upstream of `fc1` (input_proj, attn, GRU) and downstream of
`fc2` (fc3) runs with the host's exact weights. Pass-through layers are
walked into the forged module verbatim.

ReLU lives between the encode (`fc1`) and decode (`fc2`) bridge points,
so the forge accepts a known approximation: ReLU is applied in basis
coordinates in the forged model versus in `h1` coordinates in the host.
That gap is exactly what the SAE training was meant to absorb, and it is
what the MSE faithfulness score measures.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saeforge.adapters import register_adapter
from saeforge.adapters.base import ArchitectureAdapter, to_numpy


FAMILY = "econsae_temporal_wm"


# ---------------------------------------------------------------------------
# Native config

@dataclass
class TemporalWMNativeConfig:
    """Duck-typed analogue of saeforge.model.NativeModelConfig.

    Carries econ-sae's TemporalWorldModel architecture dims plus the
    saeforge-orchestrator interop fields (`family`, `tied_embeddings`,
    `rope_mode`, `forward_mode`) that the synthetic forge path reads.
    A plain dataclass is sufficient: saeforge dispatches via duck-typed
    `config.family` and never validates against `NativeModelConfig`'s
    transformer-shaped invariants.
    """

    family: str = FAMILY
    n_features: int = 0
    # TemporalWorldModel architecture
    agent_dim: int = 17
    macro_dim: int = 10
    shock_dim: int = 10
    embed_dim: int = 64
    n_heads: int = 4
    n_attn_layers: int = 1
    gru_hidden: int = 128
    n_gru_layers: int = 1
    h2_dim: int = 128
    # DualHeadRegimeWM extras (0 disables the dual-head wiring)
    n_regime_labels: int = 0
    # saeforge interop fields read by the synthetic orchestrator path
    tied_embeddings: bool = False
    rope_mode: str = "standard"
    forward_mode: str = "native_in_basis"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TemporalWMNativeConfig":
        return cls(**payload)


# ---------------------------------------------------------------------------
# Forged native module

class ForgedTemporalWorldModel(nn.Module):
    """TemporalWorldModel with `h1` in basis coordinates.

    Forward pass mirrors `econsae.sae.world_model.TemporalWorldModel.forward`
    exactly except that `fc1` outputs an `n_features`-wide tensor and
    `fc2` reads one — the basis bridge. When `config.n_regime_labels > 0`
    the dual-head readouts from DualHeadRegimeWM are also constructed, so
    the same forged module covers both host classes.
    """

    def __init__(self, config: TemporalWMNativeConfig):
        super().__init__()
        self.config = config
        in_dim = config.agent_dim + config.macro_dim + config.shock_dim

        self.input_proj = nn.Linear(in_dim, config.embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
            for _ in range(config.n_attn_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(config.embed_dim) for _ in range(config.n_attn_layers)
        ])
        self.gru = nn.GRU(
            config.embed_dim, config.gru_hidden,
            num_layers=config.n_gru_layers, batch_first=True,
        )
        # Bridge point: basis-coord h1 of width n_features.
        self.fc1 = nn.Linear(config.gru_hidden, config.n_features)
        self.fc2 = nn.Linear(config.n_features, config.h2_dim)
        self.fc3 = nn.Linear(config.h2_dim, config.agent_dim)

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(config.agent_dim))
        self.register_buffer("y_std", torch.ones(config.agent_dim))

        if config.n_regime_labels > 0:
            self.regime_scale = nn.Parameter(torch.ones(config.n_regime_labels))
            self.regime_bias = nn.Parameter(torch.zeros(config.n_regime_labels))
            self.regime_head_pooled = nn.Linear(
                config.n_features, config.n_regime_labels
            )

    def forward(self, x: torch.Tensor, return_h1: bool = False):
        B, T, N, _ = x.shape
        h = x.reshape(B * T, N, -1)
        h = self.input_proj(h)
        for attn, norm in zip(self.attn_layers, self.attn_norms):
            attended, _ = attn(h, h, h, need_weights=False)
            h = norm(h + attended)
        h = h.reshape(B, T, N, -1)
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, -1)
        h, _ = self.gru(h)
        h = h.reshape(B, N, T, -1).permute(0, 2, 1, 3).contiguous()
        h1 = F.relu(self.fc1(h))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_h1:
            return out, h1
        return out


# ---------------------------------------------------------------------------
# Adapter

class TemporalWMAdapter(ArchitectureAdapter):
    """ArchitectureAdapter for econ-sae's TemporalWorldModel hosts."""

    family = FAMILY

    def walk(self, host: Any, projector, *, attention_width: str = "host"
             ) -> dict[str, np.ndarray]:
        if attention_width != "host":
            raise ValueError(
                f"TemporalWMAdapter supports attention_width='host' only; "
                f"got {attention_width!r}"
            )
        sb = float(projector.scale_boost)
        W_dec = projector.basis.W_dec               # (n_features, h1_dim)
        E = projector.basis.pseudoinverse()         # (h1_dim, n_features)

        out: dict[str, np.ndarray] = {}

        # Pass-through: input_proj
        out["input_proj.weight"] = to_numpy(host.input_proj.weight)
        out["input_proj.bias"] = to_numpy(host.input_proj.bias)

        # Pass-through: attention layers + post-attention norms
        for i, (attn, norm) in enumerate(zip(host.attn_layers, host.attn_norms)):
            out[f"attn_layers.{i}.in_proj_weight"] = to_numpy(attn.in_proj_weight)
            out[f"attn_layers.{i}.in_proj_bias"] = to_numpy(attn.in_proj_bias)
            out[f"attn_layers.{i}.out_proj.weight"] = to_numpy(attn.out_proj.weight)
            out[f"attn_layers.{i}.out_proj.bias"] = to_numpy(attn.out_proj.bias)
            out[f"attn_norms.{i}.weight"] = to_numpy(norm.weight)
            out[f"attn_norms.{i}.bias"] = to_numpy(norm.bias)

        # Pass-through: GRU (per-layer 4 tensors: weight_ih, weight_hh, bias_ih, bias_hh)
        for layer in range(host.n_gru_layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                key = f"{name}_l{layer}"
                out[f"gru.{key}"] = to_numpy(getattr(host.gru, key))

        # Projected: fc1 (encode), fc2 (decode)
        fc1_w = to_numpy(host.fc1.weight)            # (h1_dim, gru_hidden)
        fc1_b = to_numpy(host.fc1.bias)              # (h1_dim,)
        fc2_w = to_numpy(host.fc2.weight)            # (h2_dim, h1_dim)
        fc2_b = to_numpy(host.fc2.bias)              # (h2_dim,)
        out["fc1.weight"] = sb * (E.T @ fc1_w)       # (n_features, gru_hidden)
        out["fc1.bias"] = sb * (E.T @ fc1_b)         # (n_features,)
        out["fc2.weight"] = fc2_w @ W_dec.T / sb     # (h2_dim, n_features)
        out["fc2.bias"] = fc2_b.copy()

        # Pass-through: fc3
        out["fc3.weight"] = to_numpy(host.fc3.weight)
        out["fc3.bias"] = to_numpy(host.fc3.bias)

        # Normalisation buffers
        out["x_mean"] = to_numpy(host.x_mean)
        out["x_std"] = to_numpy(host.x_std)
        out["y_mean"] = to_numpy(host.y_mean)
        out["y_std"] = to_numpy(host.y_std)

        # DualHeadRegimeWM extras. Only the pooled head is projection-faithful;
        # the per-channel readout maps the last K coords of h1 in basis space,
        # which is not the same K coords as the host saw, so its forged values
        # will not match the host's. We still copy regime_scale / regime_bias
        # verbatim so the forged module is well-formed and reportable. MSE
        # faithfulness is measured on next-state output anyway, which the
        # dual-head does not feed into.
        if getattr(host, "n_regime_labels", 0) > 0:
            out["regime_scale"] = to_numpy(host.regime_scale)
            out["regime_bias"] = to_numpy(host.regime_bias)
            ph_w = to_numpy(host.regime_head_pooled.weight)  # (K, h1_dim)
            ph_b = to_numpy(host.regime_head_pooled.bias)
            out["regime_head_pooled.weight"] = ph_w @ W_dec.T / sb
            out["regime_head_pooled.bias"] = ph_b.copy()

        return out

    def build_native_config(self, host: Any, n_features: int,
                            *, attention_width: str = "host"
                            ) -> TemporalWMNativeConfig:
        if attention_width != "host":
            raise ValueError(
                f"TemporalWMAdapter supports attention_width='host' only; "
                f"got {attention_width!r}"
            )
        return TemporalWMNativeConfig(
            family=self.family,
            n_features=int(n_features),
            agent_dim=int(host.agent_dim),
            macro_dim=int(host.macro_dim),
            shock_dim=int(host.shock_dim),
            embed_dim=int(host.embed_dim),
            n_heads=int(host.n_heads),
            n_attn_layers=int(host.n_attn_layers),
            gru_hidden=int(host.gru_hidden),
            n_gru_layers=int(host.n_gru_layers),
            h2_dim=int(host.h2_dim),
            n_regime_labels=int(getattr(host, "n_regime_labels", 0)),
        )

    def native_module_class(self) -> type:
        return ForgedTemporalWorldModel

    def default_faithfulness_target(self):
        return NextStateMSE()

    # Sae-forge's base ArchitectureAdapter exposes two optional helpers that
    # only the bundled FSM-orchestrator paths invoke (host-wrapped fallback,
    # gradient checkpointing). The synthetic-imperative path used by
    # `forge_pipeline.py` stage 7 does not call either, so leaving the base's
    # NotImplementedError defaults in place is correct — they will surface a
    # clear error if a caller ever tries to use those paths against this host.


# ---------------------------------------------------------------------------
# Faithfulness target

class NextStateMSE:
    """MSE between forged and host next-state predictions.

    Reads the eval inputs from `ctx["_eval_input_ids"]` — saeforge's
    synthetic-imperative path populates that key from the `eval_input_ids`
    argument of `ForgePipeline.run_synthetic`. The expected tensor shape
    is `(B, T, N, in_dim)` already z-scored using the host's `x_mean` /
    `x_std`. Score is mean squared error across all output coordinates;
    lower is better.
    """

    name = "next_state_mse"
    better_when: Literal["lower"] = "lower"

    def score(self, *, forged: Any, host: Any, ctx: Mapping[str, Any]
              ) -> tuple[float, float]:
        try:
            x = ctx["_eval_input_ids"]
        except KeyError as e:
            raise KeyError(
                "NextStateMSE expects ctx['_eval_input_ids'] (a z-scored "
                "(B, T, N, in_dim) float tensor — pass it as run_synthetic's "
                "`eval_input_ids` argument)"
            ) from e
        forged_module = (
            forged.torch_module if hasattr(forged, "torch_module") else forged
        )
        device = next(forged_module.parameters()).device
        x = x.to(device).float()
        host = host.to(device).eval()
        forged_module = forged_module.to(device).eval()
        with torch.no_grad():
            y_host = host(x)
            y_forged = forged_module(x)
        mse = float(F.mse_loss(y_forged, y_host).item())
        # FSM progress check wants a positive-real `perplexity_analog`
        # monotonically increasing in score for `better_when == "lower"`.
        perp = math.exp(mse) if mse < 20 else float("inf")
        return mse, perp


# ---------------------------------------------------------------------------
# Registration at module import time. DualHeadRegimeWM lives under
# `scripts/` — import lazily so this module remains importable even when
# the scripts package can't load (the registry simply lacks the
# subclass entry in that case).

def _register() -> None:
    from econsae.sae.world_model import TemporalWorldModel

    adapter = TemporalWMAdapter()
    # More-specific subclass first so first-match-wins dispatch keeps it.
    try:
        from scripts.regime_dual_head_experiment import DualHeadRegimeWM
        register_adapter(DualHeadRegimeWM, adapter)
    except ImportError:
        pass
    register_adapter(TemporalWorldModel, adapter)


_register()
