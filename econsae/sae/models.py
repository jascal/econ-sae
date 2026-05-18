"""Three SAE variants: TopK, L1, JumpReLU.

Ported from sm-sae with no structural changes -- the model code is generic
across substrates. All three share an encoder/decoder structure and a
common interface:

    sae = TopKSAE(input_dim, n_features, k=8)
    z   = sae.encode(x)            # sparse latent (B, n_features)
    x_hat = sae.decode(z)          # reconstruction (B, input_dim)
    out = sae(x)                   # SaeOutput dataclass

Each variant defines:
  - encode(x)            -> sparse z
  - sparsity_loss(z)     -> scalar added to recon loss in training

Dead-neuron handling: each SAE tracks per-feature inactivity (steps_dead)
via register_activation(z); the training loop resamples dead features
periodically.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SaeOutput:
    x_hat: torch.Tensor
    z: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0: float


# ---------------------------------------------------------------------------
class _BaseSAE(nn.Module):
    """Shared infrastructure: encoder, decoder, dead-neuron tracker."""

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features

        self.W_enc = nn.Parameter(torch.empty(n_features, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(input_dim, n_features))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9)

        self.register_buffer("steps_dead", torch.zeros(n_features, dtype=torch.long))

    def pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_dec

    def register_activation(self, z: torch.Tensor) -> None:
        with torch.no_grad():
            active = (z.abs() > 1e-9).any(dim=0)
            self.steps_dead = torch.where(
                active, torch.zeros_like(self.steps_dead), self.steps_dead + 1
            )

    def resample_dead(self, x_batch: torch.Tensor, threshold: int = 200) -> int:
        dead = (self.steps_dead >= threshold).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0
        with torch.no_grad():
            idx = torch.randint(0, x_batch.shape[0], (len(dead),), device=x_batch.device)
            seeds = x_batch[idx] - self.b_dec
            seeds = seeds / seeds.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            self.W_enc[dead] = seeds * 0.2
            self.W_dec[:, dead] = seeds.t()
            self.b_enc[dead] = 0.0
            self.steps_dead[dead] = 0
        return int(len(dead))

    # ---- subclass interface ----
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> SaeOutput:
        z = self.encode(x)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)
        sparsity = self.sparsity_loss(z)
        l0 = float((z.abs() > 1e-9).float().sum(dim=-1).mean())
        return SaeOutput(x_hat=x_hat, z=z, recon_loss=recon,
                         sparsity_loss=sparsity, l0=l0)


# ---------------------------------------------------------------------------
class TopKSAE(_BaseSAE):
    """Keep the k largest pre-activations, zero the rest. No sparsity penalty."""

    def __init__(self, input_dim: int, n_features: int, k: int = 8):
        super().__init__(input_dim, n_features)
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.pre_activation(x))
        topk_vals, topk_idx = torch.topk(a, k=min(self.k, self.n_features), dim=-1)
        z = torch.zeros_like(a)
        z.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return z

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=z.device)


# ---------------------------------------------------------------------------
class L1SAE(_BaseSAE):
    """ReLU encoder + L1 penalty on activations (scaled by decoder column norms
    so that the penalty is shrinkage-invariant -- standard SAE trick)."""

    def __init__(self, input_dim: int, n_features: int, l1_coeff: float = 1e-2):
        super().__init__(input_dim, n_features)
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pre_activation(x))

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        dec_norms = self.W_dec.norm(dim=0)
        return self.l1_coeff * (z.abs() * dec_norms).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
class JumpReLUSAE(_BaseSAE):
    """Per-feature learnable hard threshold (theta). Below theta -> 0, above -> pre_act.
    Sparsity penalty is an L0-surrogate via the active-feature count."""

    def __init__(self, input_dim: int, n_features: int,
                 l0_coeff: float = 5e-3, init_theta: float = 0.05):
        super().__init__(input_dim, n_features)
        self.log_theta = nn.Parameter(torch.full(
            (n_features,), float(torch.log(torch.tensor(init_theta)))
        ))
        self.l0_coeff = l0_coeff

    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pre_activation(x)
        gate = (a > self.theta()).float()
        return gate * a

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        active = (z.abs() > 1e-9).float()
        return self.l0_coeff * active.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
def make_sae(kind: str, input_dim: int, n_features: int, **kwargs) -> _BaseSAE:
    if kind == "topk":     return TopKSAE(input_dim, n_features, **kwargs)
    if kind == "l1":       return L1SAE(input_dim, n_features, **kwargs)
    if kind == "jumprelu": return JumpReLUSAE(input_dim, n_features, **kwargs)
    raise ValueError(f"Unknown SAE kind: {kind}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    for kind in ("topk", "l1", "jumprelu"):
        sae = make_sae(kind, input_dim=16, n_features=32, **(
            {"k": 4} if kind == "topk" else
            {"l1_coeff": 1e-2} if kind == "l1" else
            {"l0_coeff": 5e-3}
        ))
        out = sae(x)
        print(f"{kind:10s}  recon={float(out.recon_loss):.4f}  "
              f"sparsity={float(out.sparsity_loss):.4f}  L0={out.l0:.2f}")
