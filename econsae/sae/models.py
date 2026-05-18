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
class GatedSAE(_BaseSAE):
    """Gated SAE (Rajamanoharan et al. 2024) with separate gate and magnitude
    heads sharing tied weights up to a learnable rescaling.

    Architecture (per feature):
        gate_pre = W_enc (x - b_dec) + b_gate       # b_gate := b_enc
        mag_pre  = (exp(r) * W_enc) (x - b_dec) + b_mag
        gate     = Heaviside(gate_pre)              # binary {0, 1}
        mag      = ReLU(mag_pre)                    # continuous magnitude
        z        = gate * mag                       # SAE output

    The Heaviside has zero gradient almost everywhere; the *auxiliary
    loss* below trains W_enc and b_gate via differentiable ReLU on the
    same gate pre-activation:
        recon       = ||x - decode(z)||^2
        aux_recon   = ||x - decode(ReLU(gate_pre))||^2
        sparsity    = lambda * ||W_dec_columns||_2 * ||ReLU(gate_pre)||_1

    The total training loss is `recon + aux_recon + sparsity`. We pack
    `recon + aux_recon` into SaeOutput.recon_loss so the standard
    training loop's `loss = recon_loss + sparsity_loss` works unchanged.

    Hypothesis for econ-sae: the gate's explicit step-shaped activation
    should align better with threshold-defined regime labels
    (`phase:expansion := GDP[t] > 1.10 * trailing_mean`) than the smooth
    ReLU activations of L1 / JumpReLU SAEs.
    """

    def __init__(self, input_dim: int, n_features: int, l1_coeff: float = 1e-3):
        super().__init__(input_dim, n_features)
        # Magnitude branch parameters (gate branch reuses W_enc + b_enc).
        self.b_mag = nn.Parameter(torch.zeros(n_features))
        self.r_mag = nn.Parameter(torch.zeros(n_features))   # log of mag scale
        self.l1_coeff = l1_coeff

    def _gate_pre(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def _mag_pre(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.r_mag.exp().unsqueeze(-1)          # (n_features, 1)
        return F.linear(x - self.b_dec, self.W_enc * scale, self.b_mag)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        gate = (self._gate_pre(x) > 0).float()
        mag = F.relu(self._mag_pre(x))
        return gate * mag

    def sparsity_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Sparsity is computed in forward() from gate_pre directly; this
        # stub keeps the _BaseSAE interface alive but is unused.
        return torch.tensor(0.0, device=z.device)

    def forward(self, x: torch.Tensor) -> SaeOutput:
        gate_pre = self._gate_pre(x)
        mag = F.relu(self._mag_pre(x))
        gate = (gate_pre > 0).float()
        z = gate * mag
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x)

        # Auxiliary recon path: forces the gate encoder to encode
        # reconstruction-relevant info even though the gate forward is
        # binary (zero-gradient).
        aux_z = F.relu(gate_pre)
        aux_x_hat = self.decode(aux_z)
        aux_recon = F.mse_loss(aux_x_hat, x)

        # L1 sparsity on the auxiliary ReLU activations, scaled by decoder
        # column norms (shrinkage-invariant trick from the SAE literature).
        dec_norms = self.W_dec.norm(dim=0)
        sparsity = self.l1_coeff * (aux_z * dec_norms).sum(dim=-1).mean()

        l0 = float((z.abs() > 1e-9).float().sum(dim=-1).mean())
        return SaeOutput(
            x_hat=x_hat, z=z,
            recon_loss=recon + aux_recon,
            sparsity_loss=sparsity,
            l0=l0,
        )


# ---------------------------------------------------------------------------
def make_sae(kind: str, input_dim: int, n_features: int, **kwargs) -> _BaseSAE:
    if kind == "topk":     return TopKSAE(input_dim, n_features, **kwargs)
    if kind == "l1":       return L1SAE(input_dim, n_features, **kwargs)
    if kind == "jumprelu": return JumpReLUSAE(input_dim, n_features, **kwargs)
    if kind == "gated":    return GatedSAE(input_dim, n_features, **kwargs)
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
