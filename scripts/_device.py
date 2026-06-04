"""Shared device resolution for the pipeline scripts.

`resolve_device("auto")` picks CUDA when a GPU is present, otherwise CPU.
Pass `"cpu"` or `"cuda"` to force a specific device. The `ECONSAE_DEVICE`
environment variable overrides the default when no explicit flag is given.
"""

from __future__ import annotations

import os

import torch


def resolve_device(arg: str | None = None) -> str:
    """Return a concrete torch device string ("cpu" or "cuda")."""
    choice = (arg or os.environ.get("ECONSAE_DEVICE") or "auto").lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "device=cuda requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build or use --device cpu."
        )
    return choice
