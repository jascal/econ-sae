"""Shared helper for the `--calibrated` benchmark flag (Phase 10).

Resolves a calibrated-config path into the pieces the Phase 1 pipeline
(`train_all.py` / `evaluate.py`) needs to run a calibrated arm without
clobbering the baseline artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Arm:
    label: str                 # "baseline" | "calibrated"
    sim_config: object | None  # SimConfig or None
    base_rate: float           # phase:high_rate / contraction label threshold
    monetary_step: float
    suffix: str                # "" | "__calibrated" (checkpoint/summary suffix)


def resolve_arm(calibrated_path: str | None) -> Arm:
    """`None` -> the baseline arm (defaults). A path -> the calibrated arm,
    with the rate thresholds taken from the fitted `SimConfig`."""
    if not calibrated_path:
        return Arm("baseline", None, 0.02, 0.01, "")
    from econsae.calibration import SimConfig
    cfg = SimConfig.from_json(calibrated_path)
    return Arm("calibrated", cfg,
               cfg.shock.base_interest_rate, cfg.shock.monetary_step,
               "__calibrated")
