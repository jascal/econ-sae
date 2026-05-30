"""Serializable bundle of the calibratable simulator parameters.

This module imports nothing from the simulator (avoids an import cycle:
`ensemble.py` does a local import of `SimConfig`). The dataclass defaults
are byte-identical to the literals in `draw_shock_schedule`
(`econsae/simulator/shocks.py`) and `Economy` (`econsae/simulator/core.py`),
so `SimConfig.default()` reproduces the pre-Phase-10 simulator exactly.

Two destinations:
  - `ShockParams`      -> kwargs of `draw_shock_schedule`
  - `BehavioralParams` -> instance attributes set on `Economy`

`base_interest_rate` doubles as the ensemble's initial bank rate (see
`generate_ensemble`), so the policy-rate floor and starting level stay
consistent under calibration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace


@dataclass(frozen=True)
class ShockParams:
    """The 13 kwargs of `econsae.simulator.shocks.draw_shock_schedule`."""
    tfp_ar: float = 0.7
    tfp_factor_vol: float = 0.04
    tfp_idio_vol: float = 0.02
    sentiment_ar: float = 0.6
    sentiment_factor_vol: float = 0.03
    sentiment_idio_vol: float = 0.015
    monetary_prob: float = 0.10
    monetary_step: float = 0.01
    base_interest_rate: float = 0.02
    fiscal_prob: float = 0.08
    fiscal_impulse: float = 0.4
    agg_tfp_prob: float = 0.02
    agg_tfp_size: float = 0.10


@dataclass(frozen=True)
class BehavioralParams:
    """`Economy` instance attributes that shape volatility / leverage."""
    sentiment_strength: float = 0.0
    repayment_rate: float = 0.10
    deposit_rate_spread: float = 0.30


# Names a flat optimizer vector may address, partitioned by destination.
_SHOCK_NAMES = tuple(f.name for f in fields(ShockParams))
_BEHAVIORAL_NAMES = tuple(f.name for f in fields(BehavioralParams))


@dataclass(frozen=True)
class SimConfig:
    shock: ShockParams = field(default_factory=ShockParams)
    behavioral: BehavioralParams = field(default_factory=BehavioralParams)

    # --- construction ------------------------------------------------------
    @classmethod
    def default(cls) -> "SimConfig":
        """The pre-Phase-10 simulator defaults (backward-compat anchor)."""
        return cls()

    def with_overrides(self, **flat: float) -> "SimConfig":
        """Return a copy with flat param names routed to shock/behavioral.

        Used by the optimizer to map a flat parameter vector onto the
        nested config. Unknown names raise -- typos shouldn't silently
        no-op during a fit.
        """
        shock_over = {k: v for k, v in flat.items() if k in _SHOCK_NAMES}
        beh_over = {k: v for k, v in flat.items() if k in _BEHAVIORAL_NAMES}
        unknown = set(flat) - set(shock_over) - set(beh_over)
        if unknown:
            raise KeyError(f"unknown calibratable params: {sorted(unknown)}")
        return SimConfig(
            shock=replace(self.shock, **shock_over),
            behavioral=replace(self.behavioral, **beh_over),
        )

    # --- views -------------------------------------------------------------
    def shock_kwargs(self) -> dict:
        """The 13-kwarg dict for `draw_shock_schedule`."""
        return asdict(self.shock)

    def behavioral_kwargs(self) -> dict:
        """The behavioral attrs to set on `Economy`."""
        return asdict(self.behavioral)

    def flat(self) -> dict:
        """All calibratable params as one flat name->value dict."""
        return {**self.shock_kwargs(), **self.behavioral_kwargs()}

    # --- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"shock": asdict(self.shock), "behavioral": asdict(self.behavioral)}

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        return cls(
            shock=ShockParams(**d.get("shock", {})),
            behavioral=BehavioralParams(**d.get("behavioral", {})),
        )

    def to_json(self, path: str, *, extra: dict | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload["_meta"] = extra
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str) -> "SimConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
