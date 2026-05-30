"""Phase 10: calibration of simulator parameters to historical macro data.

The simulator (`econsae.simulator`) exposes ~13 shock-schedule knobs and a
handful of behavioral knobs that shape its macro dynamics. Until Phase 10
none of the shock knobs were reachable from the ensemble/build-data layer
(`generate_ensemble` hard-coded `draw_shock_schedule(n_periods, seed)`).

This package adds:
  - `SimConfig`        a serializable bundle of the calibratable parameters,
                       with defaults byte-identical to the pre-Phase-10 sim.
  - `compute_moments`  scale-invariant summary statistics of an ensemble's
                       macro series (growth rates, volatilities, ratios).
  - `moment_distance`  weighted relative-error objective vs target moments.
  - `calibrate`        derivative-free fit of `SimConfig` to a vendored set
                       of historical US macro moments (`data/macro_targets_us.json`).

Nothing here imports torch; the calibration loop only runs the simulator.
"""

from econsae.calibration.config import (
    BehavioralParams,
    ShockParams,
    SimConfig,
)
from econsae.calibration.moments import (
    MOMENT_KEYS,
    compute_moments,
    moment_distance,
)
from econsae.calibration.optimize import CalibrationResult, calibrate

__all__ = [
    "BehavioralParams",
    "ShockParams",
    "SimConfig",
    "MOMENT_KEYS",
    "compute_moments",
    "moment_distance",
    "CalibrationResult",
    "calibrate",
]
