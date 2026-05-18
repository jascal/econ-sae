"""Sector / cohort taxonomy for the multi-good economy.

This module is the source of truth for the discrete categorical features
of Phase 0.5: which goods sectors exist, which household cohorts exist,
their behavioral parameters, and their cross-sector preferences.

DESIGN
------
Three goods sectors {food, services, durables} produce three distinct
goods with their own inventories, prices, and production functions. Three
household cohorts {young, prime, retiree} have distinct
  - marginal propensities to consume
  - sector-preference CES weights (food vs services vs durables)
  - credit access (max debt-to-income ratio)
  - reservation wage + productivity profile

The cohort and sector taxonomies make the ground-truth feature
factorization 6-dimensional --- sector × cohort × txn_kind × shock_kind ×
balance_sheet_quartile × credit_constrained --- with cross-axis interactions
that no clean product factorization can recover. SAE / Polygram should
struggle to decompose conjunctive features like "young AND
credit-constrained AND durables-stockout".

Encodings
---------
Each cohort / sector has an integer code stored in the agent's `subsector`
coordinate. These codes namespace cleanly with the `sector` coord:
  sector = SECTOR_HH      ->  subsector in HH_COHORT_CODES.values()
  sector = SECTOR_FIRM    ->  subsector in FIRM_SECTOR_CODES.values()
  sector = SECTOR_BANK    ->  subsector = 0
  sector = SECTOR_GOV     ->  subsector = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Goods sectors ----------------------------------------------------------
GOODS_SECTORS = ("food", "services", "durables")
GOODS_INDEX = {s: i for i, s in enumerate(GOODS_SECTORS)}
N_GOODS = len(GOODS_SECTORS)

# Firm sector tags (one firm specializes in one sector)
FIRM_SECTOR_CODES = {"food": 1, "services": 2, "durables": 3}


# --- Household cohorts ------------------------------------------------------
HH_COHORTS = ("young", "prime", "retiree")
HH_COHORT_CODES = {"young": 1, "prime": 2, "retiree": 3}


@dataclass(frozen=True)
class CohortProfile:
    """Behavioral parameters for one household cohort.

    `ces_weights` are the (unnormalized) CES preference weights over
    (food, services, durables). At consumption time the household allocates
    its spending budget proportional to these weights times a (price-aware)
    factor. The weights deliberately overlap (every cohort buys food) so
    no SAE feature trivially equals "is_durables_purchase from young HH".

    `max_dti` is the household's maximum debt-to-income ratio when taking
    consumer credit. Set to 0.0 to forbid borrowing.
    """
    name: str
    mpc_mean: float
    mpc_std: float
    productivity_mean: float
    productivity_std: float
    reservation_wage: float
    tax_rate: float
    ces_weights: tuple[float, float, float]   # (food, services, durables)
    max_dti: float                            # debt-to-income ratio cap


COHORT_PROFILES: dict[str, CohortProfile] = {
    "young": CohortProfile(
        name="young",
        mpc_mean=0.92, mpc_std=0.04,
        productivity_mean=0.85, productivity_std=0.10,
        reservation_wage=0.9, tax_rate=0.18,
        # Young: heavy services (rent, transport), some durables (first home),
        #        low food share
        ces_weights=(0.25, 0.45, 0.30),
        max_dti=1.5,                          # easiest access to credit
    ),
    "prime": CohortProfile(
        name="prime",
        mpc_mean=0.78, mpc_std=0.06,
        productivity_mean=1.10, productivity_std=0.12,
        reservation_wage=1.1, tax_rate=0.25,
        # Prime-age: balanced
        ces_weights=(0.30, 0.40, 0.30),
        max_dti=0.8,
    ),
    "retiree": CohortProfile(
        name="retiree",
        mpc_mean=0.85, mpc_std=0.04,
        productivity_mean=0.40, productivity_std=0.10,
        reservation_wage=0.7, tax_rate=0.15,
        # Retirees: heavy food + services, durables low (already own them)
        ces_weights=(0.45, 0.45, 0.10),
        max_dti=0.0,
    ),
}


@dataclass(frozen=True)
class FirmSectorProfile:
    """Parameters for a firm in one goods sector."""
    name: str
    productivity_mean: float
    productivity_std: float
    markup_target: float           # firm targets price = markup * marginal_cost
    price_stickiness: float        # in [0,1]; 0 = flexible, 1 = frozen
    initial_inventory: float
    initial_price: float
    initial_wage: float
    # Sector-level demand shock sensitivity (used by shock generator)
    cyclicality: float             # how much sector demand swings with cycle


FIRM_SECTOR_PROFILES: dict[str, FirmSectorProfile] = {
    "food": FirmSectorProfile(
        name="food",
        productivity_mean=2.5, productivity_std=0.20,    # high TFP
        markup_target=1.10, price_stickiness=0.30,        # competitive, flexible
        initial_inventory=80.0, initial_price=1.00,
        initial_wage=0.9,
        cyclicality=0.3,                                  # acyclical (staple)
    ),
    "services": FirmSectorProfile(
        name="services",
        productivity_mean=1.2, productivity_std=0.15,    # labor-heavy, low TFP
        markup_target=1.25, price_stickiness=0.70,        # sticky (wage-driven)
        initial_inventory=20.0, initial_price=1.50,
        initial_wage=1.1,
        cyclicality=0.7,
    ),
    "durables": FirmSectorProfile(
        name="durables",
        productivity_mean=1.8, productivity_std=0.25,    # mid TFP, very cyclical
        markup_target=1.35, price_stickiness=0.50,
        initial_inventory=40.0, initial_price=2.00,
        initial_wage=1.2,
        cyclicality=1.3,                                  # very cyclical
    ),
}


# --- Helpers ----------------------------------------------------------------
def cohort_from_code(code: int) -> str | None:
    for name, c in HH_COHORT_CODES.items():
        if c == code:
            return name
    return None


def firm_sector_from_code(code: int) -> str | None:
    for name, c in FIRM_SECTOR_CODES.items():
        if c == code:
            return name
    return None


def ces_share(weights: tuple[float, float, float], prices: np.ndarray,
              elasticity: float = 0.8) -> np.ndarray:
    """Cobb-Douglas-like share when elasticity=1; CES at other elasticities.

    Returns the expenditure shares over (food, services, durables) for a
    consumer with the given preference weights facing the given prices.
    Uses the standard CES expenditure-share formula:
        share_i = (w_i * p_i^{1-eps}) / sum_j (w_j * p_j^{1-eps})
    With eps=1 this collapses to constant Cobb-Douglas shares = w_i.
    """
    w = np.asarray(weights, dtype=float)
    p = np.maximum(np.asarray(prices, dtype=float), 1e-9)
    val = w * np.power(p, 1.0 - elasticity)
    total = val.sum()
    if total <= 1e-12:
        return np.ones_like(val) / len(val)
    return val / total
