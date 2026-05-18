"""Dense vector embeddings of economic agents (Phase 0.5).

Each agent (household, firm, government, bank) is a point in R^DIM where
the coordinates split into a CONSERVED block and a SIDE block.

  CONSERVED block -- balance-sheet stocks (money, sector inventories, debt
                     assets / liabilities) and per-period flows (labor,
                     per-sector goods in / out). Summed across all agents
                     in a closed economy each entry is either invariant
                     (money) or zero (every flow / loan: one agent's +x is
                     another's -x).

  SIDE block      -- agent characteristics: productivity, wage, own-good
                     price (firms) or average paid price (households), MPC,
                     tax rate, broad sector tag, subsector tag (cohort for
                     households, goods sector for firms), one-period
                     expectation, and credit limit. These drive dynamics
                     but are not themselves conserved.

This generalizes the Phase 0 R^15 layout to support three goods sectors
(food / services / durables; see econsae.simulator.sectors), household
cohorts (young / prime / retiree), and a bank that intermediates credit.
The expanded conserved block carries six new identities (one per goods
sector x flow direction); the expanded side block carries the cohort and
goods-sector tags that drive the categorical ground-truth feature
vocabulary used in SAE alignment scoring.

Coordinate layout (CONSERVED block, indices 0..13):
    0   money                deposit balance (signed; sum invariant)
    1   inv_food             food inventory
    2   inv_services         services inventory (nearly zero -- non-storable)
    3   inv_durables         durables inventory
    4   debt_liab            amount this agent owes
    5   debt_asset           amount others owe this agent
    6   labor_supply         hours supplied this period (flow)
    7   labor_demand         hours demanded this period (flow)
    8   goods_out_food       food produced/sold this period (flow)
    9   goods_out_services   services produced/sold (flow)
   10   goods_out_durables   durables produced/sold (flow)
   11   goods_in_food        food consumed (flow)
   12   goods_in_services    services consumed (flow)
   13   goods_in_durables    durables consumed (flow)

SIDE block (indices 14..22):
   14   productivity         output per labor hour (firm) / wage productivity (HH)
   15   wage                 wage offered (firm) / reservation wage (HH)
   16   price                own-good price (firm) / avg paid price (HH)
   17   mpc                  marginal propensity to consume
   18   tax_rate             effective tax rate
   19   sector               broad sector code (HH / firm / gov / bank / cb)
   20   subsector            cohort code (HH) or goods-sector code (firm)
   21   expectation          one-period expectation (income/demand)
   22   credit_limit         max allowed debt_liab for this agent

Identities (verified by simulator.check_conservation, each ~0):
    sum_i money[i]                      = M_initial          (closed economy)
    sum_i (debt_liab - debt_asset)[i]   = 0
    sum_i labor_supply[i]               = sum_i labor_demand[i]
    sum_i goods_in_food[i]              = sum_i goods_out_food[i]      (and svc, dur)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from econsae.sectors import (
    COHORT_PROFILES, FIRM_SECTOR_PROFILES, FIRM_SECTOR_CODES, HH_COHORT_CODES,
    GOODS_SECTORS,
)


# --- Coordinate layout ------------------------------------------------------
CONSERVED = slice(0, 14)
SIDE = slice(14, 23)
DIM = 23

COORDS = [
    # conserved block (0..13)
    "money", "inv_food", "inv_services", "inv_durables",
    "debt_liab", "debt_asset",
    "labor_supply", "labor_demand",
    "goods_out_food", "goods_out_services", "goods_out_durables",
    "goods_in_food", "goods_in_services", "goods_in_durables",
    # side block (14..22)
    "productivity", "wage", "price", "mpc", "tax_rate",
    "sector", "subsector", "expectation", "credit_limit",
]
assert len(COORDS) == DIM

# Stock coords accumulate; flow coords reset to zero between periods.
STOCK_COORDS = (
    "money", "inv_food", "inv_services", "inv_durables",
    "debt_liab", "debt_asset",
)
FLOW_COORDS = (
    "labor_supply", "labor_demand",
    "goods_out_food", "goods_out_services", "goods_out_durables",
    "goods_in_food", "goods_in_services", "goods_in_durables",
)

# Per-sector flow coord names (used by simulator + ground_truth)
INVENTORY_COORDS = {s: f"inv_{s}" for s in GOODS_SECTORS}
GOODS_OUT_COORDS = {s: f"goods_out_{s}" for s in GOODS_SECTORS}
GOODS_IN_COORDS = {s: f"goods_in_{s}" for s in GOODS_SECTORS}

# Broad sector tags
SECTOR_HH = 0
SECTOR_FIRM = 1
SECTOR_GOV = 2
SECTOR_BANK = 3
SECTOR_CB = 4
SECTOR_NAMES = {
    SECTOR_HH: "household", SECTOR_FIRM: "firm",
    SECTOR_GOV: "government", SECTOR_BANK: "bank", SECTOR_CB: "central_bank",
}

COORD_IDX = {c: i for i, c in enumerate(COORDS)}


# --- Agent ------------------------------------------------------------------
@dataclass
class Agent:
    name: str
    kind: str                    # "household" | "firm" | "government" | "bank"
    vec: np.ndarray = field(repr=False)
    # Convenience: cohort/firm-sector names for downstream code paths that
    # would otherwise have to invert the subsector code each time.
    cohort: str | None = None
    firm_sector: str | None = None

    def get(self, coord: str) -> float:
        return float(self.vec[COORD_IDX[coord]])

    def set(self, coord: str, value: float) -> None:
        self.vec[COORD_IDX[coord]] = value

    def add(self, coord: str, delta: float) -> None:
        self.vec[COORD_IDX[coord]] += delta

    def reset_flows(self) -> None:
        for c in FLOW_COORDS:
            self.vec[COORD_IDX[c]] = 0.0


# --- Builders ---------------------------------------------------------------
def _blank_vec() -> np.ndarray:
    return np.zeros(DIM, dtype=float)


COHORT_INITIAL_MONEY = {
    # Young: very low buffer; routinely taps consumer credit each period
    "young": 0.05,
    # Prime: a few periods of savings
    "prime": 3.0,
    # Retiree: moderate buffer, drawing down
    "retiree": 2.0,
}


def make_household(
    name: str,
    cohort: str,
    money: float | None = None,
    rng: np.random.Generator | None = None,
) -> Agent:
    """Build a household whose parameters are drawn from its cohort profile."""
    rng = rng or np.random.default_rng()
    p = COHORT_PROFILES[cohort]
    productivity = float(np.clip(rng.normal(p.productivity_mean, p.productivity_std), 0.1, None))
    mpc = float(np.clip(rng.normal(p.mpc_mean, p.mpc_std), 0.0, 1.0))
    if money is None:
        money = COHORT_INITIAL_MONEY[cohort]
    v = _blank_vec()
    v[COORD_IDX["money"]] = money
    v[COORD_IDX["productivity"]] = productivity
    v[COORD_IDX["wage"]] = p.reservation_wage
    v[COORD_IDX["price"]] = 1.0
    v[COORD_IDX["mpc"]] = mpc
    v[COORD_IDX["tax_rate"]] = p.tax_rate
    v[COORD_IDX["sector"]] = SECTOR_HH
    v[COORD_IDX["subsector"]] = HH_COHORT_CODES[cohort]
    v[COORD_IDX["expectation"]] = p.reservation_wage * productivity
    v[COORD_IDX["credit_limit"]] = 0.0   # set after first income realization
    return Agent(name=name, kind="household", vec=v, cohort=cohort)


def make_firm(
    name: str,
    sector: str,
    money: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Agent:
    """Build a firm specialized in one goods sector."""
    rng = rng or np.random.default_rng()
    p = FIRM_SECTOR_PROFILES[sector]
    productivity = float(np.clip(rng.normal(p.productivity_mean, p.productivity_std), 0.1, None))
    v = _blank_vec()
    v[COORD_IDX["money"]] = money
    v[COORD_IDX[f"inv_{sector}"]] = p.initial_inventory
    v[COORD_IDX["productivity"]] = productivity
    v[COORD_IDX["wage"]] = p.initial_wage
    v[COORD_IDX["price"]] = p.initial_price
    v[COORD_IDX["mpc"]] = 0.0
    v[COORD_IDX["tax_rate"]] = 0.25
    v[COORD_IDX["sector"]] = SECTOR_FIRM
    v[COORD_IDX["subsector"]] = FIRM_SECTOR_CODES[sector]
    v[COORD_IDX["expectation"]] = p.initial_inventory
    v[COORD_IDX["credit_limit"]] = max(money * 2.0, 4.0)   # firms borrow up to 2x equity, floor 4
    return Agent(name=name, kind="firm", vec=v, firm_sector=sector)


def make_government(name: str = "gov", money: float = 0.0,
                    transfer_per_hh: float = 0.0) -> Agent:
    v = _blank_vec()
    v[COORD_IDX["money"]] = money
    v[COORD_IDX["sector"]] = SECTOR_GOV
    v[COORD_IDX["wage"]] = transfer_per_hh   # transfer policy stored in wage coord
    return Agent(name=name, kind="government", vec=v)


def make_bank(name: str = "bank", money: float = 50.0,
              interest_rate: float = 0.02) -> Agent:
    """A bank intermediates credit: it holds money as deposits-from-itself,
    lends to households (consumer credit) and firms (working capital), and
    collects interest each period.

    The bank's money coord is its cash buffer (own equity + undeployed
    deposits). The total amount it has lent out lives in debt_asset (loans
    receivable). The interest rate is stored in the price coord for
    convenience.
    """
    v = _blank_vec()
    v[COORD_IDX["money"]] = money
    v[COORD_IDX["price"]] = interest_rate    # interest rate stored in price coord
    v[COORD_IDX["sector"]] = SECTOR_BANK
    return Agent(name=name, kind="bank", vec=v)


# --- AgentSet + economy builder ---------------------------------------------
@dataclass
class AgentSet:
    agents: list[Agent]

    def __len__(self) -> int: return len(self.agents)
    def __iter__(self): return iter(self.agents)
    def __getitem__(self, i): return self.agents[i]

    def by_kind(self, kind: str) -> list[Agent]:
        return [a for a in self.agents if a.kind == kind]

    def by_cohort(self, cohort: str) -> list[Agent]:
        return [a for a in self.agents if a.cohort == cohort]

    def by_firm_sector(self, sector: str) -> list[Agent]:
        return [a for a in self.agents if a.firm_sector == sector]

    def names(self) -> list[str]:
        return [a.name for a in self.agents]

    def stack(self) -> np.ndarray:
        return np.stack([a.vec for a in self.agents], axis=0)

    def reset_flows(self) -> None:
        for a in self.agents:
            a.reset_flows()


def build_economy(
    households_per_cohort: int = 4,
    firms_per_sector: int = 1,
    seed: int = 0,
    interest_rate: float = 0.02,
) -> AgentSet:
    """Construct a Phase 0.5 multi-good, multi-cohort closed economy.

    With defaults (4 HH x 3 cohorts) + (1 firm x 3 sectors) + gov + bank
    that's 12 + 3 + 1 + 1 = 17 agents.

    Initial money distribution is chosen so the total money stock is
    invariant under the rules and easy to reason about:
        12 households * 100 + 3 firms * 200 + 1 gov * 0 + 1 bank * 500
        = 1200 + 600 + 0 + 500 = 2300
    """
    rng = np.random.default_rng(seed)
    agents: list[Agent] = []

    # Households: cohorts in declared order so they get adjacent agent indices.
    for cohort in ("young", "prime", "retiree"):
        for k in range(households_per_cohort):
            agents.append(make_household(
                name=f"hh_{cohort}_{k:02d}",
                cohort=cohort,
                rng=rng,
            ))

    # Firms: one per sector by default
    for sector in ("food", "services", "durables"):
        for k in range(firms_per_sector):
            agents.append(make_firm(
                name=f"firm_{sector}_{k:02d}",
                sector=sector,
                rng=rng,
            ))

    agents.append(make_government("gov", money=0.0, transfer_per_hh=0.0))
    agents.append(make_bank("bank", money=50.0, interest_rate=interest_rate))
    return AgentSet(agents=agents)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    econ = build_economy()
    X = econ.stack()
    print(f"economy: {len(econ)} agents in R^{DIM}")
    print(f"  shape: {X.shape}")
    print(f"  total money:     {X[:, COORD_IDX['money']].sum():.2f}")
    print(f"  total inventory: food={X[:, COORD_IDX['inv_food']].sum():.0f}  "
          f"svc={X[:, COORD_IDX['inv_services']].sum():.0f}  "
          f"dur={X[:, COORD_IDX['inv_durables']].sum():.0f}")
    print(f"  households: " + ", ".join(
        f"{c}={len(econ.by_cohort(c))}" for c in ('young', 'prime', 'retiree')))
    print(f"  firms:      " + ", ".join(
        f"{s}={len(econ.by_firm_sector(s))}" for s in ('food', 'services', 'durables')))
    print(f"  COORDS ({DIM}): {COORDS}")
