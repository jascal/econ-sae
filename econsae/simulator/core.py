"""Phase 0.5: multi-good, multi-cohort, credit-enabled closed economy.

Eight transaction kinds, three goods sectors, three household cohorts, one
bank intermediating credit, sticky prices, stochastic shock channel. All
exactly stock-flow consistent: every transaction is double-entry, every
conserved coord (money, debt, per-sector goods flows, labor) has a
matching aggregate identity, and `check_conservation` returns residuals at
machine precision.

STEP STRUCTURE (one period)
---------------------------
   0. reset flow coords (per-sector flows + labor; stocks persist)
   1. apply optional shock
   2. INTEREST: each debtor pays r * debt_liab to the bank
   3. PRODUCTION: firms produce in their own sector based on expectation
   4. FIRM CREDIT: firms short on wage bill borrow from bank
   5. WAGES: firms hire hours across HHs, paying w * hours each
   6. TAXES: HH wage income taxed at cohort tax_rate
   7. TRANSFERS: gov distributes its balance per-HH (set by shock policy)
   8. HH CREDIT: liquidity-constrained HHs draw on consumer credit
   9. CONSUMPTION: each HH allocates mpc * disposable_income across
                   sectors via CES expenditure shares (cohort-specific
                   weights, current per-sector prices). Sales clear from
                   the relevant firm's inventory.
  10. REPAYMENT: agents with surplus cash repay a slice of outstanding debt
  11. STICKY PRICES: firms adjust own-good price toward markup * MC
  12. EXPECTATIONS + CREDIT LIMITS update for next period

ANALOGY TO sm-sae (extended)
----------------------------
  vertex catalog        ->  TXN_KINDS (8 kinds)
  conserved charge      ->  per-coord aggregate identity (12 of them, listed
                           in `check_conservation`)
  decay cascade         ->  shock propagation through the txn graph
  generation tag        ->  household cohort
  color triplet         ->  goods sector triplet (food/services/durables)
  flavor                ->  txn sector tag
  the SM is factorial   ->  econ-sae is partially-factorial-with-overlap:
                           conjunctive features (young AND credit-constrained
                           AND services-stockout) have no clean product
                           decomposition, by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from econsae.embeddings import (
    AgentSet, Agent, COORDS, COORD_IDX, DIM, build_economy,
    GOODS_OUT_COORDS, GOODS_IN_COORDS, INVENTORY_COORDS,
    SECTOR_HH, SECTOR_FIRM, SECTOR_GOV, SECTOR_BANK,
)
from econsae.sectors import (
    GOODS_SECTORS, COHORT_PROFILES, FIRM_SECTOR_PROFILES, ces_share,
)


# ---------------------------------------------------------------------------
# Transaction kinds -- the "vertices" of the economy. Each carries a clean
# semantic tag that becomes a ground-truth feature in SAE alignment.
# ---------------------------------------------------------------------------
TXN_KINDS = (
    "wage",
    "purchase",          # subkind = sector in {food, services, durables}
    "tax",
    "transfer",
    "loan_origination",
    "loan_repayment",
    "interest",          # debtor -> bank
    "deposit_interest",  # bank -> depositor (closes the bank-profit leak)
    "dividend",          # firm -> HH (closes the firm-profit leak)
)


@dataclass
class Transaction:
    """A single double-entry vertex.

    For every kind, money flows sender->receiver. Additional conserved-coord
    flows depend on kind:

        wage             : labor_demand (firm), labor_supply (HH); side_amount = hours
        purchase         : per-sector goods_out (firm) + goods_in (HH); inv_<sector> -= goods
                           sector tag carried in `sector`
        loan_origination : sender.debt_asset += L, receiver.debt_liab += L   (bank -> debtor)
        loan_repayment   : sender.debt_liab -= P, receiver.debt_asset -= P  (debtor -> bank)
        interest         : pure money flow (no stock change in debt)
        tax / transfer   : pure money flow
    """
    kind: str
    sender: str
    receiver: str
    amount: float                            # money amount
    sector: str | None = None                # for "purchase"
    side_amount: float = 0.0                 # hours / goods units
    period: int = -1

    def apply(self, agents: dict[str, Agent]) -> None:
        s = agents[self.sender]
        r = agents[self.receiver]

        # Money always flows sender -> receiver
        s.add("money", -self.amount)
        r.add("money", +self.amount)

        if self.kind == "wage":
            # Firm is the sender of money (employer), HH the receiver.
            s.add("labor_demand", +self.side_amount)
            r.add("labor_supply", +self.side_amount)

        elif self.kind == "purchase":
            sec = self.sector
            assert sec is not None, "purchase txn needs a sector"
            # HH = money-sender (buyer), firm = money-receiver (seller)
            s.add(GOODS_IN_COORDS[sec], +self.side_amount)
            r.add(GOODS_OUT_COORDS[sec], +self.side_amount)
            r.add(INVENTORY_COORDS[sec], -self.side_amount)

        elif self.kind == "loan_origination":
            # bank = money-sender (loan disbursed), debtor = money-receiver
            s.add("debt_asset", +self.amount)
            r.add("debt_liab", +self.amount)

        elif self.kind == "loan_repayment":
            # debtor = money-sender, bank = money-receiver
            s.add("debt_liab", -self.amount)
            r.add("debt_asset", -self.amount)

        elif self.kind in ("interest", "deposit_interest", "tax", "transfer", "dividend"):
            pass  # pure money flow

        else:
            raise ValueError(f"unknown txn kind: {self.kind}")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    n_agents: int
    dim: int
    states: list[np.ndarray] = field(default_factory=list)        # each (n_agents, DIM)
    txn_log: list[list[Transaction]] = field(default_factory=list)
    macros: list[dict[str, float]] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    agent_kinds: list[str] = field(default_factory=list)
    agent_cohorts: list[str | None] = field(default_factory=list)
    agent_firm_sectors: list[str | None] = field(default_factory=list)
    shock_schedule: list[dict] = field(default_factory=list)

    @property
    def T(self) -> int: return len(self.states)

    def stack_states(self) -> np.ndarray:
        return np.stack(self.states, axis=0)


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------
@dataclass
class Economy:
    agent_set: AgentSet
    hours_per_hh: float = 1.0
    period: int = 0
    repayment_rate: float = 0.10      # fraction of debt repaid each period (when above cash buffer)
    cash_buffer_frac: float = 0.5     # repay only when money >= cash_buffer_frac * expectation
    fiscal_rule: str = "balanced"     # "balanced" | "fixed"; balanced = redistribute last period's tax
    deposit_rate_spread: float = 0.3  # bank keeps this fraction of policy rate as net interest margin
    wealth_target_ratio: float = 3.0  # HH target cash buffer = ratio * expected income
    buffer_drawdown: float = 0.25     # HH spends this fraction of excess cash above buffer target
    firm_cash_buffer_ratio: float = 2.0  # firms target money = ratio * wage_bill; pay rest as dividend
    dividend_rate: float = 0.30       # fraction of firm's excess cash paid out each period
    # --- Sentiment-driven MPC ----------------------------------------------
    # When sentiment_strength > 0, households scale their permanent-income
    # consumption target by (1 +/- strength) when the trailing GDP regime
    # is expansion / contraction. The regime is computed from a history
    # buffer (`_gdp_history`) that lives OUTSIDE agent state, so the world
    # model can only encode it by tracking past macros itself. This makes
    # regime features informative for next-state prediction.
    sentiment_window: int = 5
    sentiment_expansion_thr: float = 1.10
    sentiment_contraction_thr: float = 0.90
    sentiment_strength: float = 0.0    # 0 = off (default keeps backward compat)
    _last_tax_revenue: float = 0.0
    _gdp_history: list[float] = field(default_factory=list)

    @classmethod
    def small(cls, households_per_cohort: int = 4, firms_per_sector: int = 1,
              seed: int = 0, interest_rate: float = 0.02) -> "Economy":
        return cls(agent_set=build_economy(
            households_per_cohort=households_per_cohort,
            firms_per_sector=firms_per_sector,
            seed=seed, interest_rate=interest_rate,
        ))

    # --- accessors --------------------------------------------------------
    def households(self) -> list[Agent]:  return self.agent_set.by_kind("household")
    def firms(self)      -> list[Agent]:  return self.agent_set.by_kind("firm")

    def government(self) -> Agent:
        govs = self.agent_set.by_kind("government")
        assert len(govs) == 1, "expected exactly one government agent"
        return govs[0]

    def bank(self) -> Agent:
        banks = self.agent_set.by_kind("bank")
        assert len(banks) == 1, "expected exactly one bank"
        return banks[0]

    def _agent_index(self) -> dict[str, Agent]:
        return {a.name: a for a in self.agent_set.agents}

    def interest_rate(self) -> float:
        return float(self.bank().get("price"))

    # ---------------------------------------------------------------------
    # ONE STEP
    # ---------------------------------------------------------------------
    def step(self, shock: dict | None = None) -> tuple[np.ndarray, list[Transaction], dict]:
        ag = self._agent_index()
        self.agent_set.reset_flows()

        # Default fiscal rule: redistribute last period's tax revenue as
        # equal per-HH transfers. A `transfer_per_hh` entry in `shock`
        # overrides this. The rule keeps the closed economy from leaking
        # money into a passive government account.
        if self.fiscal_rule == "balanced" and len(self.households()) > 0:
            per_hh = max(self._last_tax_revenue / len(self.households()), 0.0)
            self.government().set("wage", per_hh)

        if shock:
            self._apply_shock(shock)

        # Sentiment multiplier from past GDPs (NOT stored in any agent
        # coord). Set to 1.0 if the buffer is too short OR the simulator is
        # configured with sentiment_strength=0 (the default).
        sentiment_mult = 1.0
        if (self.sentiment_strength > 0.0
                and len(self._gdp_history) >= self.sentiment_window + 1):
            last_gdp = self._gdp_history[-1]
            trailing = self._gdp_history[-(self.sentiment_window + 1):-1]
            trailing_mean = sum(trailing) / max(len(trailing), 1)
            if trailing_mean > 1e-9:
                if last_gdp > self.sentiment_expansion_thr * trailing_mean:
                    sentiment_mult = 1.0 + self.sentiment_strength
                elif last_gdp < self.sentiment_contraction_thr * trailing_mean:
                    sentiment_mult = 1.0 - self.sentiment_strength

        txns: list[Transaction] = []
        hhs = self.households()
        firms = self.firms()
        gov = self.government()
        bank = self.bank()
        r = self.interest_rate()

        # ---- 2a. Deposit interest: bank pays depositors at (1 - spread) * r.
        #          Paid at TOP of period so it becomes part of current
        #          disposable income, gets taxed, and recycles into demand.
        depositor_rate = max(0.0, (1.0 - self.deposit_rate_spread) * r)
        deposit_yield_received = {hh.name: 0.0 for hh in hhs}
        if depositor_rate > 1e-12:
            for hh in hhs:
                bal = max(hh.get("money"), 0.0)
                yield_amt = depositor_rate * bal
                yield_amt = min(yield_amt, max(bank.get("money"), 0.0))
                if yield_amt <= 1e-12:
                    continue
                txns.append(Transaction(
                    kind="deposit_interest", sender=bank.name, receiver=hh.name,
                    amount=yield_amt, period=self.period,
                ))
                txns[-1].apply(ag)
                deposit_yield_received[hh.name] += yield_amt

        # ---- 2b. Interest on outstanding debt ----
        for a in hhs + firms:
            d = a.get("debt_liab")
            if d <= 1e-12:
                continue
            interest_due = r * d
            if interest_due <= 1e-12:
                continue
            interest_due = min(interest_due, max(a.get("money"), 0.0))
            if interest_due <= 1e-12:
                continue
            txns.append(Transaction(
                kind="interest", sender=a.name, receiver=bank.name,
                amount=interest_due, period=self.period,
            ))
            txns[-1].apply(ag)

        # ---- 3. Production (per-sector) ----
        for f in firms:
            sec = f.firm_sector
            target = max(f.get("expectation"), 0.0)
            hours_needed = target / max(f.get("productivity"), 1e-9)
            # Hours actually committed; cap by available labor pool implicitly via wages step
            f.add(INVENTORY_COORDS[sec], hours_needed * f.get("productivity"))

        # ---- 4. Firm credit: top up cash if wage bill exceeds cash ----
        firm_hours_needed = {
            f.name: max(f.get("expectation"), 0.0) / max(f.get("productivity"), 1e-9)
            for f in firms
        }
        for f in firms:
            wage_bill = f.get("wage") * firm_hours_needed[f.name]
            shortfall = wage_bill - f.get("money")
            if shortfall <= 1e-9:
                continue
            available_credit = max(f.get("credit_limit") - f.get("debt_liab"), 0.0)
            available_credit = min(available_credit, max(bank.get("money"), 0.0))
            loan = max(0.0, min(shortfall, available_credit))
            if loan <= 1e-9:
                continue
            txns.append(Transaction(
                kind="loan_origination", sender=bank.name, receiver=f.name,
                amount=loan, period=self.period,
            ))
            txns[-1].apply(ag)

        # ---- 5. Wages: each firm hires hours_needed across HHs ----
        # Round-robin: cycle through HHs giving each firm whatever fraction
        # of hours_per_hh it can pay for. Hours offered are capped at the
        # remaining wage budget of the firm.
        hh_hours_remaining = {hh.name: self.hours_per_hh for hh in hhs}
        hh_order = [hh.name for hh in hhs]
        cursor = 0
        for f in firms:
            wage = f.get("wage")
            hours_remaining_firm = firm_hours_needed[f.name]
            spins = 0
            while hours_remaining_firm > 1e-9 and spins < 4 * len(hh_order):
                hh_name = hh_order[cursor % len(hh_order)]
                cursor += 1
                spins += 1
                hh_hours = hh_hours_remaining[hh_name]
                if hh_hours <= 1e-9:
                    continue
                take = min(hh_hours, hours_remaining_firm)
                pay = wage * take
                if f.get("money") < pay:
                    take = max(0.0, f.get("money") / max(wage, 1e-9))
                    pay = wage * take
                if take <= 1e-12:
                    continue
                txns.append(Transaction(
                    kind="wage", sender=f.name, receiver=hh_name,
                    amount=pay, side_amount=take, period=self.period,
                ))
                txns[-1].apply(ag)
                hh_hours_remaining[hh_name] -= take
                hours_remaining_firm -= take

        # ---- 6. Taxes: HH wage + deposit-yield income taxed at HH tax_rate ----
        wage_income = {hh.name: 0.0 for hh in hhs}
        for t in txns:
            if t.kind == "wage":
                wage_income[t.receiver] += t.amount
        for hh in hhs:
            taxable = wage_income[hh.name] + deposit_yield_received[hh.name]
            tax = hh.get("tax_rate") * taxable
            tax = min(tax, max(hh.get("money"), 0.0))
            if tax <= 1e-12:
                continue
            txns.append(Transaction(
                kind="tax", sender=hh.name, receiver=gov.name,
                amount=tax, period=self.period,
            ))
            txns[-1].apply(ag)

        # ---- 7. Transfers ----
        per_hh = gov.get("wage")  # transfer policy
        transfers_received = {hh.name: 0.0 for hh in hhs}
        if per_hh > 1e-12 and len(hhs) > 0:
            for hh in hhs:
                amt = min(per_hh, max(gov.get("money"), 0.0))
                if amt <= 1e-12:
                    break
                txns.append(Transaction(
                    kind="transfer", sender=gov.name, receiver=hh.name,
                    amount=amt, period=self.period,
                ))
                txns[-1].apply(ag)
                transfers_received[hh.name] += amt

        # ---- 8. HH credit: permanent-income consumption + liquidity bridging ----
        # disposable income this period (post-tax wage + post-tax deposit yield + transfers)
        disposable = {
            hh.name: (wage_income[hh.name] + deposit_yield_received[hh.name])
                     * (1.0 - hh.get("tax_rate"))
                     + transfers_received[hh.name]
            for hh in hhs
        }
        # Permanent-income target: spend mpc * max(disposable, expectation).
        # This causes HHs to smooth consumption when current income is below
        # expected income (e.g., young HHs early on) and to borrow rather than
        # contract spending.
        # Permanent-income MPC component + buffer-stock drawdown.
        # The latter ensures HHs that accumulate cash above their target
        # buffer (wealth_target_ratio * expected income) draw down a slice
        # of the excess; without it the closed economy ratchets toward a
        # low-output equilibrium as HHs save 1-mpc indefinitely.
        desired_spend = {}
        for hh in hhs:
            pi = hh.get("mpc") * max(disposable[hh.name], hh.get("expectation"))
            # Sentiment scales the forward-looking part of consumption but
            # NOT the buffer-stock drawdown; intuitively, regime perception
            # affects how much HHs spend out of income, not how aggressively
            # they draw down precautionary cash.
            pi *= sentiment_mult
            target_cash = self.wealth_target_ratio * max(hh.get("expectation"), 0.0)
            excess_cash = max(hh.get("money") - target_cash, 0.0)
            desired_spend[hh.name] = pi + self.buffer_drawdown * excess_cash
        for hh in hhs:
            want = desired_spend[hh.name]
            cash_short = want - max(hh.get("money"), 0.0)
            if cash_short <= 1e-9:
                continue
            avail = max(hh.get("credit_limit") - hh.get("debt_liab"), 0.0)
            avail = min(avail, max(bank.get("money"), 0.0))
            loan = max(0.0, min(cash_short, avail))
            if loan <= 1e-9:
                continue
            txns.append(Transaction(
                kind="loan_origination", sender=bank.name, receiver=hh.name,
                amount=loan, period=self.period,
            ))
            txns[-1].apply(ag)

        # ---- 9. Consumption: CES allocation across sectors ----
        # Build current per-sector prices from each firm's price.
        firms_by_sector = {sec: self.agent_set.by_firm_sector(sec) for sec in GOODS_SECTORS}
        sector_prices = np.array([
            float(firms_by_sector[sec][0].get("price")) if firms_by_sector[sec] else 1.0
            for sec in GOODS_SECTORS
        ])
        for hh in hhs:
            cohort = hh.cohort
            assert cohort is not None
            profile = COHORT_PROFILES[cohort]
            shares = ces_share(profile.ces_weights, sector_prices, elasticity=0.8)
            budget = min(desired_spend[hh.name], max(hh.get("money"), 0.0))
            if budget <= 1e-12:
                continue
            avg_price_num, avg_price_denom = 0.0, 0.0
            for si, sec in enumerate(GOODS_SECTORS):
                allotment = budget * shares[si]
                if allotment <= 1e-12:
                    continue
                firm_list = firms_by_sector[sec]
                if not firm_list:
                    continue
                # For simplicity, route all sector demand to the first firm in that sector.
                firm = firm_list[0]
                p = max(firm.get("price"), 1e-9)
                inv = max(firm.get(INVENTORY_COORDS[sec]), 0.0)
                goods = min(allotment / p, inv)
                amt = goods * p
                if amt <= 1e-12:
                    continue
                txns.append(Transaction(
                    kind="purchase", sender=hh.name, receiver=firm.name,
                    amount=amt, sector=sec, side_amount=goods, period=self.period,
                ))
                txns[-1].apply(ag)
                avg_price_num += amt
                avg_price_denom += goods
            if avg_price_denom > 1e-9:
                hh.set("price", avg_price_num / avg_price_denom)

        # ---- 10. Repayment: voluntary slice when above cash buffer ----
        for a in hhs + firms:
            debt = a.get("debt_liab")
            if debt <= 1e-9:
                continue
            buf = self.cash_buffer_frac * max(a.get("expectation"), 1e-9)
            surplus = a.get("money") - buf
            if surplus <= 1e-9:
                continue
            repay = min(surplus, self.repayment_rate * debt, debt)
            if repay <= 1e-9:
                continue
            txns.append(Transaction(
                kind="loan_repayment", sender=a.name, receiver=bank.name,
                amount=repay, period=self.period,
            ))
            txns[-1].apply(ag)

        # ---- 10b. Dividends: firms with cash above buffer pay out to HHs ----
        # Closes the firm-profit leak symmetrically with deposit_interest
        # closing the bank-profit leak. Without it, accumulated firm cash
        # would slowly starve household demand. Equal-share dividends to all
        # HHs (a "universal equity" simplification).
        n_hh = max(len(hhs), 1)
        for f in firms:
            wage_bill_est = f.get("wage") * firm_hours_needed[f.name]
            cash_target = self.firm_cash_buffer_ratio * wage_bill_est
            excess = max(f.get("money") - cash_target, 0.0)
            payout_total = self.dividend_rate * excess
            if payout_total <= 1e-9:
                continue
            per_hh_div = payout_total / n_hh
            for hh in hhs:
                if per_hh_div <= 1e-12:
                    continue
                txns.append(Transaction(
                    kind="dividend", sender=f.name, receiver=hh.name,
                    amount=per_hh_div, period=self.period,
                ))
                txns[-1].apply(ag)

        # ---- 11. Sticky prices: adjust toward markup * unit labor cost ----
        for f in firms:
            sec = f.firm_sector
            prof = FIRM_SECTOR_PROFILES[sec]
            mc = f.get("wage") / max(f.get("productivity"), 1e-9)
            p_target = prof.markup_target * mc
            stick = prof.price_stickiness
            new_price = (1.0 - stick) * p_target + stick * f.get("price")
            f.set("price", max(new_price, 1e-6))

        # ---- 12. Expectations + credit-limit update ----
        alpha = 0.5
        for f in firms:
            sec_sold = f.get(GOODS_OUT_COORDS[f.firm_sector])
            f.set("expectation", (1 - alpha) * f.get("expectation") + alpha * sec_sold)
            # firm credit limit = 2x equity (money + receivables)
            f.set("credit_limit", 2.0 * max(f.get("money") + f.get("debt_asset"), 0.0))
        for hh in hhs:
            cohort = hh.cohort
            prof = COHORT_PROFILES[cohort]
            inc = ((wage_income[hh.name] + deposit_yield_received[hh.name])
                   * (1.0 - hh.get("tax_rate"))
                   + transfers_received[hh.name])
            hh.set("expectation", (1 - alpha) * hh.get("expectation") + alpha * inc)
            hh.set("credit_limit", prof.max_dti * max(hh.get("expectation"), 0.0))

        # ---- snapshot + macros ----
        snapshot = self.agent_set.stack().copy()
        macros = self._macros(txns)
        self._last_tax_revenue = float(macros["tax_revenue"])
        # Track GDP for the sentiment buffer (not in agent state).
        self._gdp_history.append(float(macros["GDP"]))
        self.period += 1
        return snapshot, txns, macros

    # --- rollout ---------------------------------------------------------
    def rollout(self, n_periods: int,
                shocks: Iterable[dict | None] | None = None) -> Trajectory:
        shocks = list(shocks) if shocks is not None else [None] * n_periods
        assert len(shocks) == n_periods, "shocks length must equal n_periods"
        traj = Trajectory(
            n_agents=len(self.agent_set), dim=DIM,
            agent_names=self.agent_set.names(),
            agent_kinds=[a.kind for a in self.agent_set.agents],
            agent_cohorts=[a.cohort for a in self.agent_set.agents],
            agent_firm_sectors=[a.firm_sector for a in self.agent_set.agents],
        )
        for t, sh in enumerate(shocks):
            state, txns, macros = self.step(shock=sh)
            traj.states.append(state)
            traj.txn_log.append(txns)
            traj.macros.append(macros)
            traj.shock_schedule.append(sh or {})
        return traj

    # ---------------------------------------------------------------------
    def _apply_shock(self, shock: dict) -> None:
        """Recognized shock keys:
            productivity_mult        : multiply every firm's productivity
            productivity_mult_<sec>  : multiply only sector <sec>'s firms
            mpc_add                  : add to every HH's mpc (sentiment)
            mpc_add_<cohort>         : add only to that cohort's mpc
            transfer_per_hh          : set government's per-HH transfer
            tax_rate_add             : add to every HH tax_rate
            interest_rate            : set bank's policy rate
            interest_rate_add        : add to current policy rate
        """
        if "productivity_mult" in shock:
            m = float(shock["productivity_mult"])
            for f in self.firms():
                f.set("productivity", f.get("productivity") * m)
        for sec in GOODS_SECTORS:
            key = f"productivity_mult_{sec}"
            if key in shock:
                m = float(shock[key])
                for f in self.agent_set.by_firm_sector(sec):
                    f.set("productivity", f.get("productivity") * m)
        if "mpc_add" in shock:
            d = float(shock["mpc_add"])
            for hh in self.households():
                hh.set("mpc", float(np.clip(hh.get("mpc") + d, 0.0, 1.0)))
        for cohort in ("young", "prime", "retiree"):
            key = f"mpc_add_{cohort}"
            if key in shock:
                d = float(shock[key])
                for hh in self.agent_set.by_cohort(cohort):
                    hh.set("mpc", float(np.clip(hh.get("mpc") + d, 0.0, 1.0)))
        if "transfer_per_hh" in shock:
            self.government().set("wage", float(shock["transfer_per_hh"]))
        if "tax_rate_add" in shock:
            d = float(shock["tax_rate_add"])
            for hh in self.households():
                hh.set("tax_rate", float(np.clip(hh.get("tax_rate") + d, 0.0, 1.0)))
        if "interest_rate" in shock:
            self.bank().set("price", float(shock["interest_rate"]))
        if "interest_rate_add" in shock:
            d = float(shock["interest_rate_add"])
            self.bank().set("price",
                            float(np.clip(self.bank().get("price") + d, 0.0, 1.0)))

    def _macros(self, txns: list[Transaction]) -> dict[str, float]:
        firms = self.firms()
        X = self.agent_set.stack()

        def amt_of(kind: str, sector: str | None = None) -> float:
            return sum(
                t.amount for t in txns
                if t.kind == kind and (sector is None or t.sector == sector)
            )

        consumption_by_sector = {sec: amt_of("purchase", sec) for sec in GOODS_SECTORS}
        gdp = sum(consumption_by_sector.values())
        unemployment = sum(
            self.hours_per_hh - hh.get("labor_supply") for hh in self.households()
        )
        return {
            "GDP": float(gdp),
            "C_food": float(consumption_by_sector["food"]),
            "C_services": float(consumption_by_sector["services"]),
            "C_durables": float(consumption_by_sector["durables"]),
            "wages": float(amt_of("wage")),
            "tax_revenue": float(amt_of("tax")),
            "transfers": float(amt_of("transfer")),
            "interest_paid": float(amt_of("interest")),
            "deposit_interest_paid": float(amt_of("deposit_interest")),
            "dividends_paid": float(amt_of("dividend")),
            "loans_originated": float(amt_of("loan_origination")),
            "loans_repaid": float(amt_of("loan_repayment")),
            "unemployment_hours": float(unemployment),
            "money_stock": float(X[:, COORD_IDX["money"]].sum()),
            "debt_outstanding": float(X[:, COORD_IDX["debt_liab"]].sum()),
            "inventory_food": float(X[:, COORD_IDX["inv_food"]].sum()),
            "inventory_services": float(X[:, COORD_IDX["inv_services"]].sum()),
            "inventory_durables": float(X[:, COORD_IDX["inv_durables"]].sum()),
            "interest_rate": float(self.interest_rate()),
            "period": int(self.period),
        }


# ---------------------------------------------------------------------------
# Conservation checks. Each residual should be 0 to machine precision.
# ---------------------------------------------------------------------------
def check_conservation(traj: Trajectory, tol: float = 1e-8) -> dict[str, float]:
    states = traj.stack_states()              # (T, N, D)
    money = states[..., COORD_IDX["money"]].sum(axis=1)
    debt_net = (states[..., COORD_IDX["debt_liab"]]
                - states[..., COORD_IDX["debt_asset"]]).sum(axis=1)
    labor_s = states[..., COORD_IDX["labor_supply"]].sum(axis=1)
    labor_d = states[..., COORD_IDX["labor_demand"]].sum(axis=1)

    residuals: dict[str, float] = {
        "money_drift": float(np.max(np.abs(money - money[0]))),
        "debt_net":   float(np.max(np.abs(debt_net))),
        "labor_balance": float(np.max(np.abs(labor_s - labor_d))),
    }
    for sec in GOODS_SECTORS:
        gin = states[..., COORD_IDX[GOODS_IN_COORDS[sec]]].sum(axis=1)
        gout = states[..., COORD_IDX[GOODS_OUT_COORDS[sec]]].sum(axis=1)
        residuals[f"goods_balance_{sec}"] = float(np.max(np.abs(gin - gout)))
    return residuals


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    econ = Economy.small()
    shocks = [{"transfer_per_hh": 0.05}] * 30
    shocks[15] = {"transfer_per_hh": 0.20, "productivity_mult_durables": 1.10}
    traj = econ.rollout(n_periods=30, shocks=shocks)

    print(f"trajectory: T={traj.T}, agents={traj.n_agents}, dim={traj.dim}")
    print(f"  total transactions: {sum(len(p) for p in traj.txn_log)}")
    kinds = sorted({t.kind for p in traj.txn_log for t in p})
    print(f"  txn kinds: {kinds}")
    sectors = sorted({t.sector for p in traj.txn_log for t in p if t.sector})
    print(f"  txn sectors: {sectors}")

    print("\nmacros (5-period snapshots):")
    hdr = ("t", "GDP", "C_food", "C_svc", "C_dur", "loans", "debt", "M")
    print("  " + " ".join(f"{h:>8s}" for h in hdr))
    for i in (0, 5, 10, 15, 20, 25, 29):
        if i >= len(traj.macros):
            continue
        m = traj.macros[i]
        print(f"  {m['period']:>8d} {m['GDP']:>8.2f} {m['C_food']:>8.2f} "
              f"{m['C_services']:>8.2f} {m['C_durables']:>8.2f} "
              f"{m['loans_originated']:>8.2f} {m['debt_outstanding']:>8.2f} "
              f"{m['money_stock']:>8.2f}")

    print("\nconservation residuals (should be ~0):")
    for k, v in check_conservation(traj).items():
        print(f"  {k:<22s} {v:.2e}")
