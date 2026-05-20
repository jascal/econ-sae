"""Single-file HTML walkthrough of the econ-sae project.

Mirrors sm-sae's scripts/visualize.py pattern but specific to econ-sae's
substrate. Six sections, each pinned to a question a newcomer might ask:

  (a) substrate       agent embeddings + the 23 coords + conservation invariants
  (b) simulator       running one trajectory + macros over time + conservation residuals
  (c) sae             per-tier alignment results, alignment heatmaps, multi-decoder recipe
  (d) polygram        polygram bridge results: interference sweep + cancellation pairs
  (e) scoreboard      best-in-class per-tier benchmark + the recommended recipe
  (f) phase journey   how each phase moved the ceiling (Phase 1 -> Phase 5.1)

Inputs read (any missing piece is reported in-place, not fatal):
  data/econ_ensemble.npz                              -- built by scripts/build_data.py
  runs/alignment_summary.json                         -- built by scripts/evaluate.py
  runs/{macro_feed_*, scale, attn, temporal_*,
        gated_sae, regime_supervised}/*_summary.json  -- built by experiment scripts
  runs/polygram/polygram_summary.json                 -- built by scripts/polygram_demo.py

Output: a self-contained HTML file (default docs/index.html, so the
report is directly servable via GitHub Pages with Source = /docs);
all plots inlined as base64 PNGs, so the result is one shareable
artifact.

    python scripts/visualize.py [--out path]

Requires matplotlib. Install: `pip install -e ".[viz]"`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
from html import escape

from econsae.polygram_bridge import efficiency_for_display

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    sys.stderr.write(
        "visualize.py requires matplotlib and numpy.\n"
        "Install with:  pip install -e \".[viz]\"\n"
    )
    raise


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def fig_to_uri(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(uri: str, caption: str = "") -> str:
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{uri}"/>{cap}</figure>'


def missing(path: str, hint: str = "") -> str:
    h = f" &mdash; {escape(hint)}" if hint else ""
    return (f'<div class="missing">missing: <code>{escape(path)}</code>{h}</div>')


def safe(name: str, fn):
    try:
        return fn()
    except Exception as e:
        tb = traceback.format_exc()
        return (f'<section><h2>{escape(name)} (failed)</h2>'
                f'<div class="error"><pre>{escape(tb)}</pre></div></section>')


def load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt_auc(v: float) -> str:
    """AUC with a color hint."""
    klass = "auc-good" if v >= 0.95 else ("auc-mid" if v >= 0.80 else "auc-low")
    return f'<span class="{klass}">{v:.3f}</span>'


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


# ---------------------------------------------------------------------------
# Reusable feature-tier classifier
# ---------------------------------------------------------------------------
def feature_tier(name: str) -> str:
    from econsae.sae.evaluation import feature_tier as _ft
    return _ft(name)


# ---------------------------------------------------------------------------
# (a) Substrate
# ---------------------------------------------------------------------------
def section_substrate() -> str:
    from econsae.embeddings import (
        COORDS, DIM, CONSERVED, SIDE, STOCK_COORDS, FLOW_COORDS,
        build_economy,
    )
    from econsae.simulator.core import TXN_KINDS
    from econsae.sectors import (
        GOODS_SECTORS, HH_COHORTS, COHORT_PROFILES, FIRM_SECTOR_PROFILES,
        IO_MATRIX,
    )

    # Coord layout table
    coord_rows = []
    block_of = {}
    for i, c in enumerate(COORDS):
        if CONSERVED.start <= i < CONSERVED.stop:
            block_of[c] = "conserved"
        else:
            block_of[c] = "side"
    for i, c in enumerate(COORDS):
        sub = ""
        if c in STOCK_COORDS:
            sub = " (stock)"
        elif c in FLOW_COORDS:
            sub = " (flow)"
        coord_rows.append(
            f'<tr><td>{i}</td><td><code>{escape(c)}</code></td>'
            f'<td class="block-{block_of[c]}">{block_of[c]}{sub}</td></tr>'
        )
    coord_table = (
        '<table class="hier"><thead><tr><th>idx</th>'
        '<th>coord</th><th>block</th></tr></thead><tbody>'
        + "".join(coord_rows) + '</tbody></table>'
    )

    # Cohorts table
    cohort_rows = []
    for cname in HH_COHORTS:
        p = COHORT_PROFILES[cname]
        cohort_rows.append(
            f'<tr><td><code>{cname}</code></td>'
            f'<td>{p.mpc_mean:.2f}</td><td>{p.productivity_mean:.2f}</td>'
            f'<td>{p.reservation_wage:.2f}</td><td>{p.tax_rate:.2f}</td>'
            f'<td>{p.ces_weights[0]:.2f} / {p.ces_weights[1]:.2f} / '
            f'{p.ces_weights[2]:.2f}</td>'
            f'<td>{p.max_dti:.2f}</td></tr>'
        )
    cohort_table = (
        '<table class="hier"><thead><tr><th>cohort</th><th>mpc</th>'
        '<th>productivity</th><th>res. wage</th><th>tax_rate</th>'
        '<th>CES (food/svc/dur)</th><th>max DTI</th></tr></thead><tbody>'
        + "".join(cohort_rows) + '</tbody></table>'
    )

    # Firm sectors table
    firm_rows = []
    for sname in GOODS_SECTORS:
        p = FIRM_SECTOR_PROFILES[sname]
        firm_rows.append(
            f'<tr><td><code>{sname}</code></td>'
            f'<td>{p.productivity_mean:.2f}</td>'
            f'<td>{p.markup_target:.2f}</td>'
            f'<td>{p.price_stickiness:.2f}</td>'
            f'<td>{p.initial_inventory:.0f}</td>'
            f'<td>{p.initial_price:.2f}</td>'
            f'<td>{p.cyclicality:.2f}</td></tr>'
        )
    firm_table = (
        '<table class="hier"><thead><tr><th>sector</th><th>productivity</th>'
        '<th>markup</th><th>price stickiness</th><th>init inv</th>'
        '<th>init price</th><th>cyclicality</th></tr></thead><tbody>'
        + "".join(firm_rows) + '</tbody></table>'
    )

    # I-O matrix heatmap
    io_arr = np.array([[IO_MATRIX[r][c] for c in GOODS_SECTORS] for r in GOODS_SECTORS])
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(io_arr, cmap="Blues", vmin=0.0)
    ax.set_xticks(range(len(GOODS_SECTORS)))
    ax.set_xticklabels(GOODS_SECTORS)
    ax.set_yticks(range(len(GOODS_SECTORS)))
    ax.set_yticklabels(GOODS_SECTORS)
    ax.set_xlabel("input sector")
    ax.set_ylabel("output sector")
    ax.set_title("IO_MATRIX[row][col] = units of <col> per unit of <row>")
    for i in range(io_arr.shape[0]):
        for j in range(io_arr.shape[1]):
            ax.text(j, i, f"{io_arr[i, j]:.2f}", ha="center", va="center",
                    color="black" if io_arr[i, j] < 0.07 else "white", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    io_uri = fig_to_uri(fig)

    # Example agent: build a small economy + show one HH and one firm vector
    econ = build_economy(households_per_cohort=2, firms_per_sector=1, seed=0)
    sample_hh = next(a for a in econ.agents if a.kind == "household")
    sample_firm = next(a for a in econ.agents if a.kind == "firm")
    sample_bank = next(a for a in econ.agents if a.kind == "bank")
    examples = []
    for label, ag in (("household (young #0)", sample_hh),
                      ("firm (food #0)", sample_firm),
                      ("bank", sample_bank)):
        vals = " ".join(
            f'<td title="{escape(c)}" class="block-{block_of[c]}">'
            f'{ag.vec[i]:.2f}</td>'
            for i, c in enumerate(COORDS)
        )
        examples.append(
            f'<tr><th>{escape(label)}</th>{vals}</tr>'
        )
    head = "<tr><th></th>" + "".join(
        f'<th class="block-{block_of[c]}" title="{escape(c)}">{escape(c[:6])}</th>'
        for c in COORDS
    ) + "</tr>"
    examples_table = (
        '<table class="agentvec"><thead>' + head + '</thead><tbody>'
        + "".join(examples) + '</tbody></table>'
    )

    # Transaction kinds list
    txn_list = "".join(f"<code>{escape(k)}</code> " for k in TXN_KINDS)

    return f"""
<section id="substrate">
  <h2>(a) The substrate: agents as vectors in R^{DIM}</h2>
  <p>Each economic agent (household, firm, government, bank) is a point in
  R<sup>{DIM}</sup>. The coordinates split into a <strong>conserved block</strong>
  (indices 0&ndash;13: stocks and per-period flows that obey exact accounting
  identities at every transaction) and a <strong>side block</strong> (indices
  14&ndash;22: agent parameters that drive dynamics).</p>

  <h3>Coordinate layout</h3>
  {coord_table}

  <h3>Three example agent rows (default economy, seed=0)</h3>
  <p class="aside">Column headers truncated to 6 chars; hover for full name. Cells
  shaded by block (conserved &mdash; blue; side &mdash; gray).</p>
  <div class="tensorwrap">{examples_table}</div>

  <h3>Household cohorts</h3>
  <p>Three cohorts split the 12 households into behavioral classes. Cohort
  drives MPC, productivity, reservation wage, sector preferences (CES weights
  over food / services / durables), and credit access (max debt-to-income).</p>
  {cohort_table}

  <h3>Firm sectors</h3>
  <p>Three firm sectors with distinct production technology, pricing
  flexibility, and cyclicality.</p>
  {firm_table}

  <h3>Input-output network (Phase 3.4)</h3>
  <p>With <code>io_network=True</code>, each firm purchases intermediate
  goods from each sector before producing. IO_MATRIX[row][col] entries are
  the units of <em>col</em>'s good needed per unit of <em>row</em>'s output.
  A productivity shock to <code>durables</code> (the row that supplies all
  three columns) propagates through goods flows to every other sector.</p>
  {img(io_uri, "I-O matrix: column-supplier, row-consumer.")}

  <h3>Nine transaction kinds (vertices)</h3>
  <p>Every transaction is a double-entry vertex that updates two agents'
  vectors with opposite signs on the conserved block. The kinds:</p>
  <p class="kinds">{txn_list}</p>
  <p class="aside">All nine fire under stochastic shocks; <code>b2b_purchase</code>
  is conditional on the I-O network being enabled.</p>

  <h3>Conservation laws (the ground-truth axes)</h3>
  <p>For every period <em>t</em> and every trajectory:</p>
  <ul>
    <li><code>sum_i money[i]</code> = M_initial (closed economy)</li>
    <li><code>sum_i (debt_liab - debt_asset)[i]</code> = 0 (every loan two-sided)</li>
    <li><code>sum_i labor_supply[i]</code> = <code>sum_i labor_demand[i]</code></li>
    <li><code>sum_i goods_in_food[i]</code> = <code>sum_i goods_out_food[i]</code>
        (and services, durables)</li>
  </ul>
  <p>These six identities form a constructive ground truth: every trained
  model can be graded against them without post-hoc interpretation. The
  simulator preserves them to ~1e-13 (machine precision) across every
  trajectory in every experiment.</p>
</section>
"""


# ---------------------------------------------------------------------------
# (b) Simulator: example trajectory
# ---------------------------------------------------------------------------
def section_simulator() -> str:
    from econsae.simulator.core import Economy, check_conservation

    econ = Economy.small(households_per_cohort=4, firms_per_sector=1, seed=0)
    traj = econ.rollout(n_periods=40)

    # Macros over time
    keys = ["GDP", "C_food", "C_services", "C_durables",
            "wages", "tax_revenue", "interest_paid",
            "debt_outstanding", "money_stock", "interest_rate"]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True)
    for ax, k in zip(axes.flat, keys):
        vals = [m[k] for m in traj.macros]
        ax.plot(vals, color="#225", lw=1.4)
        ax.set_title(k, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.4)
    fig.suptitle("Macros over 40 periods (Phase 0.5 closed economy, no shocks)", fontsize=11)
    fig.tight_layout()
    macros_uri = fig_to_uri(fig)

    # Transaction counts per kind, summed over all periods
    txn_counts: dict[str, int] = {}
    for period in traj.txn_log:
        for t in period:
            key = t.kind if t.sector is None else f"{t.kind}:{t.sector}"
            txn_counts[key] = txn_counts.get(key, 0) + 1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    items = sorted(txn_counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    ax.bar(range(len(names)), counts, color="#558")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("count over 40 periods")
    ax.set_title("Transaction kind occurrence (one example trajectory)")
    fig.tight_layout()
    txn_uri = fig_to_uri(fig)

    # Conservation residuals
    res = check_conservation(traj)
    res_rows = "".join(
        f"<tr><td><code>{escape(k)}</code></td><td>{v:.2e}</td></tr>"
        for k, v in res.items()
    )
    res_table = (
        '<table class="summary"><thead><tr><th>identity</th>'
        '<th>max residual</th></tr></thead><tbody>'
        + res_rows + '</tbody></table>'
    )

    return f"""
<section id="simulator">
  <h2>(b) Running the simulator: one trajectory of 40 periods</h2>
  <p>This is the baseline (no shocks, no Phase 3 features) trajectory of the
  default 12 HH + 3 firm + 1 gov + 1 bank economy. The simulator runs each
  period through a deterministic sequence of transaction steps (interest
  payments, production, wages, taxes, transfers, consumption, repayment,
  dividends, sticky-price update, expectations update). Every flow is a
  double-entry transaction; conservation holds at machine precision.</p>

  {img(macros_uri, "Macros over time. GDP and consumption settle into a stable steady state under the balanced-budget fiscal rule. interest_rate is constant (no monetary shocks); debt grows monotonically as young HHs draw on consumer credit.")}

  <h3>Transaction-kind occurrence</h3>
  {img(txn_uri, "Counts per transaction kind across the trajectory. wage / purchase dominate (every period for every employed household); loan_origination is rarer (only when an agent is liquidity-constrained).")}

  <h3>Conservation residuals (should all be ~1e-13)</h3>
  {res_table}
  <p class="aside">These six identities are the constructive ground truth for
  the SAE benchmark. The simulator preserves them exactly under every
  combination of shocks and Phase 3 features.</p>
</section>
"""


# ---------------------------------------------------------------------------
# (c) SAE: alignment results + multi-decoder recipe
# ---------------------------------------------------------------------------
def section_sae() -> str:
    # Aggregate per-feature best-AUC across all experiment summaries
    summaries = [
        ("Phase 1 baseline",       "runs/alignment_summary.json",                "auc_per_feature"),
        ("Phase 1.5 scale",        "runs/scale_experiment_summary.json",         "conj_auc_per_feature"),
        ("Phase 1.6 attention",    "runs/attn_experiment_summary.json",          "conj_auc_per_feature"),
        ("Phase 1.7 temporal",     "runs/temporal_sentiment_experiment_summary.json", "auc_per_feature"),
        ("Phase 1.9 macro-feed v1","runs/macro_feed_experiment_summary.json",    "regime_auc_per_feature"),
        ("Phase 2.0 macro-feed v2","runs/macro_feed_v2_experiment_summary.json", "regime_auc_per_feature"),
        ("Phase 3.2 GatedSAE",     "runs/gated_sae_experiment_summary.json",     "regime_auc_per_feature"),
        ("Phase 4.1 macro-feed v3","runs/macro_feed_v3_experiment_summary.json", "regime_auc_per_feature"),
        ("Phase 4.2 full features","runs/phase3_features_experiment_summary.json", "auc_per_feature"),
        ("Phase 5.1 regime-sup",   "runs/regime_supervised_experiment_summary.json", "regime_auc_per_feature"),
    ]
    best_per_feature: dict[str, tuple[float, str]] = {}
    available_phases = []
    for label, path, key in summaries:
        data = load_json(path)
        if data is None:
            continue
        available_phases.append(label)
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            per = r.get(key) or {}
            for f, auc in per.items():
                cur = best_per_feature.get(f, (0.0, ""))
                if auc > cur[0]:
                    best_per_feature[f] = (float(auc), label)

    # Per-tier breakdown table
    tier_rows = {"categorical": [], "bucketed": [], "conjunctive": [], "regime": []}
    for f, (auc, src) in best_per_feature.items():
        tier = feature_tier(f)
        if tier in tier_rows:
            tier_rows[tier].append((f, auc, src))

    # Build per-tier HTML tables
    tier_html = []
    for tier in ("categorical", "bucketed", "conjunctive", "regime"):
        rows = sorted(tier_rows[tier], key=lambda x: -x[1])
        if not rows:
            tier_html.append(f"<h4>{tier} (none)</h4>")
            continue
        n_at_95 = sum(1 for _, a, _ in rows if a >= 0.95)
        mean_auc = sum(a for _, a, _ in rows) / len(rows)
        head = (f"<h4>{tier} &mdash; {n_at_95}/{len(rows)} at AUC&ge;0.95, "
                f"mean = {mean_auc:.3f}</h4>")
        if tier == "categorical" and len(rows) > 12:
            shown = rows[:6] + rows[-6:]
            note = f"<p class='aside'>showing best 6 + worst 6 of {len(rows)} categorical features.</p>"
        else:
            shown = rows
            note = ""
        body = "".join(
            f"<tr><td><code>{escape(f)}</code></td><td>{fmt_auc(a)}</td>"
            f"<td class='src'>{escape(s)}</td></tr>"
            for f, a, s in shown
        )
        tier_html.append(
            head + note +
            '<table class="summary"><thead><tr><th>feature</th>'
            '<th>best AUC</th><th>recipe</th></tr></thead><tbody>'
            + body + '</tbody></table>'
        )

    # Bar chart: regime feature AUC across phases
    regime_phases_chart = ""
    regime_features = [
        "phase:expansion", "phase:contraction",
        "phase:fiscal_active", "phase:monetary_active",
        "phase:high_rate", "phase:high_leverage",
    ]
    phase_data: dict[str, dict[str, float]] = {}
    for label, path, key in summaries:
        data = load_json(path)
        if data is None:
            continue
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            per = r.get(key) or {}
            for f in regime_features:
                if f in per:
                    cur = phase_data.setdefault(label, {})
                    cur[f] = max(cur.get(f, 0.0), float(per[f]))

    if phase_data:
        phases_list = [p for p, _, _ in summaries if p in phase_data]
        feat_x = regime_features
        fig, ax = plt.subplots(figsize=(11.5, 5.5))
        n_phases = len(phases_list)
        width = 0.8 / max(n_phases, 1)
        x = np.arange(len(feat_x))
        cmap = plt.get_cmap("viridis")
        for i, label in enumerate(phases_list):
            vals = [phase_data[label].get(f, 0.5) for f in feat_x]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
                   label=label, color=cmap(i / max(n_phases - 1, 1)))
        ax.axhline(0.95, color="red", lw=1, ls="--", alpha=0.6,
                   label="AUC = 0.95")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("phase:", "") for f in feat_x],
                            rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("best AUC")
        ax.set_ylim(0.45, 1.02)
        ax.set_title("Regime feature AUC across project phases")
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        regime_phases_chart = img(
            fig_to_uri(fig),
            "Per-feature AUC over the project's six experimental phases. "
            "The red dashed line is the AUC = 0.95 cov95 threshold. "
            "Some features were already recoverable at Phase 1 (the impulse "
            "features moved later when impulse flags were added to the "
            "input). Others (expansion / contraction) needed the supervised "
            "regime head in Phase 5.1 to climb above 0.90."
        )

    phases_seen = "".join(f"<li>{escape(p)}</li>" for p in available_phases)

    return f"""
<section id="sae">
  <h2>(c) SAE: per-tier alignment and the multi-decoder recipe</h2>
  <p>The ground-truth feature vocabulary spans four tiers of difficulty
  (categorical, bucketed, conjunctive, regime). Each tier has a different
  recovery recipe; no single (substrate, SAE) pair recovers everything.</p>
  <p>Experiments aggregated below:</p>
  <ul>{phases_seen}</ul>

  {regime_phases_chart}

  <h3>Best-in-class per-tier recovery</h3>
  {"".join(tier_html)}
</section>
"""


# ---------------------------------------------------------------------------
# (d) Polygram
# ---------------------------------------------------------------------------
def section_polygram() -> str:
    path = os.path.join(REPO_ROOT, "runs", "polygram", "polygram_summary.json")
    data = load_json(path)
    if data is None:
        return f"""
<section id="polygram">
  <h2>(d) Polygram bridge</h2>
  {missing(path, "run scripts/polygram_demo.py to populate")}
</section>
"""

    dictionary = data["dictionary"]
    by_cluster = dictionary["betas_by_cluster"]
    cluster_rows = "".join(
        f"<tr><td><code>{escape(k)}</code></td><td>{v['n']}</td>"
        f"<td>{v['min']:+.3f}</td><td>{v['median']:+.3f}</td>"
        f"<td>{v['max']:+.3f}</td></tr>"
        for k, v in by_cluster.items()
    )
    cluster_table = (
        '<table class="summary"><thead><tr><th>tier</th><th>n</th>'
        '<th>beta min</th><th>beta median</th><th>beta max</th></tr></thead>'
        '<tbody>' + cluster_rows + '</tbody></table>'
    )

    # Interference sweep result
    sweep = data.get("interference_sweep") or {}
    sweep_html = ""
    if sweep:
        sweep_html = f"""
<h3>Interference sweep: rotate phase on a cross-tier pair</h3>
<p>Target pair <code>{escape(sweep['pair'][0])}</code> /
<code>{escape(sweep['pair'][1])}</code>, sweeping
<code>{escape(sweep['knob'])}</code> from 0 to 2&pi; over
{sweep['n_samples']} values.</p>
<p>Target-pair overlap range: <strong>{sweep['overlap_min']:.4f}</strong> to
<strong>{sweep['overlap_max']:.4f}</strong> (mean {sweep['overlap_mean']:.4f}).</p>
"""

    # Cancellation table
    canc_rows = []
    for r in data.get("cancellations", []):
        if "error" in r:
            canc_rows.append(
                f"<tr><td><code>{escape(r['label'])}</code></td>"
                f"<td>{escape(' / '.join(r['pair']))}</td>"
                f"<td colspan='4' class='error'>ERROR: "
                f"{escape(r['error'][:60])}</td></tr>"
            )
            continue
        eff = efficiency_for_display(r, fmt=".1%")
        canc_rows.append(
            f"<tr><td><code>{escape(r['label'])}</code></td>"
            f"<td>{escape(' / '.join(r['pair']))}</td>"
            f"<td>{r['before_overlap']:.4f}</td>"
            f"<td>{r['after_overlap']:.4f}</td>"
            f"<td>{eff}</td>"
            f"<td>{r['tolerance_met']}</td></tr>"
        )
    canc_table = (
        '<table class="summary"><thead><tr><th>label</th><th>pair</th>'
        '<th>overlap before</th><th>after</th><th>efficiency</th>'
        '<th>met?</th></tr></thead><tbody>'
        + "".join(canc_rows) + '</tbody></table>'
    )

    return f"""
<section id="polygram">
  <h2>(d) Polygram bridge</h2>
  <p>The polygram bridge encodes econ-sae's full 51-feature vocabulary as a
  polygram Dictionary using HEA_Rung2 with <code>n_qubits =
  {dictionary['n_qubits']}</code> ({2 ** dictionary['n_qubits']} feature slots,
  no truncation). Per-feature <em>beta</em> = best-recovered AUC minus 0.5,
  so the "interpretability strength" of each feature appears as the scalar
  on the Dictionary slot.</p>

  <h3>Per-tier beta spread</h3>
  {cluster_table}

  {sweep_html}

  <h3>Cancellation experiments</h3>
  <p>For each pair, polygram searches for phase values that drive the inner
  product to zero. The "after" column is what phase tuning alone achieves
  under the HEA_Rung2 encoding's structural floor (~0.77 here, similar
  across pairs). Cross-tier and within-tier pairs converge to similar
  floors, which is informative about the encoding's geometry.</p>
  {canc_table}
</section>
"""


# ---------------------------------------------------------------------------
# (e) Scoreboard
# ---------------------------------------------------------------------------
def section_scoreboard() -> str:
    # Hard-coded best-in-class for the headline scorecard.
    # These are the same numbers reported in the README's final scoreboard.
    rows = [
        ("categorical",   "30",   "1.000", "most at 1.000",
         "per-agent SAE on attention substrate"),
        ("bucketed",      "7",    "0.928", "most 0.85&ndash;0.93",
         "per-agent SAE"),
        ("conjunctive",   "8",    "0.968", "6/8 at AUC&ge;0.95 (with I-O network + attention)",
         "per-agent SAE on Phase-3 substrate"),
        ("regime",        "6",    "0.972", "4/6 at AUC&ge;0.95 (high_leverage, fiscal, monetary, high_rate)",
         "per-period macro-feed v3 SAE on regime-supervised WM"),
        ("windowed regime", "(subset of regime)", "0.92",
         "remaining 2 features (expansion, contraction) at AUC 0.92",
         "needs feature-bottlenecked supervision or specialized SAE"),
    ]
    body = "".join(
        f"<tr><td>{tier}</td><td>{n}</td>"
        f"<td>{fmt_auc(float(m))}</td><td>{details}</td>"
        f"<td class='src'>{recipe}</td></tr>"
        for tier, n, m, details, recipe in rows
    )
    table = (
        '<table class="scorecard"><thead><tr><th>tier</th><th>n</th>'
        '<th>best mAUC</th><th>best feature recoveries</th>'
        '<th>recipe</th></tr></thead><tbody>'
        + body + '</tbody></table>'
    )

    return f"""
<section id="scoreboard">
  <h2>(e) Final benchmark scoreboard</h2>
  <p>Best-in-class per-tier results across all five experimental phases.
  The recipe column lists the (substrate, decoder) pair that achieved each
  tier's best mAUC: <strong>different tiers need different recipes.</strong>
  No single SAE / world-model combination recovers every tier; the
  multi-decoder recipe is the project's headline finding.</p>

  {table}

  <h3>The recommended two-decoder recipe</h3>
  <ol>
    <li><strong>Train an attention-enabled world model</strong>
        (cross-agent attention block + per-agent MLP head, optionally with
        cross-period GRU) on the multi-trajectory ensemble.</li>
    <li><strong>Train a per-agent SAE</strong> on its h1 activations to
        recover the categorical, bucketed, and conjunctive tiers.</li>
    <li><strong>Train a per-period macro-feed SAE</strong> on
        <code>[macro + shock + GDP_window + impulse_flags + engineered ratios
        + mean-pooled h1]</code> (z-scored) to recover the regime tier.</li>
    <li>Optionally add a supervised regime head during world-model training
        to push the windowed-regime sub-features (expansion / contraction)
        toward AUC 0.95.</li>
  </ol>
</section>
"""


# ---------------------------------------------------------------------------
# (f) Phase journey
# ---------------------------------------------------------------------------
def section_phase_journey() -> str:
    phases = [
        ("Phase 0.5", "Multi-good, multi-cohort, credit-enabled simulator. "
                       "9 transaction kinds, 6 conservation identities preserved at ~1e-13. "
                       "Stochastic shock generator + ensemble runner. "
                       "51-feature ground-truth vocabulary across 4 difficulty tiers."),
        ("Phase 1",   "Per-agent SAE pipeline: TopK / L1 / JumpReLU on three feeds "
                       "(raw, embedded, world-model activations). "
                       "Categorical features recover trivially. Conjunctive mAUC 0.84, "
                       "regime mAUC 0.60."),
        ("Phase 1.5", "Width sweep (256 to 1024). Conjunctive cov95 climbs from 4/8 to 5/8. "
                       "Width helps modestly; duration alone is a no-op."),
        ("Phase 1.6", "AttnWorldModel: cross-agent attention block. "
                       "Conjunctive mAUC 0.84 -> 0.97; 6/8 features at AUC&ge;0.95. "
                       "Biggest single architectural unlock for the per-agent tiers."),
        ("Phase 1.7", "TemporalWorldModel (attention + GRU across periods). "
                       "Falsified the 'temporal architecture unlocks regime' hypothesis. "
                       "Diagnosis: next-state MSE doesn't reward history encoding."),
        ("Phase 1.8", "Sentiment-driven MPC in the simulator. "
                       "Falsified the 'Markovian simulator' diagnosis. "
                       "Agent state already encodes history through accumulated stocks."),
        ("Phase 1.9", "Macro-feed SAE: per-period samples. <code>phase:high_leverage</code> "
                       "jumps from 0.62 to 0.97. Confirms the structural diagnosis: "
                       "regime is per-period, not per-agent."),
        ("Phase 2.0", "Macro-feed v2: GDP window + impulse flags + z-scored input. "
                       "fiscal_active and monetary_active cross AUC 0.99. Regime mAUC 0.86."),
        ("Phase 3",   "Polygram bridge (full 51-feature dictionary, HEA_Rung2). "
                       "GatedSAE (neutral on plateaued features). Taylor rule. "
                       "Input-output firm network."),
        ("Phase 4.1", "Macro-feed v3: engineered ratio inputs (gdp_deviation, "
                       "rate_deviation, leverage_ratio, inflation). "
                       "phase:high_leverage back to 0.97. Regime mAUC 0.885."),
        ("Phase 4.2", "Full Phase 3 features (sentiment + Taylor + I-O) in an SAE pipeline. "
                       "<code>firm_AND_indebted_AND_high_inventory</code> crosses AUC 0.95 "
                       "for the first time (0.92 -> 0.98) thanks to I-O cascades."),
        ("Phase 5.1", "Regime-supervised TemporalWorldModel. Regime mAUC 0.885 -> 0.972 "
                       "(biggest single-phase jump). 4/6 regime features cleanly recovered. "
                       "The remaining 2 (expansion, contraction) climb to 0.92 but plateau "
                       "due to distributed-encoding limitation of pooled supervision."),
    ]
    rows = "".join(
        f"<tr><th>{escape(p)}</th><td>{desc}</td></tr>"
        for p, desc in phases
    )
    return f"""
<section id="journey">
  <h2>(f) Phase journey: how the ceiling moved at each step</h2>
  <p>Five lessons compound across the journey:</p>
  <ol>
    <li><strong>Substrate architecture sets the ceiling for per-agent features.</strong>
        Attention block was the single biggest unlock for conjunctive recovery.</li>
    <li><strong>Decoder granularity matters as much as substrate quality.</strong>
        Per-agent SAE for categorical / bucketed / conjunctive; per-period SAE
        for regime.</li>
    <li><strong>Input encoding matters as much as decoder choice.</strong>
        Impulse flags and engineered ratios each closed a different sub-tier of
        regime features.</li>
    <li><strong>Simulator dynamics drive what's recoverable.</strong> I-O
        cascades made one conjunctive feature cleanly recoverable for the
        first time.</li>
    <li><strong>Some labels need supervision.</strong> Threshold-on-ratio
        regime labels stalled at AUC 0.87 with unsupervised SAEs; supervised
        regime heads pushed them to 1.00 (for high_rate) and 0.92 (for
        expansion / contraction).</li>
  </ol>
  <table class="hier"><tbody>
    {rows}
  </tbody></table>
</section>
"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1180px;
       margin: 2rem auto; padding: 0 1rem; color: #222; line-height: 1.45; }
h1 { border-bottom: 2px solid #224; padding-bottom: .3rem; }
h2 { margin-top: 3rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
h3 { margin-top: 2rem; color: #335; }
h4 { margin-top: 1.5rem; color: #335; }
code, pre { font-family: ui-monospace, Menlo, monospace; }
pre.code { background: #f6f6f8; padding: .8rem; border-radius: 4px;
           font-size: 12px; overflow-x: auto; }
figure { margin: 1rem 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
figcaption { font-size: 12px; color: #555; margin-top: .4rem; }
table { border-collapse: collapse; margin: 1rem 0; font-size: 13px; }
table th, table td { padding: 3px 10px; border-bottom: 1px solid #eee;
                     text-align: left; vertical-align: top; }
table.hier th { background: #eef; padding-top: .5rem; }
table.summary th { background: #eef; }
table.summary td.src { color: #557; font-size: 11px; }
.block-conserved { background: #eef4fb; }
.block-side { background: #f5f5f5; }
table.agentvec th, table.agentvec td { padding: 2px 6px; font-size: 10px;
                                        font-family: ui-monospace, Menlo, monospace;
                                        text-align: right; }
table.agentvec th { font-weight: normal; color: #557; }
.tensorwrap { max-height: 360px; max-width: 100%; overflow: auto;
              border: 1px solid #e2e2e6; padding: 3px;
              background: white; border-radius: 3px; }
.kinds { font-size: 13px; line-height: 1.9; }
.kinds code { background: #f3f3f8; padding: 2px 8px; border-radius: 3px; }
.aside { color: #555; font-style: italic; }
.missing { background: #ffe; border: 1px dashed #aa8; padding: .6rem;
           border-radius: 4px; color: #553; font-size: 13px; }
.error { background: #fee; border: 1px solid #c66; padding: .6rem;
         border-radius: 4px; color: #511; font-size: 12px; }
.auc-good { color: #2a6; font-weight: bold; }
.auc-mid  { color: #b80; }
.auc-low  { color: #c33; }
table.scorecard td { background: #fafaff; }
nav { padding: 1rem; background: #f6f6f8; border-radius: 4px; }
nav a { margin-right: 1rem; color: #224; }
"""


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Default output is docs/index.html so the report can be served directly
    # via GitHub Pages (Settings -> Pages -> Branch=main, folder=/docs).
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "index.html"))
    args = ap.parse_args()

    print(f"Building report -> {args.out}")
    print("  (a) substrate   ...", flush=True)
    sub = safe("substrate", section_substrate)
    print("  (b) simulator   ...", flush=True)
    sim = safe("simulator", section_simulator)
    print("  (c) SAE         ...", flush=True)
    sae = safe("sae", section_sae)
    print("  (d) polygram    ...", flush=True)
    pol = safe("polygram", section_polygram)
    print("  (e) scoreboard  ...", flush=True)
    score = safe("scoreboard", section_scoreboard)
    print("  (f) phase journey...", flush=True)
    journey = safe("phase journey", section_phase_journey)

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>econ-sae: lifecycle walkthrough</title>
<style>{CSS}</style>
</head><body>
<h1>econ-sae lifecycle walkthrough</h1>
<p>econ-sae is a <strong>benchmark fixture</strong> for evaluating techniques
that extract structured representations from sparse autoencoders, sibling
to sm-sae. A stock-flow-consistent simulated macroeconomy (households,
firms, banks, government) gives us exact ground truth at multiple
granularities (sector, cohort, transaction kind, conjunctive composition,
regime / phase). Any candidate SAE technique can be scored quantitatively
against the 51-feature vocabulary across four difficulty tiers.</p>
<p>The headline finding of the project: <strong>different feature tiers
need different (substrate, decoder, input-encoding) recipes.</strong>
No single SAE recovers everything; the multi-recipe scoreboard below
documents what works for each tier.</p>
<nav>
  <a href="#substrate">(a) substrate</a>
  <a href="#simulator">(b) simulator</a>
  <a href="#sae">(c) SAE</a>
  <a href="#polygram">(d) polygram</a>
  <a href="#scoreboard">(e) scoreboard</a>
  <a href="#journey">(f) phase journey</a>
</nav>
{sub}{sim}{sae}{pol}{score}{journey}
</body></html>
"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    size = os.path.getsize(args.out)
    print(f"Wrote {args.out}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
