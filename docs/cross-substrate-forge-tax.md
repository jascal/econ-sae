# Cross-substrate: the cov95 forge tax exists in both bio-sae and econ-sae — but its *mechanism* differs

`scripts/forge_cov_mechanism.py` (econ-sae) ports bio-sae's **N1 mechanism
ablation** to the econ substrate. The headline: **the forge tax is universal in
existence but host-architecture-specific in mechanism** — so the right fix is
host-dependent, not a single universal knob.

## What bio-sae's N1 found (frozen ESM-2)

bio-sae forges a **frozen, foreign, deep transformer**. Its N1 ablation
*exonerated every single knob*: the forge keeps full rank 320 yet smears cov95
(0.72 → 0.04); a **rank-128 projection of the host keeps cov95 0.685** (96% of host
0.717), rank-32 still 0.533; one LayerNorm does nothing; TopK is minor. Conclusion:
the tax is an **emergent distortion of the deep forward pass**, *not* rank /
over-completeness. The fix there is to **route around it** — preserve the sharp
atoms verbatim (validated: K≈160 atoms → host cov95).

## What econ-sae's N1 finds (trainable DualHeadRegimeWM)

Host = `DualHeadRegimeWM` (per-period attention + LayerNorm + GRU + dense fc1/fc2
bridge), SAE = JumpReLU on the 223-d macro feed (n=6400 periods). Tiers present:
categorical (21) + regime (6).

**N1-rank — rank-SENSITIVE (the opposite of bio).** The strong tier (regime, host
cov95 1.0) needs **near-full rank**: rank-128 → 0.67, rank-64 → 0.33, rank-32 →
0.50 (vs bio's Pfam holding 0.96 at rank-128). All-GT cov95: rank-128 0.259 vs full
0.333 (78%). The GT signal is **spread across the full subspace**, not low-rank
concentrated.

**N1-width — over-completeness DEGRADES cov95 (bio exonerated it).** Same host/feed,
three SAE widths:

| SAE width | over-complete | cov95 (all) | regime | categorical | mAUC |
|---|---|---|---|---|---|
| 512 | 2.3× | 0.333 | 1.00 | 0.14 | 0.733 |
| 1024 | 4.6× | 0.296 | 1.00 | 0.10 | 0.717 |
| 2048 | 9.2× | 0.185 | 0.67 | 0.05 | 0.718 |

cov95 falls monotonically with over-completeness (mAUC stays ~0.72 — the same
mAUC-robust / cov95-fragile split bio shows). This matches econ's own forge-MSE
blow-up with width (w256 MSE 5.3 → w1024 MSE 443). *Caveat: the three widths are
separately-trained SAEs, so this conflates trainability with over-completeness —
suggestive, not airtight.*

**N1-LN** — one LayerNorm → 0.370 (≈ host 0.333). Exonerated, **same as bio**.

## The synthesis

| | bio-sae | econ-sae |
|---|---|---|
| host | frozen, foreign, deep transformer (ESM-2) | trainable, dense fc1/fc2 bridge + attention |
| rank | **robust** (rank-128 → 96% of host) | **sensitive** (rank-128 → 67% on regime) |
| over-completeness | **exonerated** | **degrades cov95** (0.33 → 0.19 at 9.2×) |
| LayerNorm (1×) | exonerated | exonerated |
| tax mechanism | **emergent forward distortion** | **rank / over-completeness bottleneck** |
| → the lever | **preserve verbatim** (route around) | **concentrate the basis** (reduce over-completeness; supervised concentration) |

**The escape from the forge tax depends on whether you own the host.** A frozen
foreign host (bio) gives you no handle on the forward pass, so you preserve the
fragile atoms verbatim. A trainable host (econ) lets you concentrate the basis so
few dims need forging — which econ already demonstrated (supervised dual-head → 128
feats → 6 clusters, forge MSE 0.78). These are *complementary* levers across the
program, selected by host-ownership, not rival theories of one universal tax.

Caveats: econ's probe is the per-period macro feed (categorical + regime tiers
only; no conjunctive/bucketed); it is a host-side ablation (the forged-vs-host tax
is reported in the existing `runs/forge/...` results); the rank-robustness gap is
strong on regime, moderate on categorical. A cleaner confirmation would re-run on a
single SAE with a true width-controlled over-completeness sweep.
