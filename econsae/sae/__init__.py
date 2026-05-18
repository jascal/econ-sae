"""SAE training, evaluation, and feeds for econ-sae.

Lands in the next iteration. The plan:

  data.py       -- feeds analogous to sm-sae:
                     feed_raw       :  agent vectors stacked across all periods
                     feed_macros    :  per-period macro aggregate vectors
                     feed_acts      :  activations of a small predictive world
                                      model fit to trajectories
                   each feed carries a `sample_features` set of ground-truth
                   labels (sector, txn_kind_present, shock_kind, ...) for AUC
                   alignment scoring.
  models.py     -- TopK / L1 / JumpReLU SAE classes (will share the base
                   class layout from sm-sae's smsae/sae/models.py).
  train.py      -- training loop with dead-neuron resampling.
  evaluation.py -- AUC-of-feature-vs-ground-truth alignment, coverage,
                   monosemanticity, intervention/steering tests.
"""
