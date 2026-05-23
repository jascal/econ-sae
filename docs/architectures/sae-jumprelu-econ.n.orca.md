# architecture JumpReLUSae_Econ

> JumpReLU sparse autoencoder (theta_init=0.1, l0_coeff=0.005). Each feature has a learnable threshold `theta_j`; the gate `x_j * 1{x_j > theta_j}` is hard-thresholded with a straight-through estimator. Sparsity is encouraged via an L0 count penalty (not part of the forward graph rendered here).

## hyperparameters

| Name       | Type  | Default |
|------------|-------|---------|
| input_dim  | int   | 96      |
| n_features | int   | 512     |
| theta_init | float | 0.1     |
| l0_coeff   | float | 0.005   |

## tensors

| Name  | Shape          | Dtype   |
|-------|----------------|---------|
| x     | (B, input_dim) | float32 |
| x_hat | (B, input_dim) | float32 |

## layer x [input]
> LLM / world-model activation to reconstruct

## layer x_hat [output]
> Reconstructed activation

## layer encoder
- op: Linear(input_dim, n_features)

## layer jumprelu
> Per-feature learnable hard threshold + STE gate
- op: JumpReLU(n_features, theta_init)

## layer decoder
- op: Linear(n_features, input_dim)

## flow

| Source   | Target   | Tensor  |
|----------|----------|---------|
| x        | encoder  | x       |
| encoder  | jumprelu | z_pre   |
| jumprelu | decoder  | z_gated |
| decoder  | x_hat    | x_hat   |

## invariants
- output_shape: (B, input_dim)

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
