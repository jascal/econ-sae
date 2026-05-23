# architecture L1Sae_Econ

> Vanilla L1 sparse autoencoder (l1_coeff=0.01). ReLU pre-activation gives non-negative features; sparsity is encouraged via an L1 penalty `l1_coeff * |z|.sum()` added to the reconstruction loss (the penalty is not part of the forward graph rendered here).

## hyperparameters

| Name       | Type  | Default |
|------------|-------|---------|
| input_dim  | int   | 96      |
| n_features | int   | 512     |
| l1_coeff   | float | 0.01    |

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

## layer relu
- op: ReLU()

## layer decoder
- op: Linear(n_features, input_dim)

## flow

| Source  | Target  | Tensor |
|---------|---------|--------|
| x       | encoder | x      |
| encoder | relu    | z_pre  |
| relu    | decoder | z      |
| decoder | x_hat   | x_hat  |

## invariants
- output_shape: (B, input_dim)

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
