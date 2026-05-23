# architecture TopKSae_Econ

> Top-k sparse autoencoder (k=16). Sparsity is enforced structurally — every sample keeps exactly `k` features active. No L1 / L0 penalty needed at the loss layer.

## hyperparameters

| Name       | Type | Default |
|------------|------|---------|
| input_dim  | int  | 96      |
| n_features | int  | 512     |
| k          | int  | 16      |

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

## layer topk
> Keep only the k largest features per sample
- op: TopK(k)

## layer decoder
- op: Linear(n_features, input_dim)

## flow

| Source  | Target  | Tensor   |
|---------|---------|----------|
| x       | encoder | x        |
| encoder | relu    | z_pre    |
| relu    | topk    | z_relu   |
| topk    | decoder | z_sparse |
| decoder | x_hat   | x_hat    |

## invariants
- output_shape: (B, input_dim)

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
