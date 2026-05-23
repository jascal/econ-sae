# architecture WorldModel

> Baseline world-model MLP: 43 -> 96 -> 48 -> 23. The first hidden layer `h1` (post-ReLU) is the canonical SAE training target — its activations are extracted per agent per period and fed into the encoder.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| input_dim | int  | 43      |
| h1_dim    | int  | 96      |
| h2_dim    | int  | 48      |
| out_dim   | int  | 23      |

## tensors

| Name | Shape          | Dtype   |
|------|----------------|---------|
| x    | (B, input_dim) | float32 |
| y    | (B, out_dim)   | float32 |

## layer x [input]
> Concatenated [agent_state, macro_state, shock_state]

## layer fc1
- op: Linear(input_dim, h1_dim)

## layer act1
> ReLU on H1 — SAE training substrate is read here
- op: ReLU()

## layer fc2
- op: Linear(h1_dim, h2_dim)

## layer act2
- op: ReLU()

## layer head
- op: Linear(h2_dim, out_dim)

## layer y [output]
> Predicted next-period agent state

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| x      | fc1    | x      |
| fc1    | act1   | z1     |
| act1   | fc2    | h1     |
| fc2    | act2   | z2     |
| act2   | head   | h2     |
| head   | y      | y_hat  |

## invariants
- output_shape: (B, out_dim)
