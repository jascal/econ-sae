# architecture DeepWorldModel

> Deep world-model MLP: 43 -> 192 -> 128 -> 64 -> 23. The first hidden layer is the SAE training substrate; later hidden layers compress before the output head.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| input_dim | int  | 43      |
| h1_dim    | int  | 192     |
| h2_dim    | int  | 128     |
| h3_dim    | int  | 64      |
| out_dim   | int  | 23      |

## tensors

| Name | Shape          | Dtype   |
|------|----------------|---------|
| x    | (B, input_dim) | float32 |
| y    | (B, out_dim)   | float32 |

## layer x [input]

## layer fc1
- op: Linear(input_dim, h1_dim)

## layer act1
> ReLU on first hidden layer — SAE substrate
- op: ReLU()

## layer fc2
- op: Linear(h1_dim, h2_dim)

## layer act2
- op: ReLU()

## layer fc3
- op: Linear(h2_dim, h3_dim)

## layer act3
- op: ReLU()

## layer head
- op: Linear(h3_dim, out_dim)

## layer y [output]

## flow

| Source | Target | Tensor |
|--------|--------|--------|
| x      | fc1    | x1     |
| fc1    | act1   | z1     |
| act1   | fc2    | x2     |
| fc2    | act2   | z2     |
| act2   | fc3    | x3     |
| fc3    | act3   | z3     |
| act3   | head   | h_last |
| head   | y      | y_hat  |

## invariants
- output_shape: (B, out_dim)
