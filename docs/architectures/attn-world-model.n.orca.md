# architecture AttnWorldModel

> Per-agent attention world model: embed -> MHA -> +residual -> LN -> MLP(192->128->23). Multi-head self-attention is applied across the agent dimension each period, so each agent's prediction can attend to every other agent's state.

## hyperparameters

| Name      | Type  | Default |
|-----------|-------|---------|
| input_dim | int   | 43      |
| embed_dim | int   | 64      |
| n_heads   | int   | 4       |
| h1_dim    | int   | 192     |
| h2_dim    | int   | 128     |
| out_dim   | int   | 23      |
| dropout   | float | 0.0     |

## tensors

| Name | Shape             | Dtype   |
|------|-------------------|---------|
| x    | (B, N, input_dim) | float32 |
| y    | (B, N, out_dim)   | float32 |

## layer x [input]
> Per-agent state — N agents, input_dim features each

## layer embed
- op: Linear(input_dim, embed_dim)

## layer attn
> Multi-head self-attention across agents
- op: MultiHeadAttention(embed_dim, n_heads, dropout)

## layer add_attn
- op: Add()

## layer ln
- op: LayerNorm(embed_dim)

## layer fc1
- op: Linear(embed_dim, h1_dim)

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

## flow

| Source   | Target   | Tensor   |
|----------|----------|----------|
| x        | embed    | x        |
| embed    | attn     | tokens   |
| attn     | add_attn | attn_out |
| embed    | add_attn | tok_skip |
| add_attn | ln       | r        |
| ln       | fc1      | r_n      |
| fc1      | act1     | z1       |
| act1     | fc2      | h1       |
| fc2      | act2     | z2       |
| act2     | head     | h2       |
| head     | y        | y_hat    |

## invariants
- output_shape: (B, N, out_dim)
