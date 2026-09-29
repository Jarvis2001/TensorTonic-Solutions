**What is this code?**

This is a PyTorch implementation of the **Kimi Delta Attention (KDA) recurrence**. It is a linear attention mechanism that processes a sequence token-by-token, maintaining a fixed-size memory matrix (state) rather than computing a full \(S \times S\) attention matrix.

**What does the function do?**

It takes a sequence of queries (\(Q\)), keys (\(K\)), values (\(V\)), decay logits, write strengths, output gates, and an initial state. It iterates through the sequence, updating the state matrix \(S_t\) at each step, and produces a sequence of outputs and the final state.

**What is the core mathematical formula?**

\[

S_t = (I - \beta_t k_t k_t^T) \text{Diag}(\alpha_t) S_{t-1} + \beta_t k_t v_t^T

\]

\[

\tilde{o}_t = S_t^T q_t

\]

**What are the key components?**

- **State (\(S_t\)):** A matrix of shape `(B, H, D_k, D_v)` acting as associative memory.

- **Decay (\(\alpha_t\)):** Channel-wise retention value derived from `decay_logits`.

- **Write Strength (\(\beta_t\)):** A scalar per head controlling how much new information is written.

- **Erase Term:** \((I - \beta_t k_t k_t^T)\), which removes old associations.

- **Write Term:** \(\beta_t k_t v_t^T\), which adds new associations.

---

**Why use KDA instead of standard Softmax Attention?**

Standard attention has \(O(S^2)\) complexity. KDA has \(O(S)\) complexity, making it highly efficient for long sequences while still allowing dynamic memory updates.

**Why is the decay channel-wise?**

A single scalar decay per head would erase or retain all features uniformly. Channel-wise decay (\(\alpha_t \in \mathbb{R}^{D_k}\)) allows the model to selectively remember some features while forgetting others within the same head.

**Why is there an "erase" term?**

Without the erase term, the state would just accumulate \(k v^T\) indefinitely. If the same key appears again with a new value, the old value would still be there, causing interference. The erase term implements the "delta rule," updating the memory by replacing the old association.

**Why "read after writing"?**

The theory states: _"The current output is read from \(S_t\), not from \(S_{t-1}\)"_. If you read before writing, the current token cannot attend to itself, causing a one-step lag and preventing the first token from using its own information.

**Why RMS normalize the output?**

RMS normalization stabilizes the scale of the readout vector across different heads and sequence steps, preventing exploding activations before the output gate is applied.

**Why does the code clone the initial state?**

To avoid mutating the supplied tensor. The problem explicitly lists "Mutating the initial state" as a common mistake to avoid.

---

**Who created this concept?**

The Kimi Delta Attention (KDA) was introduced by the **Kimi team (Moonshot AI)** as part of their work on efficient linear attention architectures (building upon ideas like DeltaNet and Mamba).



---

**Where is this used in a model architecture?**

It replaces the standard Multi-Head Self-Attention layer in a Transformer block. It is typically sandwiched between a normalization layer and a feed-forward network (MLP).

**Where in the code does the state update happen?**

Inside the `for t in range(S):` loop. Specifically, the lines:

```python

write_term = torch.matmul(erase_term, decay_state)

write = beta_t * torch.matmul(k_t_col, v_t_row)

state = write_term + write

```

**Where does the output projection happen?**

At the end of the loop:

```python

o_out = torch.matmul(o_concat, output_projection.transpose(-1, -2))

```

---

**When does the state update occur?**

Sequentially, at every time step \(t\) from \(0\) to \(S-1\). The state at step \(t\) depends entirely on the state at step \(t-1\).

**When is the output generated?**

Immediately after the state update for the current token \(t\).

**When would you use this in practice?**

During **autoregressive inference** (generating text token by token), where the state can be cached and updated in \(O(1)\) time per token, rather than recomputing the entire attention over the context window.

---

**Which dimensions correspond to what?**

- `B`: Batch size

- `S`: Sequence length

- `H`: Number of attention heads

- `D_k`: Key/Query dimension (width of keys)

- `D_v`: Value dimension (width of values)

**Which lines handle the channel-wise decay?**

```python

alpha = torch.exp(g_min * torch.sigmoid(decay_logits))

alpha_t = alpha[:, t].unsqueeze(-1) # (B, H, D_k, 1)

decay_state = alpha_t * state

```

The `unsqueeze(-1)` ensures it broadcasts correctly against the `(B, H, D_k, D_v)` state.

**Which lines handle the outer products?**

```python

k_t_col = k_t.unsqueeze(-1)        # (B, H, D_k, 1)

k_t_row = k_t.unsqueeze(-2)        # (B, H, 1, D_k)

v_t_row = v_t.unsqueeze(-2)        # (B, H, 1, D_v)

```

These align the dimensions so `torch.matmul` correctly computes \(k_t k_t^T\) and \(k_t v_t^T\).

---

**How does the recurrence differ from an RNN?**

While it operates sequentially like an RNN, it uses a matrix-valued state (\(D_k \times D_v\)) and outer-product updates, which is a hallmark of linear attention. Unlike a standard RNN, it does not have a nonlinearity inside the state update, which allows for parallel scan training.

**How do you correctly implement the output projection?**

The expected test cases showed that `output_projection` has shape `(output_dim, H  D_v)`_. To apply it to _`o_concat`_ of shape _`(B, H  D_v)`, you must transpose the projection matrix:

```python

o_out = torch.matmul(o_concat, output_projection.transpose(-1, -2))

```

**How do you avoid shape mismatches?**

By carefully tracking the dimensions. The state is always `(B, H, D_k, D_v)`. The key outer product requires `(B, H, D_k, 1)` and `(B, H, 1, D_k)`. The value outer product requires `(B, H, D_k, 1)` and `(B, H, 1, D_v)`. The readout requires transposing the state to `(B, H, D_v, D_k)` and multiplying by `q_t` of shape `(B, H, D_k, 1)`.

**How can this be optimized?**

While the provided code uses a naive Python `for` loop, in production, this recurrence is typically implemented using a **parallel scan (associative scan)** algorithm. Because the state update is a linear recurrence (with a transition matrix), it can be computed in \(O(\log S)\) parallel steps on a GPU, making training much faster.