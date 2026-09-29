# Kimi Delta Attention (KDA) — Notes

## What

### What is this code?

This is a PyTorch implementation of the **Kimi Delta Attention (KDA)** recurrence. It is a linear attention mechanism that processes a sequence token-by-token, maintaining a fixed-size memory matrix (the *state*) rather than computing a full $S \times S$ attention matrix.

### What does the function do?

It takes a sequence of queries ($Q$), keys ($K$), values ($V$), decay logits, write strengths, output gates, and an initial state. It iterates through the sequence, updating the state matrix $S_t$ at each step, and produces a sequence of outputs and the final state.

### Core mathematical formula

$$
S_t = (I - \beta_t\, k_t k_t^{T})\,\mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t\, k_t v_t^{T}
$$

$$
\tilde{o}_t = S_t^{T} q_t
$$

### Key components

- **State ($S_t$):** a matrix of shape $(B, H, D_k, D_v)$ acting as associative memory.
- **Decay ($\alpha_t$):** channel-wise retention value derived from `decay_logits`.
- **Write strength ($\beta_t$):** a scalar per head controlling how much new information is written.
- **Erase term:** $(I - \beta_t k_t k_t^{T})$, which removes old associations.
- **Write term:** $\beta_t k_t v_t^{T}$, which adds new associations.

---

## Why

### Why use KDA instead of standard softmax attention?

Standard attention has $O(S^2)$ complexity. KDA has $O(S)$ complexity, making it highly efficient for long sequences while still allowing dynamic memory updates.

### Why is the decay channel-wise?

A single scalar decay per head would erase or retain all features uniformly. Channel-wise decay ($\alpha_t \in \mathbb{R}^{D_k}$) allows the model to selectively remember some features while forgetting others within the same head.

### Why is there an "erase" term?

Without the erase term, the state would just accumulate $k v^{T}$ indefinitely. If the same key appears again with a new value, the old value would still be there, causing interference. The erase term implements the **delta rule**, updating the memory by *replacing* the old association rather than adding to it.

### Why "read after writing"?

The theory states: The current output is read from $S_t$, not from $S_{t-1}$. If you read before writing, the current token cannot attend to itself, causing a one-step lag and preventing the first token from using its own information.

### Why RMS-normalize the output?

RMS normalization stabilizes the scale of the readout vector across different heads and sequence steps, preventing exploding activations before the output gate is applied.

### Why does the code clone the initial state?

To avoid mutating the supplied tensor. The problem explicitly lists "mutating the initial state" as a common mistake to avoid.

---

## Who

### Who created this concept?

Kimi Delta Attention was introduced by the Kimi team (**Moonshot AI**) as part of their work on efficient linear attention architectures, building upon ideas like DeltaNet and Mamba.

---

## Where

### Where is this used in a model architecture?

It replaces the standard multi-head self-attention layer in a Transformer block. It is typically sandwiched between a normalization layer and a feed-forward network (MLP).

### Where in the code does the state update happen?

Inside the `for t in range(S):` loop. Specifically:

```python
erased_state = torch.matmul(erase_term, decay_state)   # (I - beta k k^T) Diag(alpha) S_{t-1}
write_term   = beta_t * torch.matmul(k_t_col, v_t_row) # beta k v^T
state        = erased_state + write_term
```

### Where does the output projection happen?

At the end of the loop:

```python
o_out = torch.matmul(o_concat, output_projection)  # or output_projection.T, depending on layout
```

---

## When

### When does the state update occur?

Sequentially, at every time step $t$ from $0$ to $S-1$. The state at step $t$ depends entirely on the state at step $t-1$.

### When is the output generated?

Immediately after the state update for the current token $t$.

### When would you use this in practice?

During **autoregressive inference** (generating text token by token), where the state can be cached and updated in $O(1)$ time per token, rather than recomputing attention over the entire context window at every step.

---

## Which

### Which dimensions correspond to what?

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $S$ | Sequence length |
| $H$ | Number of attention heads |
| $D_k$ | Key/query dimension (width of keys) |
| $D_v$ | Value dimension (width of values) |

### Which lines handle the channel-wise decay?

```python
alpha = torch.exp(g_min * torch.sigmoid(decay_logits))  # (B, S, H, D_k)
alpha_t = alpha[:, t].unsqueeze(-1)                     # (B, H, D_k, 1)
decay_state = alpha_t * state                           # broadcasts vs (B, H, D_k, D_v)
```

The `unsqueeze(-1)` ensures the retention vector broadcasts correctly against the $(B, H, D_k, D_v)$ state — one decay value per key channel.

### Which lines handle the outer products?

```python
k_t_col = k_t.unsqueeze(-1)   # (B, H, D_k, 1)
k_t_row = k_t.unsqueeze(-2)   # (B, H, 1, D_k)
v_t_row = v_t.unsqueeze(-2)   # (B, H, 1, D_v)
```

These align the dimensions so `torch.matmul` computes $k_t k_t^{\top}$ (shape ($D_k$, $D_k$)) and $k_t v_t^{\top}$ (shape ($D_k$, $D_v$)).

---

## How

### How does the recurrence differ from an RNN?

While it operates sequentially like an RNN, it uses a **matrix-valued state** ($D_k \times D_v$) with outer-product updates — a hallmark of linear attention. Unlike a standard RNN, it has no nonlinearity inside the state update, which is exactly what allows parallel-scan training (see the optimizations below).

### How do you avoid shape mismatches?

By carefully tracking dimensions:

- The state is always $(B, H, D_k, D_v)$.
- Key outer product: $(B, H, D_k, 1) \times (B, H, 1, D_k) \rightarrow (B, H, D_k, D_k)$.
- Value outer product: $(B, H, D_k, 1) \times (B, H, 1, D_v) \rightarrow (B, H, D_k, D_v)$.
- Readout: transpose state to $(B, H, D_v, D_k)$, multiply by $q_t$ of shape $(B, H, D_k, 1)$.

### How can this be optimized?

While the naive implementation uses a Python `for` loop, in production this recurrence is typically implemented with a **chunked parallel scan**. Because the state update is a linear (affine) recurrence, it can be computed in $O(\log S)$ parallel depth on a GPU, making training much faster. See the next section.

---

# Optimizations

This section covers **both** optimizations together:

- **Opt 1 — Hoisted Precomputation:** move shape-only work out of the loop.
- **Opt 2 — Chunked Parallel Scan:** convert the sequential recurrence into a parallel scan.

## What

### What is Optimization 1?

A loop-restructuring pass. Every tensor that does not depend on the running state $S_t$ — the outer products $k k^{T}$ and $k v^{T}$, the decay $\alpha$, the write strength $\beta$, the identity matrix $I$ — is computed **once, for the whole sequence**, before the loop begins. Inside the loop, only the state-dependent operations remain.

### What is Optimization 2?

An algorithmic reformulation. The KDA recurrence is a **linear (affine) recurrence**:

$$
S_t = A_t S_{t-1} + B_t
$$

Linear recurrences are **associative under composition**, meaning $(A_2, B_2) \circ (A_1, B_1) = (A_2 A_1,\; A_2 B_1 + B_2)$. This lets you replace $S$ sequential steps with a **chunked scan**: process $C$ tokens in parallel per block, then combine block-boundary states with only $S/C$ sequential steps.

### What does "hoisting" mean?

Moving an invariant computation out of a loop — in compilers, *loop-invariant code motion*. Here the invariants are: $k k^{T}$, $k v^{T}$, $\alpha$, $\beta$, $I$, and all unsqueeze/expand reshapes — none depend on $t$ or on $S_t$.

### What does "parallel scan" mean?

A parallel prefix algorithm. For an associative operator $\circ$, an array $[x_0, x_1, \dots, x_{N-1}]$ can be reduced to the prefix $[x_0,\; x_0 \circ x_1,\; x_0 \circ x_1 \circ x_2,\; \dots]$ in $O(\log N)$ depth (Blelloch scan), or in $O(N)$ work with sequential composition inside chunks plus sequential composition of chunk summaries.

### What is the chunk size $C$?

The number of tokens processed per parallel block. Typical values: $C \in \{32, 64, 128\}$. Larger $C$ → more parallelism but more memory; smaller $C$ → less parallelism but less memory. The sweet spot depends on $D_k$, $D_v$, $H$, and available GPU memory.

### What are Phase 1, Phase 2, Phase 3?

- **Phase 1 (intra-chunk):** for each chunk, compute prefix products $P_i = A_i \cdots A_0$ and local states $L_i = A_i L_{i-1} + B_i$. Fully parallel over chunks and positions within a chunk.
- **Phase 2 (inter-chunk):** a short sequential scan over $N = S/C$ chunk-boundary states: $S_{\text{start}}[n+1] = P_{\text{last}}[n]\, S_{\text{start}}[n] + L_{\text{last}}[n]$.
- **Phase 3 (reconstruction):** every token state is $S_{s+i} = P_i\, S_{\text{start}}[\text{chunk}] + L_i$. Readout, gating, and projection all happen in one batched pass.

### What changed in memory layout?

Optimization 1 keeps the same tensors but reduces per-step allocations. Optimization 2 introduces intermediate tensors of shape $(C, B, N, H, D_k, D_k)$ for the $P$'s and $(C, B, N, H, D_k, D_v)$ for the $L$'s — the parallel prefix tensors.

### What is the "WY representation" mentioned in Opt 3?

$A_t = I - \beta_t k_t k_t^{T}$. The outer-product form is cheap to construct but expensive to multiply. The WY form stores $A = I - W K^{T}$ where $W$ and $K$ are $(D_k \times r)$ with $r$ small, so matrix–vector products cost $O(D_k r)$ instead of $O(D_k^2)$. This is the trick the FLA kernels use.

## Why

### Why does the original code run slowly?

1. **Serial depth $= S$.** Modern GPUs want thousands of independent operations in flight. A `for` loop over $S = 4096$ gives one token's worth of work at a time.
2. **Kernel-launch overhead dominates.** Each `torch.matmul`, `unsqueeze`, `sigmoid`, `mean` on tiny tensors is a separate CUDA kernel. For $S = 4096$ with ~15 ops/token, that is ~60,000 launches; at ~5 µs each, ~0.3 s of pure launch overhead per layer.
3. **Repeated invariant work.** `torch.eye`, unsqueezes, and outer products are recomputed $S$ times but never change.

### Why is the recurrence associative?

Because $S_t$ is a **linear** function of $S_{t-1}$. Linear maps compose: if $S_1 = A_1 S_0 + B_1$ and $S_2 = A_2 S_1 + B_2$, then

$$
S_2 = A_2 A_1 S_0 + (A_2 B_1 + B_2),
$$

which has the same $(A, B)$ structure. That closure property is exactly what makes the scan possible.

### Why chunk at all instead of a full parallel scan?

A full $\log S$ scan of $D_k \times D_k$ matrices is memory-prohibitive (each $A_t$ costs $H \cdot D_k^2$ floats) and has poor cache behavior. Chunking keeps sequential depth at $S/C$ while all large multiplications inside chunks run fully parallel — a good balance.

### Why precompute $k k^{T}$ and $k v^{T}$ rather than leave them inside the loop?

They do not depend on state. Computing them once as a single batched $(B, S, H, D_k, D_k)$ matmul is **one** kernel launch instead of $S$ launches. This alone often gives 1.5–2× on GPU.

### Why is Phase 2 unavoidable as a sequential step?

Because $S_{\text{start}}[n+1]$ depends on $S_{\text{start}}[n]$. You could make it parallel with a Blelloch tree scan, but $N = S/C$ is already small (e.g., 64 for $S = 4096$, $C = 64$), so the sequential cost is negligible and the tree scan's extra memory traffic usually is not worth it.

### Why does the chunked version numerically match the sequential version?

Because matrix multiplication is associative (in exact arithmetic). Floating-point rounding differs, but only by a few ULP per operation, so final states differ by $O(\varepsilon)$ — well within tolerance for training or inference.

### Why pad the sequence?

Because the chunked algorithm assumes $S \bmod C = 0$. Padding to the nearest multiple with zeros makes $N = S/C$ an integer. A padded step has $k = 0$, $v = 0$, hence $A_t = \mathrm{Diag}(\alpha_t)$ and $B_t = 0$ — the state merely decays, so padding is safe. Trailing outputs are sliced off at the end.

### Why is Opt 3 (Triton / WY form) even faster?

1. **Eliminates the $(B, S, H, D_k, D_v)$ intermediate tensor $S_{\text{all}}$.** Materializing all intermediate states is memory-bound: for $S = 4096$, $D_k = D_v = 128$, $H = 8$, $B = 1$ in fp32, that is ~2 GB.
2. **Replaces $D_k \times D_k$ matrix multiplies with $D_k \times r$ low-rank updates** (WY form). For $r \ll D_k$, this reduces arithmetic by 10–30×.

## Who

### Who benefits from these optimizations?

- **Training engineers** fitting KDA into LLM pretraining pipelines — Opt 1+2 give 5–15× throughput over the naive loop.
- **Inference engineers** deploying long-context models — the constant-memory state and $O(1)$ per-token update survive, but prompt prefill benefits most.
- **Kernel developers** porting to Triton/CUDA — Opt 2 is the reference structure they start from.

### Who wrote the original algorithm?

The Kimi/Moonshot AI team published KDA as a linear attention variant; the chunked-scan structure is borrowed from the earlier **DeltaNet** work (Yang et al., 2024) and the **Mamba-2 / SSD** chunking scheme (Dao & Gu, 2024).

### Who maintains production implementations?

The **Flash-Linear-Attention (FLA)** library from the fla-org GitHub project. Its `chunk_delta_rule` and `chunk_kda` kernels are what most downstream code imports.

## Where

### Where is the biggest time saving?

In the serial depth: original $O(S)$ → Opt 1 still $O(S)$ but ~3× fewer kernels per step → Opt 2 $O(S/C)$.

### Where does each optimization live in the code?

| Code region | Opt 1 | Opt 2 |
|---|---|---|
| Before the loop | $k k^{T}$, $k v^{T}$, $\alpha$, $\beta$, $I$ | same, plus padding |
| Inside the loop | state update + readout | — |
| Phase 1 loop (length $C$) | — | intra-chunk scan |
| Phase 2 loop (length $N$) | — | inter-chunk scan |
| After Phase 2 | — | reconstruct all $S_t$, readout, project |

### Where does memory grow in Opt 2?

Two places: $P$'s of shape $(C, B, N, H, D_k, D_k)$ and $L$'s of shape $(C, B, N, H, D_k, D_v)$. Total additional memory:

$$
O\big(C \cdot B \cdot H \cdot D_k \cdot (D_k + D_v)\big)
$$

For typical values ($C = 64$, $B = 1$, $H = 8$, $D_k = D_v = 128$), that is roughly $64 \cdot 8 \cdot 128 \cdot 256 \cdot 4$ bytes ≈ 8 MB — small.

### Where does the "read after writing" rule live?

Still in Phase 3, after $S_{\text{all}}$ is reconstructed. The rule is a property of the state semantics, not the execution order — chunking does not affect it.

### Where would you put a Triton kernel?

Over Phase 1 (the $C$-step intra-chunk loop) and Phase 3 (reconstruction + readout). Phase 2 is small enough to leave in PyTorch.

## When

### When should you use Opt 1 alone?

When $S$ is small ($S < 256$) or when debugging and you want the code shape close to the original. The speedup is real but limited.

### When should you use Opt 2?

When $S \geq 512$ and you are on a GPU. The larger $S$, the bigger the win (asymptotically $O(S/C)$ vs. $O(S)$).

### When is chunking not helpful?

- **Very short sequences** ($S < C$) — the chunk scaffold is pure overhead.
- **Autoregressive decoding with $S = 1$** — you are already at the sequential limit; the state-cached step handles a single token.
- **Memory-constrained devices** where the extra $P$ / $L$ tensors do not fit.

### When does numerical error become a concern?

After ~1000 chunks of fp16 accumulation. In practice, KDA layers are followed by LayerNorm, which clamps the effect. For fp32 there is no concern at any practical $S$.

### When is the recurrence "read after writing"?

At every step $t$: the state is updated first ($S_t$), then $o_t = S_t^{T} q_t$. Chunking preserves this — the reconstruction formula $S_{s+i} = P_i S_{\text{start}} + L_i$ includes $A_t$ and $B_t$ for the current token, so $S_{s+i}$ is already the post-update state.

## Which

### Which lines are "loop-invariant" in the original code?

```python
I = torch.eye(D_k, ...)                       # invariant
k_t_col = k_t.unsqueeze(-1)                   # per-t shape, same op every t
k_t_row = k_t.unsqueeze(-2)
v_t_row = v_t.unsqueeze(-2)
torch.matmul(k_t_col, k_t_row)                # k_t k_t^T — no state dependence
torch.matmul(k_t_col, v_t_row)                # k_t v_t^T — no state dependence
alpha = torch.exp(g_min * sigmoid(...))       # depends only on decay_logits
```

All of these move out of the loop in Opt 1.

### Which operations are state-dependent and must stay in the loop?

```python
decay_state = alpha_t * state
erased_state = torch.matmul(erase_term, decay_state)
state = erased_state + write_term
```

Plus the readout — in the sequential version. In Opt 2, even these get restructured.

### Which dimension is chunked?

The **sequence dimension** $S$. Everything else ($B$, $H$, $D_k$, $D_v$) stays intact.

### Which `torch.matmul` dominates compute in Opt 2?

`P @ S_start` in Phase 3: $(C, B, N, H, D_k, D_k) \times (1, B, N, H, D_k, D_v)$. That is $O(S \cdot H \cdot D_k^2 \cdot D_v)$ FLOPs — the same total as the original loop, but done in one batched call instead of $S$ small calls.

### Which optimization gives more speedup, Opt 1 or Opt 2?

- **Small $S$:** Opt 1.
- **Large $S$:** Opt 2, and the gap grows with $S$.
- **Combined:** they are multiplicative — Opt 1 removes per-step overhead, Opt 2 kills serial depth.

### Which parameters are constants and which are tunable?

- **Constants:** $g_{\min}$, $\varepsilon$ (eps), $D_k$, $D_v$, $H$.
- **Tunable:** chunk size $C$. Tune by sweeping $\{16, 32, 64, 128, 256\}$ and measuring wall-clock on your hardware.

## How

### How does hoisting work, mechanically?

Compute `kk = torch.matmul(k_col, k_row)` where `k_col = key.unsqueeze(-1)` and `k_row = key.unsqueeze(-2)`. Because `key` is a whole-sequence tensor of shape $(B, S, H, D_k)$, the unsqueezes and matmul are batched over $S$. Result: `kk` of shape $(B, S, H, D_k, D_k)$ in **one kernel launch**. Inside the loop, `kk_t = kk[:, t]` is a view.

### How does the chunked scan work, step by step?

Let the chunk size be $C$ and $N = S/C$. Write

$$
A_t = (I - \beta_t k_t k_t^{T})\,\mathrm{Diag}(\alpha_t), \qquad B_t = \beta_t k_t v_t^{T}.
$$

**Phase 1 — intra-chunk (for each chunk independently):**

- Initialize $P = I$, $L = 0$.
- For $i = 0, \dots, C-1$:
  - $P \leftarrow A_{s+i}\, P$
  - $L \leftarrow A_{s+i}\, L + B_{s+i}$
  - Store $P$ and $L$.
- After the loop, $P_i$ is the product of the chunk's transitions up to $i$, and $L_i$ is the local state assuming the chunk started from $0$.

**Phase 2 — combine chunk ends:**

- $S_{\text{start}}[0] = S_{\text{initial}}$
- For $n = 0, \dots, N-2$: $S_{\text{start}}[n+1] = P_{\text{last}}[n]\, S_{\text{start}}[n] + L_{\text{last}}[n]$.
- Only $N$ matmuls of size $D_k \times D_k \times D_v$ — cheap.

**Phase 3 — reconstruct every state:**

- $S_{s+i} = P_i\, S_{\text{start}}[\text{chunk}] + L_i$, batched over $i$ and chunk.
- Then one batched matmul for the readout, one batched RMS norm, one gated multiply, one output projection.

### How does padding work?

```python
# F.pad pads the sequence dimension with zeros:
key = F.pad(key, (0, 0, 0, 0, 0, pad))
```

Zero keys → $k k^{T} = 0$, $k v^{T} = 0$ → $A = \mathrm{Diag}(\alpha)$, $B = 0$. A padded step decays the state but writes nothing. At the end, `outputs = outputs[:, :S]` drops the padded outputs, and `final_state` is taken at index $S-1$, not $S_{\text{pad}} - 1$.

### How do you verify correctness?

Run the optimized version on the same inputs as the naive loop and compare:

```python
ref  = kda_recurrence_naive(...)
fast = kda_recurrence_chunked(...)

assert torch.allclose(ref["outputs"],     fast["outputs"],     atol=1e-5)
assert torch.allclose(ref["final_state"], fast["final_state"], atol=1e-5)
```

The 1-D example from the theory is a good smoke test: $\text{prev} = 2$, $\alpha = 0.5$, $k = 1$, $v = 4$, $\beta = 0.25$ → $S_{\text{new}} = 1.75$.

### How do you choose $C$?

Sweep. Start at $C = 64$. Measure:

- **Wall-clock time** (end-to-end).
- **Peak GPU memory** (`torch.cuda.max_memory_allocated()`).

Rule of thumb: increase $C$ until memory pressure or L2-cache misses slow you down; decrease it if the machine is memory-starved or $S/C$ leaves an awkward remainder.

### How do you go further than these two optimizations?

1. **Unroll Phase 1 into a Blelloch scan** over the chunk dimension to reach $O(\log C)$ depth. Rarely worth it in PyTorch; worth it in Triton.
2. **Fuse Phase 1 and Phase 3 into one Triton kernel.** Never materialize $S_{\text{all}}$. Use the identity $(P_i S_{\text{start}} + L_i)^{T} q = S_{\text{start}}^{T} (P_i^{T} q) + L_i^{T} q$.
3. **Use the WY representation** to avoid materializing $D_k \times D_k$ matrices: store $A_t = I - W_t K_t^{T}$ with $W_t = \beta_t k_t$ and $K_t = k_t$, and multiply against vectors via two $D_k$-sized matvecs.
4. **Mixed precision.** Keep the state in fp32, $k k^{\top}$ / $k v^{\top}$ in
bf16. Halves memory for the $P$ / $L$ / $S_{\text{all}}$ tensors.
5. **Reuse the state buffer across layers/calls.** The final state of one layer is the initial state of the next; avoid clones when the caller does not need the original.

### How does this compare to FlashAttention?

Different complexity class. FlashAttention computes **exact softmax attention** in $O(S^2)$ FLOPs with $O(S)$ memory per tile. KDA computes a **linear-attention approximation** in $O(S)$ FLOPs with $O(1)$ state memory. The chunked scan here is the linear-attention analogue of FlashAttention tiling — same idea (break into blocks), different math (associative scan vs. online softmax).

---

## Quick reference — side by side

| Aspect | Original | Opt 1 | Opt 2 (chunked) | Opt 3 (Triton/WY) |
|---|---|---|---|---|
| Serial depth | $S$ | $S$ | $S/C$ | $S/C$ |
| Memory per layer | $O(1)$ state | $O(1)$ state | $+\ O(S H (D_k^2 + D_k D_v)/C)$ | $O(1)$ state |
| Implementation | Difficult | Difficult | Difficult | Difficult |
| When to use | debugging | small $S$, prototyping | large $S$, PyTorch-only | production training/inference |
