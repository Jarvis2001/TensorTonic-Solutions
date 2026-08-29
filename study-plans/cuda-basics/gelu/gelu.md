# <span style="font-size: 20px;">GELU</span>

<span style="font-size: 14px;">GELU gates each value by the Gaussian cumulative distribution, $\text{output}[i] = 0.5 \cdot \text{input}[i] \cdot (1 + \operatorname{erf}(\text{input}[i] / \sqrt{2}))$. It is an **embarrassingly parallel map**: every output depends on exactly one input at the same index, with zero communication between threads. The systems angle is the transcendental `erff`, the most expensive of the common activations, which pushes per-element arithmetic up - yet the roofline still pins the kernel to memory bandwidth at scale.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
\text{output}[i] = 0.5 \cdot \text{input}[i] \cdot \left(1 + \operatorname{erf}\!\left(\frac{\text{input}[i]}{\sqrt{2}}\right)\right)
$$

<span style="font-size: 14px;">Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory. Output element $i$ reads only input element $i$ and writes only output element $i$. Nothing is shared, reused, or reordered - the index is the only structure.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks.</span>

<span style="font-size: 14px;">The body is guarded by `if (idx < N)` because rounding the grid up leaves surplus tail threads; without the check they read and write past the buffers. There is no `__syncthreads()` and no shared state - the whole computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` and stores `output[idx]`. There is no reuse, so nothing belongs in `__shared__` memory - the `erff` result and the intermediate products live in registers for their brief lifetime and are gone. A map loads each datum exactly once.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. The constant $1/\sqrt{2}$ is a compile-time literal baked into a register, not a memory load, so it adds no global traffic.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes - one 4-byte load and one 4-byte store - and performs a scale, the transcendental `erff`, an add, and two multiplies. `erff` is the costly part; the GPU evaluates it on the **Special Function Unit (SFU)** as a polynomial-and-range-reduction sequence, more instructions than `expf` or `tanhf`. Counting roughly twenty effective FLOPs, the **arithmetic intensity** is about:</span>

$$
\frac{\sim 20 \text{ FLOP}}{8 \text{ bytes}} \approx 2.5 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That is the highest intensity of the common activations, so GELU leans furthest toward compute. Still, the ridge point of the **roofline** sits in the tens of FLOPs per byte, so at $\approx 2.5$ GELU remains below it: **memory-bound at scale**. The extra SFU work overlaps with outstanding memory transactions, and with enough warps in flight the DRAM bandwidth - not the SFU - sets the runtime. GELU is simply the activation where the SFU has the most to do while waiting on memory.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles; the SFU `erff` costs more than the other activations but still far less than a memory round trip. The GPU hides the dominant memory latency with **massive multithreading**: when a warp stalls on its load of `input`, the SM scheduler switches to another resident warp. Because GELU's per-element instruction count is higher, occupancy matters even more here - you need enough warps that the longer SFU sequences of stalled-then-resumed warps keep the issue ports busy. GELU has no data-dependent branch, so every active lane takes the same path and there is no **warp divergence**.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel computes the exact `erff` formula. Optimizations fall into two camps:</span>

<span style="font-size: 14px;">1. **Approximate the function**: the well-known tanh approximation $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ replaces `erff` with a single `tanhf` plus a few multiplies, cutting SFU work. Because the kernel is memory-bound at scale this rarely changes wall-clock time, but it reduces the chance the SFU throttles at low occupancy. The fast-math `__expf`/`__tanhf` intrinsics go further on precision-for-speed.</span>

<span style="font-size: 14px;">2. **Approach the bandwidth ceiling**: a **grid-stride loop** lets a fixed grid handle any $N$ and amortizes launch overhead, while vectorized `float4` loads move 16 bytes per instruction and apply GELU componentwise, issuing fewer, wider transactions.</span>

<span style="font-size: 14px;">Both camps sit on top of an already memory-limited pipeline. Cheaper math helps only until DRAM bandwidth becomes the wall again.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$ and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [0, 1, -1, \ldots]$, thread 0 computes $0.5 \cdot 0 \cdot (1 + \operatorname{erf}(0)) = 0$, thread 1 computes $0.5 \cdot 1 \cdot (1 + \operatorname{erf}(0.707)) \approx 0.841$, and thread 2 computes $0.5 \cdot (-1) \cdot (1 + \operatorname{erf}(-0.707)) \approx -0.159$. Each lane runs the identical `erff` sequence regardless of sign, so the warp never diverges; only the data flowing through the SFU differs.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Treating it as compute-bound.** Even at $\approx 2.5$ FLOP/byte, the most arithmetic-heavy activation here, GELU is memory-bound at scale; coalescing and occupancy set the runtime, not the `erff` cost.</span>
* <span style="font-size: 14px;">**Low occupancy exposing the SFU.** GELU's long per-element sequence is only hidden when many warps are resident; too few warps and the SFU throttles.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---