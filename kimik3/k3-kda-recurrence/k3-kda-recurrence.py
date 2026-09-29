#Chunked parallel scan(Deepseek)
import torch
import torch.nn.functional as F

def kda_recurrence(
    query, key, value, decay_logits, write_strength,
    output_gate_logits, output_projection, initial_state,
    g_min=-5.0, eps=1e-6, chunk_size=64,
):
    B, S, H, D_k = key.shape
    _, _, _, D_v = value.shape
    device, dtype = key.device, key.dtype

    # ---- Pad sequence to a multiple of chunk_size ----
    pad = (-S) % chunk_size
    if pad:
        key   = F.pad(key,   (0, 0, 0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, 0, 0, pad))
        query = F.pad(query, (0, 0, 0, 0, 0, pad))
        if decay_logits.ndim == 4:
            decay_logits = F.pad(decay_logits, (0, 0, 0, 0, 0, pad))
        else:
            decay_logits = F.pad(decay_logits, (0, 0, 0, 0, 0, pad))
        if write_strength.ndim == 4:
            write_strength = F.pad(write_strength, (0, 0, 0, 0, 0, pad))
        else:
            write_strength = F.pad(write_strength, (0, 0, 0, 0, 0, pad))
        if output_gate_logits.ndim == 4:
            output_gate_logits = F.pad(output_gate_logits, (0, 0, 0, 0, 0, pad))
        else:
            output_gate_logits = F.pad(output_gate_logits, (0, 0, 0, 0, 0, pad))

    S_pad = S + pad
    N = S_pad // chunk_size

    # ---- Broadcast-ready alpha / beta / gate ----
    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))
    if decay_logits.ndim == 3:
        alpha = alpha.unsqueeze(-1)
    alpha = alpha.unsqueeze(-1)              # (B, S, H, D_k, 1) or (B, S, H, 1, 1)

    if write_strength.ndim == 3:
        write_strength = write_strength.unsqueeze(-1)
    beta = write_strength.unsqueeze(-1)      # (B, S, H,1,1)

    if output_gate_logits.ndim == 3:
        output_gate_logits = output_gate_logits.unsqueeze(-1)

    # ---- Whole-sequence outer products (one batched matmul each) ----
    k_col = key.unsqueeze(-1)                # (B, S_pad, H, D_k, 1)
    k_row = key.unsqueeze(-2)                # (B, S_pad, H, 1, D_k)
    v_row = value.unsqueeze(-2)              # (B, S_pad, H, 1, D_v)
    kk = k_col * k_row                       # (B, S_pad, H, D_k, D_k)
    kv = k_col * v_row                       # (B, S_pad, H, D_k, D_v)

    I = torch.eye(D_k, device=device, dtype=dtype)

    # =========================================================
    # Phase 1: within-chunk prefix products P_i and local states L_i
    #          shape (C, B, N, H, D_k, D_k) / (C, B, N, H, D_k, D_v)
    # =========================================================
    Ps = []
    Ls = []
    P = I.expand(B, N, H, D_k, D_k).clone()
    L = torch.zeros(B, N, H, D_k, D_v, device=device, dtype=dtype)

    for i in range(chunk_size):
        sl = slice(i, S_pad, chunk_size)     # position i inside every chunk
        kk_i    = kk[:, sl]
        kv_i    = kv[:, sl]
        alpha_i = alpha[:, sl]
        beta_i  = beta[:, sl]

        # A_i = (I - beta * k k^T) @ Diag(alpha)
        A_i = (I - beta_i * kk_i) * alpha_i.transpose(-1, -2)
        B_i = beta_i * kv_i

        P = A_i @ P                          # prefix transition
        L = A_i @ L + B_i                    # local state

        Ps.append(P)
        Ls.append(L)

    Ps = torch.stack(Ps, dim=0)              # (C, B, N, H, D_k, D_k)
    Ls = torch.stack(Ls, dim=0)              # (C, B, N, H, D_k, D_v)

    # =========================================================
    # Phase 2: cross-chunk scan (N sequential steps)
    # =========================================================
    P_last = Ps[-1]                          # (B, N, H, D_k, D_k)
    L_last = Ls[-1]                          # (B, N, H, D_k, D_v)

    S_starts = [initial_state]
    for n in range(N - 1):
        S_starts.append(P_last[:, n] @ S_starts[n] + L_last[:, n])
    S_start = torch.stack(S_starts, dim=1)   # (B, N, H, D_k, D_v)

    # =========================================================
    # Phase 3: reconstruct every S_t and compute outputs in parallel
    # =========================================================
    S_all = Ps @ S_start.unsqueeze(0) + Ls   # (C, B, N, H, D_k, D_v)
    S_all = S_all.permute(1, 0, 2, 3, 4, 5).reshape(B, S_pad, H, D_k, D_v)

    q_col = query.unsqueeze(-1)              # (B, S_pad, H, D_k, 1)
    o_tilde = (S_all.transpose(-2, -1) @ q_col).squeeze(-1)   # (B, S_pad, H, D_v)

    rms = torch.sqrt(o_tilde.pow(2).mean(-1, keepdim=True) + eps)
    o_gated = (o_tilde / rms) * torch.sigmoid(output_gate_logits)

    o_concat = o_gated.reshape(B, S_pad, H * D_v)
    outputs  = o_concat @ output_projection.T                  # (B, S_pad, out_dim)

    S_final = S_all[:, S - 1]

    if pad:
        outputs = outputs[:, :S]

    return {"outputs": outputs, "final_state": S_final}