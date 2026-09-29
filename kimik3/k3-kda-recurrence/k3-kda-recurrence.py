# Precompute outside the loop
import torch

def kda_recurrence(
    query, key, value, decay_logits, write_strength,
    output_gate_logits, output_projection, initial_state,
    g_min=-5.0, eps=1e-6,
):
    B, S, H, D_k = key.shape
    _, _, _, D_v = value.shape
    device, dtype = key.device, key.dtype

    # ---- Hoist all shape-only work out of the loop ----
    k_col = key.unsqueeze(-1)       # (B,S,H,D_k,1)
    k_row = key.unsqueeze(-2)       # (B,S,H,1,D_k)
    v_row = value.unsqueeze(-2)     # (B,S,H,1,D_v)
    kk = k_col * k_row              # (B,S,H,D_k,D_k)
    kv = k_col * v_row              # (B,S,H,D_k,D_v)

    # alpha broadcast-ready: (B,S,H,D_k,1) or (B,S,H,1,1)
    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))
    if decay_logits.ndim == 3:
        alpha = alpha.unsqueeze(-1)
    alpha = alpha.unsqueeze(-1)

    # beta: (B,S,H,1,1)
    if write_strength.ndim == 3:
        write_strength = write_strength.unsqueeze(-1)
    beta = write_strength.unsqueeze(-1)

    # gate: (B,S,H,D_v) or (B,S,H,1)
    if output_gate_logits.ndim == 3:
        output_gate_logits = output_gate_logits.unsqueeze(-1)

    I = torch.eye(D_k, device=device, dtype=dtype)
    state = initial_state.clone()
    outputs_list = []

    for t in range(S):
        kk_t   = kk[:, t]                 # (B,H,D_k,D_k)
        kv_t   = kv[:, t]                 # (B,H,D_k,D_v)
        alpha_t = alpha[:, t]             # (B,H,D_k,1) or (B,H,1,1)
        beta_t  = beta[:, t]              # (B,H,1,1)
        gate_t  = output_gate_logits[:, t]

        # (I - beta * k k^T) @ Diag(alpha) @ state + beta * k v^T
        erase = I - beta_t * kk_t                                # (B,H,D_k,D_k)
        A_t = erase * alpha_t.transpose(-1, -2)                  # right-mul by Diag(alpha)
        state = A_t @ state + beta_t * kv_t                      # (B,H,D_k,D_v)

        q_t_col = query[:, t].unsqueeze(-1)
        o_tilde = (state.transpose(-2, -1) @ q_t_col).squeeze(-1)
        rms = torch.sqrt(o_tilde.pow(2).mean(-1, keepdim=True) + eps)
        o_gated = (o_tilde / rms) * torch.sigmoid(gate_t)
        outputs_list.append(o_gated.reshape(B, H * D_v) @ output_projection.T)

    return {
        "outputs": torch.stack(outputs_list, dim=1),
        "final_state": state,
    }