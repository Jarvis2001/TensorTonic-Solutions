import torch

def kda_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_logits: torch.Tensor,
    write_strength: torch.Tensor,
    output_gate_logits: torch.Tensor,
    output_projection: torch.Tensor,
    initial_state: torch.Tensor,
    g_min: float = -5.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """
    KDA recurrence.
    
    Shapes:
        query:              (B, S, H, D_k)
        key:                (B, S, H, D_k)
        value:              (B, S, H, D_v)
        decay_logits:       (B, S, H, D_k) or (B, S, H, 1) or (B, S, H)
        write_strength:     (B, S, H, 1) or (B, S, H)
        output_gate_logits: (B, S, H, D_v) or (B, S, H, 1) or (B, S, H)
        output_projection:  (output_dim, H * D_v)
        initial_state:      (B, H, D_k, D_v)
        
    Returns:
        dict with:
            outputs:     (B, S, output_dim)
            final_state: (B, H, D_k, D_v)
    """
    B, S, H, D_k = key.shape
    _, _, _, D_v = value.shape

    # 1. Convert every decay logit into a channel-wise retention value
    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(-1)  # (B, S, H, 1)

    # Ensure write strength is shaped correctly for broadcasting
    if write_strength.ndim == 3:
        write_strength = write_strength.unsqueeze(-1) # (B, S, H, 1)

    # 2. Start from the supplied initial state (clone to avoid mutation)
    state = initial_state.clone()  # (B, H, D_k, D_v)
    
    # Identity matrix for the erase term
    I = torch.eye(D_k, dtype=key.dtype, device=key.device)

    outputs_list = []

    # 3. Iterate over sequence positions
    for t in range(S):
        q_t = query[:, t]                  # (B, H, D_k)
        k_t = key[:, t]                    # (B, H, D_k)
        v_t = value[:, t]                  # (B, H, D_v)

        # Shape beta_t to (B, H, 1, 1) for scalar-per-head broadcasting
        beta_t = write_strength[:, t]      # (B, H, 1)
        beta_t = beta_t.unsqueeze(-1)      # (B, H, 1, 1)

        # Shape alpha_t to (B, H, D_k, 1) for channel-wise broadcasting
        alpha_t = alpha[:, t]              # (B, H, D_k) or (B, H, 1)
        alpha_t = alpha_t.unsqueeze(-1)    # (B, H, D_k, 1) or (B, H, 1, 1)

        gate_t = output_gate_logits[:, t]  # (B, H, D_v) or (B, H, 1) or (B, H)
        if gate_t.ndim == 2:
            gate_t = gate_t.unsqueeze(-1)  # (B, H, 1)

        # --- State Update (S_t) ---
        # Prepare outer products
        k_t_col = k_t.unsqueeze(-1)        # (B, H, D_k, 1)
        k_t_row = k_t.unsqueeze(-2)        # (B, H, 1, D_k)
        v_t_row = v_t.unsqueeze(-2)        # (B, H, 1, D_v)

        # Erase term: (I - beta_t * k_t k_t^T)
        erase_term = I - beta_t * torch.matmul(k_t_col, k_t_row)

        # Decay old state: Diag(alpha_t) * S_{t-1}
        decay_state = alpha_t * state

        # Apply erase: (I - beta_t k_t k_t^T) @ decay_state
        write_term = torch.matmul(erase_term, decay_state)

        # Write new info: beta_t * k_t v_t^T
        write = beta_t * torch.matmul(k_t_col, v_t_row)

        # New state
        state = write_term + write  # (B, H, D_k, D_v)

        # --- Read After Writing (Output) ---
        q_t_col = q_t.unsqueeze(-1)        # (B, H, D_k, 1)
        
        # S_t^T @ q_t
        state_T = state.transpose(-2, -1)  # (B, H, D_v, D_k)
        o_tilde_t = torch.matmul(state_T, q_t_col).squeeze(-1)  # (B, H, D_v)

        # RMS Normalization (independently per head)
        rms = torch.sqrt(torch.mean(o_tilde_t ** 2, dim=-1, keepdim=True) + eps)
        o_norm = o_tilde_t / rms

        # Output Gating
        o_gated = o_norm * torch.sigmoid(gate_t)  # (B, H, D_v)

        # Join heads
        o_concat = o_gated.reshape(B, H * D_v)

        # Apply output projection (transpose to match (H*D_v, output_dim))
        o_out = torch.matmul(o_concat, output_projection.transpose(-1, -2))

        outputs_list.append(o_out)

    # Stack outputs along sequence dimension
    outputs = torch.stack(outputs_list, dim=1)  # (B, S, output_dim)

    return {
        "outputs": outputs,
        "final_state": state,
    }