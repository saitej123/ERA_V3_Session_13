import torch
import math
from typing import Optional, Tuple

def create_causal_mask(
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a causal attention mask for self-attention."""
    mask = torch.ones((seq_length, seq_length), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def create_attention_mask(
    input_ids: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Create attention mask from input ids and optional padding mask."""
    batch_size, seq_length = input_ids.shape
    device = input_ids.device

    if causal:
        mask = create_causal_mask(seq_length, device)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            mask = mask | padding_mask
    else:
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = torch.zeros((batch_size, 1, 1, seq_length), device=device)

    return mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention with mask."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def get_slope_matrix(
    n_heads: int,
    seq_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Get slope matrix for ALiBi positional embeddings."""
    def get_slopes(n: int) -> torch.Tensor:
        def get_slopes_power_of_2(n: int) -> torch.Tensor:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)], device=device, dtype=dtype)

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return torch.cat([
                get_slopes_power_of_2(closest_power_of_2),
                get_slopes(n - closest_power_of_2),
            ])

    slopes = get_slopes(n_heads)
    # Create a position matrix [seq_length, seq_length]
    positions = torch.arange(seq_length, device=device, dtype=dtype).unsqueeze(0)
    # Create a distance matrix [seq_length, seq_length]
    distances = positions - positions.T
    # Create the final slope matrix [n_heads, seq_length, seq_length]
    slope_matrix = slopes.unsqueeze(-1).unsqueeze(-1) * distances.unsqueeze(0)
    
    return slope_matrix


def apply_rotary_embeddings(
    x: torch.Tensor,
    sincos: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensors."""
    sin, cos = sincos.chunk(2, dim=-1)
    sin = sin[None, offset : x.shape[1] + offset, None, :]
    cos = cos[None, offset : x.shape[1] + offset, None, :]
    
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1) 