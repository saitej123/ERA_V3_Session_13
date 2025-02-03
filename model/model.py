import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn import flash_attn_func
from rotary_embedding_torch import RotaryEmbedding


@dataclass
class SmolLM2Config:
    name: str = "SmolLM2-135"
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    gradient_checkpointing: bool = True


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # Handle input dimensions
        orig_dtype = x.dtype
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dimension to be {self.dim}, got {x.size(-1)}")
        
        # Cast to float32 for better numerical stability
        x = x.to(torch.float32)
        
        # Check for NaN or inf values
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input tensor contains NaN or inf values")
            
        # Compute statistics along last dimension only
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Add eps inside sqrt for stability
        inv_std = 1 / torch.sqrt(var + self.eps)
        
        # Normalize
        x_norm = (x - mean) * inv_std
        
        # Scale and shift
        x = x_norm * self.weight + self.bias
        
        # Return to original dtype
        return x.to(orig_dtype)


class MLP(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.n_heads
        self.dropout = config.attention_dropout
        self.use_flash_attention = config.use_flash_attention

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            query_states = self.rotary_emb.rotate_queries_or_keys(query_states)
            key_states = self.rotary_emb.rotate_queries_or_keys(key_states)

        if self.use_flash_attention and flash_attn_func is not None:
            attn_output = flash_attn_func(
                query_states, key_states, value_states,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            attn_weights = attn_weights / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.ln1 = LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Shape checks
        batch_size, seq_length, hidden_dim = hidden_states.size()
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(f"Expected hidden_dim {self.config.hidden_dim}, got {hidden_dim}")
            
        # First Layer Norm
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        # Self Attention
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs + residual
        
        # Second Layer Norm and MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        
        return hidden_states


class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        
        # Initialize embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Initialize transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Initialize final layer norm and head
        self.ln_f = LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register buffer for position IDs
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Ensure input is at least 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Create causal attention mask
        seq_length = input_ids.size(-1)
        device = input_ids.device
        
        # Create causal mask [1, 1, seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Convert boolean mask to float
        causal_mask = causal_mask.to(torch.float32)
        causal_mask = causal_mask.masked_fill(causal_mask.bool(), float('-inf'))
        
        # Combine with attention_mask if provided
        if attention_mask is not None:
            # Convert attention_mask to float and expand
            attention_mask = attention_mask.to(torch.float32)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_length, -1)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            causal_mask = causal_mask + attention_mask
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        #