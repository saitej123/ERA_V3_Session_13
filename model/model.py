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

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


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
        self.ln1 = LayerNorm(config.hidden_dim, config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_dim, config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = residual + self.dropout(self.mlp(hidden_states))

        return hidden_states


class SmolLM2(nn.Module):
    def __init__(self, config: SmolLM2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.hidden_dim, config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('fc2.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

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
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=None,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs, _ = self.forward(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids 