from typing import Optional, Tuple
import math

import attrs
import torch
from torch import nn
from torch.nn import functional as F


# Defaults to 13b.
@attrs.define
class Baichuan2ModelConfig:
    hidden_size: int = 5120
    initializer_range: float = 0.02
    intermediate_size: int = 13696
    model_max_length: int = 4096
    model_max_batch_size: int = 1
    num_attention_heads: int = 40
    num_hidden_layers: int = 40
    pad_token_id: int = 0
    rms_norm_eps: float = 1e-06
    vocab_size: int = 125696
    debug: bool = False


def _get_interleave(n: int):

    def _get_interleave_power_of_2(n: int):
        start = 2**(-(2**-(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
        )


def _fill_with_neg_inf(t: torch.Tensor):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _gen_alibi_mask(n_head: int, max_pos: int):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        # convert into half-precision
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class MLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BaichuanAttention(torch.nn.Module):

    def __init__(self, config: Baichuan2ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        cache_shape = (
            config.model_max_batch_size,
            self.num_heads,
            config.model_max_length,
            self.head_dim,
        )
        self.register_buffer(
            'k_cache',
            torch.zeros(cache_shape, dtype=self.W_pack.weight.dtype),
            persistent=False,
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(cache_shape, dtype=self.W_pack.weight.dtype),
            persistent=False,
        )

    def forward(
        self,
        input_pos: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 1. q_len >= 1 for prefilling.
        # 2. q_len = 1 for generating.
        _, q_len, _ = hidden_states.size()
        # TODO: bachify.
        bsz = 1

        proj = self.W_pack(hidden_states)
        proj = (proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2))
        # [batch_size, num_heads, q_len, head_dim]
        query_states = (proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
        key_states = (proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))
        value_states = (proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))

        # input_pos: [q_len]
        self.k_cache[:, :, input_pos] = key_states
        self.v_cache[:, :, input_pos] = value_states

        key_states = self.k_cache
        value_states = self.v_cache

        # attention_mask: [num_heads, seq, seq]
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask[:, input_pos].unsqueeze(0),
        )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class BaichuanLayer(torch.nn.Module):

    def __init__(self, config: Baichuan2ModelConfig):
        super().__init__()
        self.debug = config.debug
        self.hidden_size = config.hidden_size
        self.self_attn = BaichuanAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def forward(
        self,
        input_pos: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        if self.debug:
            layer_device = self.input_layernorm.weight.device
            input_pos = input_pos.to(layer_device)
            hidden_states = hidden_states.to(layer_device)
            attention_mask = attention_mask.to(layer_device)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            input_pos=input_pos,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class NormHead(nn.Module):

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states: torch.Tensor):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
            self.first_flag = True
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


class Baichuan2Model(torch.nn.Module):

    def __init__(self, config: Baichuan2ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # [num_heads, model_max_length, model_max_length]
        self.register_buffer(
            "alibi_mask",
            _gen_alibi_mask(config.num_attention_heads, config.model_max_length),
            persistent=False,
        )

        self.layers = torch.nn.ModuleList([
            BaichuanLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.lm_head = NormHead(config.hidden_size, config.vocab_size)

    def forward(self, input_pos: torch.Tensor, input_ids: torch.Tensor):
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                input_pos=input_pos,
                hidden_states=hidden_states,
                attention_mask=self.alibi_mask,
            )
        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits
