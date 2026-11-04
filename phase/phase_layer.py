# Extracted from the local PhaseSNN implementation; see NOTICE and LICENSE.
import math
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2RotaryEmbedding, rotate_half, repeat_kv
import utils.hadamard_utils as hadamard_utils

def seq_matmul(A, B):
    S_q = A.cumsum(dim=0)
    S_k = B.cumsum(dim=0)

    term1 = torch.matmul(S_q, B)

    term2 = torch.matmul(A, S_k)

    term3 = torch.matmul(A, B)

    attn_scores = term1 + term2 - term3
    return attn_scores

def compute_hadamard_v2(A, B):
    """
    计算满足 sum(C_m) = (sum(A_i)) ⊙ (sum(B_j)) 的 C_m 矩阵（适用于 A, B 为 (T, bsz, Q, K)）

    参数:
        A: torch.Tensor, 形状为 (T, bsz, Q, K)
        B: torch.Tensor, 形状为 (T, bsz, Q, K)

    返回:
        torch.Tensor: 形状为 (T, bsz, Q, K)，包含所有 C_m 矩阵
    """
    T, bsz, Q, K = A.shape


    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(0)
    outer_hadamard = A_exp * B_exp


    diag_terms = torch.diagonal(outer_hadamard, dim1=0, dim2=1)
    diag_terms = diag_terms.permute(3, 0, 1, 2)


    transposed = outer_hadamard.permute(1, 0, 2, 3, 4)
    off_diag_terms = (outer_hadamard + transposed) / 2


    mask = ~torch.eye(T, dtype=torch.bool, device=A.device)
    off_diag_sum = off_diag_terms[mask].view(T, T-1, bsz, Q, K).sum(dim=1)


    C = diag_terms + off_diag_sum


    return C

def compute_hadamard_v3(A, B):
    """
    计算满足 sum(C_m) = (sum(A_i)) ⊙ (sum(B_j)) 的 C_m 矩阵（适用于 A, B 为 (T, bsz, Q, K)）

    参数:
        A: torch.Tensor, 形状为 (T, bsz, Q, K)
        B: torch.Tensor, 形状为 (T, bsz, Q, K)

    返回:
        torch.Tensor: 形状为 (T, bsz, Q, K)，包含所有 C_m 矩阵
    """
    T, bsz, Q, K = A.shape


    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(0)
    outer_hadamard = A_exp.float() * B_exp.float()


    diag_terms = torch.diagonal(outer_hadamard, dim1=0, dim2=1)
    diag_terms = diag_terms.permute(3, 0, 1, 2)


    transposed = outer_hadamard.permute(1, 0, 2, 3, 4)
    off_diag_terms = (outer_hadamard + transposed) / 2


    mask = ~torch.eye(T, dtype=torch.bool, device=A.device)
    off_diag_sum = off_diag_terms[mask].view(T, T-1, bsz, Q, K).sum(dim=1)


    C = diag_terms + off_diag_sum


    return C.half()

class phaseSnnLinear2(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        T,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.T = T

    def forward(self, input: torch.Tensor):
        BT, D, L = input.shape
        bsz = BT // self.T
        input = input.view(self.T, bsz, D, L)
        bias = self.bias
        input = input.to(self.weight.dtype)
        out = self.fwd_func(input, self.weight, bias, **self.fwd_kwargs)

        return out.sum(dim=0)

class phaseSnnLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module,
        T,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16


        self.online_full_had=False
        self.use_temporary_parameter=False

        self.T = T


    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype


        if self.online_full_had:
            if self.fp32_had:
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else:
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias
        if bias is not None:
            bias = bias / self.T

        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and hasattr(self, "input_quantizer"):
            input = self.input_quantizer(input)
        input = input.to(weight.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        if self.use_act_quant and hasattr(self, "output_quantizer"):
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant

class phaseSnnIdentity(nn.Module):
    def __init__(
        self,
        out_features,
        T,
    ):
        super().__init__()


        self.use_act_quant = False


        self.out_features = out_features


    def forward(self, x):
        out = x
        if self.use_act_quant:
            out = self.input_quantizer(out)
        return out

    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant

class phaseSnnEmbedding(nn.Module):
    def __init__(
        self,
        original_embedding,
        T=4,
    ) -> None:
        super().__init__()
        self.oriebd = original_embedding
        self.register_buffer('weight', self.oriebd.weight)
        self.fwd_func = F.embedding
        self.T = T

    def forward(self, input: torch.Tensor) -> torch.Tensor:


        out = self.fwd_func(
            input,
            self.weight,
            self.oriebd.padding_idx,
            self.oriebd.max_norm,
            self.oriebd.norm_type,
            self.oriebd.scale_grad_by_freq,
            self.oriebd.sparse,
        )
        B,  L, D=out.shape
        out = out/self.T
        out.unsqueeze_(0)
        out = out.repeat(self.T, 1, 1,  1)


        return out.view(-1, L, D)

class snnQwen2RMSNorm(nn.Module):
    def __init__(
        self,
        ori_norm,
        T=2
    ):
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = ori_norm.variance_epsilon
        self.use_temporary_parameter = False
        self.use_act_quant = False

        self.out_features = self.weight.shape[-1]

        self.T = T


    def set_quant_state(self, act_quant: bool = False):
        self.use_act_quant = act_quant

    def forward(self, x):
        BT, D, L = x.shape
        B = BT // self.T
        x = x.view(self.T, B, D, L)
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        X = torch.zeros_like(x[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X = X + x[t]
            variance = X.pow(2).mean(-1, keepdim=True)
            Y = X * torch.rsqrt(variance + self.variance_epsilon)
            Y =  Y.to(input_dtype) * weight
            Out.append(Y-Y_pre)
            Y_pre = Y
        Out = torch.stack(Out, dim=0)
        if self.use_act_quant:
            Out = self.output_quantizer(Out)
        Out = Out.view(-1, D, L)


        return Out

class snnQwen2MLP(nn.Module):
    def __init__(
        self,
        org_module,
        config,
        T=2,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = phaseSnnLinear(
            org_module.gate_proj, T
        )
        self.up_proj = phaseSnnLinear(
            org_module.up_proj, T
        )
        self.down_proj = phaseSnnLinear(
            org_module.down_proj, T
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.T = T

        self.silu_Identity = phaseSnnIdentity(self.gate_proj.out_features, T)

    def forward(self, x):
        T= self.T
        BT, D, L  = x.shape
        bsz = BT // T
        x = x.view(T, bsz, D, L)

        out_up = self.up_proj(x)
        out_g = self.gate_proj(x)

        X = torch.zeros_like(out_g[0])
        Y_pre = 0
        Out = []
        for t in range(self.T):
            X += out_g[t]
            Y = self.act_fn(X)
            Out.append(Y-Y_pre)
            Y_pre = Y
        act_out_g = torch.stack(Out, dim=0)
        act_out_g = self.silu_Identity(act_out_g)
        out = compute_hadamard_v2(act_out_g, out_up)
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = compute_hadamard_v3(act_out_g, out_up)

        down_proj = self.down_proj(out)
        down_proj = down_proj.view(-1, D, L)

        return down_proj

class snnQwenAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, org_module, config: Qwen2Config, T):
        super().__init__()
        self.config = config
        self.layer_idx = org_module.layer_idx


        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.T = T

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )


        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = phaseSnnLinear(
            org_module.q_proj, self.T
        )
        self.k_proj = phaseSnnLinear(
            org_module.k_proj, self.T
        )
        self.v_proj = phaseSnnLinear(
            org_module.v_proj, self.T
        )
        self.o_proj = phaseSnnLinear(
            org_module.o_proj, self.T
        )


        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=org_module.max_position_embeddings,
            base=org_module.rope_theta,
        )
        self.is_causal = org_module.is_causal
        self.flag = True

        if self.flag:
            self.q_Identity = phaseSnnIdentity(org_module.q_proj.out_features, T)
            self.k_Identity = phaseSnnIdentity(org_module.q_proj.out_features, T)
            self.v_Identity = phaseSnnIdentity(org_module.q_proj.out_features, T)
            self.softmax_Identity = phaseSnnIdentity(self.num_heads, T)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        Tbsz, q_len, _ = hidden_states.size()
        T = self.T
        bsz = Tbsz // T
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(T * bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(T * bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(T * bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)


        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        key_states = key_states.view(T, bsz, self.num_key_value_heads, q_len, self.head_dim)
        value_states = value_states.view(T, bsz, self.num_key_value_heads, q_len, self.head_dim)
        key_states = key_states.permute(1, 0, 2, 3, 4)
        value_states = value_states.permute(1, 0, 2, 3, 4)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = key_states.permute(1, 0, 2, 3, 4)
        value_states = value_states.permute(1, 0, 2, 3, 4)
        T, B, H, Q, D = key_states.shape
        key_states = key_states.contiguous().view(-1, H, Q, D)
        value_states = value_states.contiguous().view(-1, H, Q, D)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        if self.flag:
            BT, H, Q, D = key_states.shape

            query_states = query_states.transpose(1, 2).contiguous().view(T, bsz, q_len, -1)
            key_states = key_states.transpose(1, 2).contiguous().view(T, bsz, Q, -1)
            value_states = value_states.transpose(1, 2).contiguous().view(T, bsz, Q, -1)
            query_states = self.q_Identity(query_states)
            value_states = self.v_Identity(value_states)
            key_states = self.k_Identity(key_states)
            query_states = query_states.view(T, bsz, q_len, self.num_heads, self.head_dim).transpose(2, 3)
            key_states = key_states.view(T, bsz, Q, H, D).transpose(2, 3)
            value_states = value_states.view(T, bsz, Q, H, D).transpose(2, 3)

        key_states = key_states.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn_weights = seq_matmul(query_states, key_states)

        T, B, H, Q, K = attn_weights.shape
        attn_weights = attn_weights.view(-1, H, Q, K)

        if attn_weights.size() != (T * bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(T * bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )


        if attention_mask is not None:
            attention_mask = attention_mask.repeat(T, 1, 1, 1)
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attention_mask


        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(K, K, dtype=torch.bool, device=query_states.device), diagonal=1
            )
            causal_mask = causal_mask[K-Q:]
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))


        BT, Q, K, D = attn_weights.shape
        attn_weights = attn_weights.view(T, bsz, Q, K, D)

        X = torch.zeros_like(attn_weights[0])
        Y_pre = 0
        Out = []

        for t in range(self.T):
            X = X + attn_weights[t]
            Y = nn.functional.softmax(X, dim=-1)
            Out.append(Y-Y_pre)
            Y_pre = Y

        attn_weights = torch.stack(Out, dim=0)


        if attn_weights.size() != (T, bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"`attn_output` should be of size {(T, bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.flag:
            T, B, head,d1,d2=attn_weights.shape
            attn_weights = attn_weights.view(T, bsz, head, -1).transpose(2,3)
            attn_weights = self.softmax_Identity(attn_weights)
            attn_weights = attn_weights.transpose(2, 3).contiguous().view(-1, head, d1, d2)

        dropout_p=self.attention_dropout if self.training else 0.0

        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)


        TB, head,d1,d2 = attn_weights.shape
        attn_weights = attn_weights.view(T, bsz, head, d1, d2)
        attn_output = seq_matmul(attn_weights, value_states)


        attn_output = attn_output.transpose(2, 3).contiguous()
        attn_output = attn_output.reshape(T, bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        attn_output = attn_output.view(-1, q_len, self.hidden_size)


        return attn_output, None, past_key_value
