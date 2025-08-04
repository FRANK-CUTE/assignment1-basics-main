from __future__ import annotations

import math
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch

from torch import nn
from torch import Tensor
from torch.distributed.tensor import empty
from torch.nn import init
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 初始化父类 nn.Module
        super().__init__()

        # 创建一个参数张量 W，形状为 [out_features, in_features]
        # 即输出特征数 × 输入特征数
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 使用截断正态分布初始化权重（均值默认为0，标准差为0.02）
        init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: Float["... d_in"]) -> Float["... d_out"]:
        # 等价于 return x @ self.W.T
        # 使用爱因斯坦求和，明确维度对应关系
        return torch.einsum("... d, o d -> ... o", x, self.weight)


def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 初始化 Linear 模块（不使用 bias）
    model = Linear(d_in, d_out)

    # 加载外部给定的权重
    model.load_state_dict({"W": weights})

    # 前向传播，返回输出
    return model(in_features)


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # embedding矩阵形状为 (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        # 用截断正态分布初始化embedding权重
        init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids 是长整型张量，值为词表索引
        # 直接通过索引选取对应的embedding行
        return self.weight[token_ids]


def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"],
        token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    # 初始化Embedding模型
    model = Embedding(vocab_size, d_model)
    # 加载测试权重到模型参数
    model.load_state_dict({"weight": weights})
    return model(token_ids)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None, ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # 创建 SwiGLU 实例
    device = in_features.device
    dtype = in_features.dtype
    swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    # 加载权重（注意：Linear 的 weight 是 [out_features, in_features]，和 wX_weight 的维度匹配）
    swiglu.w1.weight.data.copy_(w1_weight)
    swiglu.w2.weight.data.copy_(w2_weight)
    swiglu.w3.weight.data.copy_(w3_weight)
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    # 执行前向传播
    return swiglu(in_features)

class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Float[Tensor, "... queries keys"]
    ) -> Float[Tensor, "... queries d_v"]:
        d_k = Q.shape[-1]
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

        # 注意：此处不再判断 mask 是否为 None，要求调用者一定传入
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("...qk,...kd->...qd", attn_weights, V)
        return output

def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    sdpa = ScaledDotProductAttention()
    return sdpa(Q, K, V, mask)
    raise NotImplementedError

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(self, x: Float[Tensor, "batch seq_len d_model"],
                mask: Float[Tensor, "batch seq_len seq_len"]) \
            -> Float[Tensor, "batch seq_len d_model"]:
        B, L, d_model_ = x.shape
        H = self.num_heads
        d_h = d_model_ // H

        # Linear projections and reshape into multiple heads
        q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)

        # mask = mask.unsqueeze(1)  # [B, 1, L, L] 不写都可以！！！

        attn_output = run_scaled_dot_product_attention(q, k, v, mask)

        attn_output = torch.einsum('bhld->blhd', attn_output).reshape(B, L, H * d_h)

        return self.output_proj(attn_output)

class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    def __init__(self, d_model: int, num_heads: int,
                 max_seq_len: int, theta: float,
                 device=None, dtype=None):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)

        self.max_seq_len = max_seq_len
        self.theta = theta

        # 假设你已经实现了这个类
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        mask: Float[torch.Tensor, "batch seq_len seq_len"],
        token_positions: Int[torch.Tensor, "batch seq_len"] | None = None
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        B, L, _ = x.shape
        H = self.num_heads
        d_h = self.d_k

        # 线性投影 + reshape
        q = self.q_proj(x).view(B, L, H, d_h).transpose(1, 2)  # [B, H, L, d_k]
        k = self.k_proj(x).view(B, L, H, d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_h).transpose(1, 2)

        # token_positions 自动填充
        if token_positions is None:
            token_positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)  # [B, L]

        # 应用 RoPE：对 q 和 k，head 维度作为 batch 维度广播
        q = self.rope(q, token_positions)  # [B, H, L, d_k]
        k = self.rope(k, token_positions)

        # mask: [B, 1, L, L]
        # mask = mask.unsqueeze(1) 不写都可以！！！

        attn_output = run_scaled_dot_product_attention(q, k, v, mask)  # [B, H, L, d_k]

        # 还原 shape -> [B, L, H * d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, H * d_h)

        return self.output_proj(attn_output)


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # 创建 MultiheadSelfAttention 实例
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads)

    # 赋值权重（你自定义的 Linear.weight 形状是 [out_features, in_features]，与传入权重匹配）
    mha.q_proj.weight.data.copy_(q_proj_weight)
    mha.k_proj.weight.data.copy_(k_proj_weight)
    mha.v_proj.weight.data.copy_(v_proj_weight)
    mha.output_proj.weight.data.copy_(o_proj_weight)

    # 获取 batch 和序列长度
    *batch_dims, seq_len, _ = in_features.shape
    batch_size = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1

    # 展平 batch 维度，以适配实现（假设你实现支持 [B, L, D] 格式）
    x = in_features.reshape(batch_size, seq_len, -1)

    # ======== Step 1: 构造 causal mask ========
    # mask shape: [seq_len, seq_len]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    # 扩展为 [batch_size, seq_len, seq_len] 再传入
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]

    # ======== Step 2: 执行多头注意力 ========
    output = mha(x, mask=mask)

    # ======== Step 3: 还原 batch 维度 ========
    if batch_dims:
        output = output.reshape(*batch_dims, seq_len, -1)

    return output
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    # 创建 MultiheadSelfAttention 实例
    mha = MultiheadSelfAttentionWithRoPE(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta
    )

    # 赋值权重（你自定义的 Linear.weight 形状是 [out_features, in_features]，与传入权重匹配）
    mha.q_proj.weight.data.copy_(q_proj_weight)
    mha.k_proj.weight.data.copy_(k_proj_weight)
    mha.v_proj.weight.data.copy_(v_proj_weight)
    mha.output_proj.weight.data.copy_(o_proj_weight)

    # 获取 batch 和序列长度
    *batch_dims, seq_len, _ = in_features.shape
    batch_size = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1

    # 展平 batch 维度，以适配实现（假设你实现支持 [B, L, D] 格式）
    x = in_features.reshape(batch_size, seq_len, -1)

    # ======== 构造 causal mask ========
    # mask shape: [seq_len, seq_len]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    # 扩展为 [batch_size, seq_len, seq_len] 再传入
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    # mask = mask.unsqueeze(1)  # 不需要变成 [B, 1, L, L] 这不影响后续的

    # ======== Step 2: 执行多头注意力 ========
    output = mha(x, mask=mask, token_positions=token_positions)

    # ======== Step 3: 还原 batch 维度 ========
    if batch_dims:
        output = output.reshape(*batch_dims, seq_len, -1)

    return output
    raise NotImplementedError

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 生成频率分量： [d_k // 2] 个维度上的频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

        # 生成位置索引 [0, 1, ..., max_seq_len - 1]
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
        freqs = position * inv_freq.unsqueeze(0)  # [seq_len, d_k // 2]

        # 预计算正余弦值
        cos = freqs.cos()  # [seq_len, d_k // 2]
        sin = freqs.sin()  # [seq_len, d_k // 2]

        # 注册为 buffer（非参数但会保存）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [..., seq_len, d_k]，token_positions: [..., seq_len]
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k

        # 获取每个位置的 cos/sin（形状 [..., seq_len, d_k//2]）
        cos = self.cos[token_positions]  # [..., seq_len, d_k//2]
        sin = self.sin[token_positions]  # [..., seq_len, d_k//2]

        # 拆分 x 为偶数和奇数部分（假设 x 最后一维是 d_k）
        x1 = x[..., ::2]  # [..., seq_len, d_k//2]
        x2 = x[..., 1::2] # [..., seq_len, d_k//2]

        # 应用旋转公式
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 重组回原顺序： interleave
        out = torch.stack([out1, out2], dim=-1)  # [..., seq_len, d_k//2, 2]
        return out.flatten(-2)  # [..., seq_len, d_k]

def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # 创建 RoPE 实例（会自动构造 cos 和 sin 缓存）
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)

    # 应用 RoPE 旋转位置编码
    return rope(in_query_or_key, token_positions)
    raise NotImplementedError


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    # 创建 TransformerBlock 实例
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta
    )
    
    # 加载权重
    block.load_state_dict(weights)
    
    # 创建因果mask
    batch_size, seq_len, _ = in_features.shape
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    
    # 前向传播
    return block(in_features, mask=causal_mask)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        mask: Float[Tensor, "batch seq_len seq_len"] = None,
        token_positions: Int[torch.Tensor, "batch seq_len"] | None = None
    ) -> Float[Tensor, "batch seq_len d_model"]:
        x = x + self.attn(self.ln1(x), mask=mask, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float,
    ):
        super().__init__()
        
        # 保存配置
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model)
        
        # LM head
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(
            self,
            input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.context_length, f"输入序列长度{seq_len}超过最大上下文长度{self.context_length}"
        
        # Token embeddings
        x = self.token_embeddings(input_ids)
        
        # 创建因果mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # 创建模型实例
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
    )
    
    # 加载权重
    model.load_state_dict(weights)
    
    # 前向传播
    return model(in_indices)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.einsum('...d, ...d -> ...', x, x).div(x.shape[-1]).add(self.eps))

        x_norm = x / rms.unsqueeze(-1)
        # x_norm = x / rms

        # 应用缩放参数 gamma
        result = torch.einsum('...d, d -> ...d', x_norm, self.weight)
        return result.to(in_dtype)


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms = RMSNorm(d_model, eps)
    with torch.no_grad():
        rms.weight.copy_(weights)

    return rms(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    x_exp = torch.exp(x_stable)

    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    return x_exp / x_sum


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)
    raise NotImplementedError


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[
    Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_logits, _ = inputs.max(dim=-1, keepdim=True)
    stable_logits = inputs - max_logits
    exp_logits = torch.exp(stable_logits)
    exp_sum = exp_logits.sum(dim=-1)
    batch_indices = torch.arange(targets.shape[0], device=inputs.device)
    target_logits = stable_logits[batch_indices, targets]
    return (-target_logits + torch.log(exp_sum)).mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: 要优化的参数
            lr (float): 学习率 α
            betas (tuple[float, float]): β₁ 和 β₂ 系数
            eps (float): 数值稳定性系数 ε
            weight_decay (float): 权重衰减系数 λ
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步优化。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                # 获取参数
                grad = p.grad
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                # 获取状态
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                # 获取动量
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"]
                
                # 更新偏差校正系数
                bias_correction1 = 1 - beta1 ** (step + 1)
                bias_correction2 = 1 - beta2 ** (step + 1)
                
                # 应用权重衰减
                p.data.mul_(1 - lr * weight_decay)
                
                # 更新一阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 更新二阶动量
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算步长
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                
                # 更新参数
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # 更新步数
                state["step"] += 1
                
        return loss

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # 线性预热阶段
    if it < warmup_iters:
        # 从 0 线性增加到 max_learning_rate
        return (it / warmup_iters) * max_learning_rate
    
    # 余弦衰减阶段
    # 首先确保 it 不超过总迭代次数
    t = min(it - warmup_iters, cosine_cycle_iters)
    
    # 应用余弦衰减公式：
    # α_t = α_min + 0.5(α_max - α_min)(1 + cos(πt/T_c))
    cosine_factor = 0.5 * (1 + math.cos(math.pi * t / cosine_cycle_iters))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_factor


def run_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
