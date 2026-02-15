from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import IO, Any, BinaryIO, Optional, Callable, Iterator
from collections.abc import Iterable
from cs336_basics import tokenizer
import numpy as np
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor, nn
from einops import rearrange, einsum
import math
import regex as re


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
    model = Linear(d_in, d_out)
    state_dict = {"W": weights}
    model.load_state_dict(state_dict)
    return model(in_features)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.W, mean=0., std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.W.T
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


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
    model = Embedding(vocab_size, d_model)
    state_dict = {"emb", weights}
    model.load_state_dict(state_dict)
    return model(token_ids)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.emb, mean=0., std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]


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
    swiglu = SwiGLU(d_model, d_ff)
    state_dict = {"w1": w1_weight, "w2": w2_weight, "w3": w3_weight}
    swiglu.load_state_dict(state_dict)
    return swiglu(in_features)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        nn.init.trunc_normal_(self.w1, mean=0., std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0., std=0.02)
        nn.init.trunc_normal_(self.w3, mean=0., std=0.02)

    def forward(self, x):
        mid = silu(einsum(x, self.w1, "... d_model, d_ff d_model-> ... d_ff")) * (
            einsum(x, self.w3, "... d_model, d_ff d_model-> ... d_ff"))
        return einsum(mid, self.w2, "... d_ff, d_model d_ff -> ... d_model")


def silu(x):
    return x * torch.sigmoid(x)


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

    d_q = Q.shape[-1]
    logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_q)
    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))
    prob = run_softmax(logits, -1)
    return einsum(prob, V, "... queries keys, ... keys d_v -> ... queries d_v")


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
    seq_len = in_features.shape[-2]
    Q = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    K = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    V = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    Q_mh = rearrange(Q, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)
    K_mh = rearrange(K, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)
    V_mh = rearrange(V, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)

    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mh_output = run_scaled_dot_product_attention(Q_mh, K_mh, V_mh, mask)
    mh_output = rearrange(mh_output,
                          "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)",
                          num_heads=num_heads)
    return einsum(mh_output, o_proj_weight, "... sequence_length d_k, d_model d_k -> ... sequence_length d_model")


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
    positional_encoder = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    mha = CausalMultiHeadSelfAttention(d_model, num_heads, positional_encoder=positional_encoder)
    with torch.no_grad():
        mha.q_proj.W.copy_(q_proj_weight)
        mha.k_proj.W.copy_(k_proj_weight)
        mha.v_proj.W.copy_(v_proj_weight)
        mha.o_proj.W.copy_(o_proj_weight)

    result = mha(in_features, token_positions)
    return result
    '''
    # 这是不用类实现的方法
    seq_len = in_features.shape[-2]
    Q = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    K = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
    V = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")

    Q_mh = rearrange(Q, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)
    K_mh = rearrange(K, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)
    V_mh = rearrange(V, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                     num_heads=num_heads)
    ### run_rope 方法
    # Q_rope = run_rope(d_model // num_heads, theta, max_seq_len, Q_mh, token_positions)
    # K_rope = run_rope(d_model // num_heads, theta, max_seq_len, K_mh, token_positions)
    ### run_rope 方法

    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    Q_rope = rope(Q_mh, token_positions)
    K_rope = rope(K_mh, token_positions)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mh_output = run_scaled_dot_product_attention(Q_rope, K_rope, V_mh, mask)
    mh_output = rearrange(mh_output,
                          "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)",
                          num_heads=num_heads)
    return einsum(mh_output, o_proj_weight, "... sequence_length d_k, d_model d_k -> ... sequence_length d_model")
    '''


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, positional_encoder):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)
        self.positional_encoder = positional_encoder

    def forward(self, x, token_positions=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        seq_len = x.shape[-2]
        Q_mh = rearrange(Q, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                         num_heads=self.num_heads)
        K_mh = rearrange(K, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                         num_heads=self.num_heads)
        V_mh = rearrange(V, "... sequence_length (num_heads head_dim) -> ... num_heads sequence_length head_dim",
                         num_heads=self.num_heads)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        if token_positions is not None:
            Q_rope = self.positional_encoder(Q_mh, token_positions)
            K_rope = self.positional_encoder(K_mh, token_positions)
            attn_output = run_scaled_dot_product_attention(Q=Q_rope, K=K_rope, V=V_mh, mask=causal_mask)
        else:
            attn_output = run_scaled_dot_product_attention(Q=Q_mh, K=K_mh, V=V_mh, mask=causal_mask)
        attn_output = rearrange(attn_output,
                                "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)",
                                num_heads=self.num_heads)
        result = self.o_proj(attn_output)
        return result


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
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_ang = 1 / (theta ** (2 * torch.arange(d_k // 2) / d_k))
        index = torch.arange(max_seq_len)
        angles = torch.outer(index, inv_ang)
        cos_ = torch.cos(angles)
        sin_ = torch.sin(angles)
        cos_cache = torch.repeat_interleave(cos_, 2, dim=-1)
        sin_cache = torch.repeat_interleave(sin_, 2, dim=-1)
        self.register_buffer("cos", cos_cache)
        self.register_buffer("sin", sin_cache)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        if x.ndim == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        return x * cos + x_rotated * sin


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
    bs, slen, _ = in_features.shape
    token_positions = torch.arange(slen).repeat(bs, 1)
    positional_encoder = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    block = TransformerBlock(d_model, num_heads, d_ff, positional_encoder)
    with torch.no_grad():
        block.ffn.w1.copy_(weights["ffn.w1.weight"])
        block.ffn.w2.copy_(weights["ffn.w2.weight"])
        block.ffn.w3.copy_(weights["ffn.w3.weight"])
        block.pre_norm1.gain.copy_(weights["ln1.weight"])
        block.pre_norm2.gain.copy_(weights["ln2.weight"])
        block.mha.q_proj.W.copy_(weights["attn.q_proj.weight"])
        block.mha.k_proj.W.copy_(weights["attn.k_proj.weight"])
        block.mha.v_proj.W.copy_(weights["attn.v_proj.weight"])
        block.mha.o_proj.W.copy_(weights["attn.output_proj.weight"])
    return block(in_features, token_positions)

    """
    bs, slen, _ = in_features.shape
    token_positions = torch.arange(slen).repeat(bs, 1)
    pre_norm1 = RMSNorm(d_model)
    pre_norm1.gain.data = weights["ln1.weight"]
    pre_norm2 = RMSNorm(d_model)
    pre_norm2.gain.data = weights["ln2.weight"]
    ffn = SwiGLU(d_model, d_ff)
    ffn.w1.data = weights["ffn.w1.weight"]
    ffn.w2.data = weights["ffn.w2.weight"]
    ffn.w3.data = weights["ffn.w3.weight"]

    pre_norm_x1 = pre_norm1(in_features)
    positional_encoder = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    mha = CausalMultiHeadSelfAttention(d_model, num_heads, positional_encoder=positional_encoder)
    with torch.no_grad():
        mha.q_proj.W.copy_(weights["attn.q_proj.weight"])
        mha.k_proj.W.copy_(weights["attn.k_proj.weight"])
        mha.v_proj.W.copy_(weights["attn.v_proj.weight"])
        mha.o_proj.W.copy_(weights["attn.output_proj.weight"])

    mha_result = mha(pre_norm_x1, token_positions)
    sub1 = mha_result + in_features
    pre_norm_x2 = pre_norm2(sub1)
    result = ffn(pre_norm_x2) + sub1
    return result
    """


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, positional_encoder):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.positional_encoder = positional_encoder
        self.pre_norm1 = RMSNorm(d_model)
        self.pre_norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.mha = CausalMultiHeadSelfAttention(self.d_model, self.num_heads,
                                                positional_encoder=self.positional_encoder)

    def forward(self, in_features, token_positions=None):
        pre_norm_x1 = self.pre_norm1(in_features)
        mha_result = self.mha(pre_norm_x1, token_positions)
        sub1 = mha_result + in_features
        pre_norm_x2 = self.pre_norm2(sub1)
        result = self.ffn(pre_norm_x2) + sub1
        return result


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
    trans_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    with torch.no_grad():
        trans_lm.embd.emb.copy_(weights["token_embeddings.weight"])
        trans_lm.ln_final.gain.copy_(weights["ln_final.weight"])
        trans_lm.lm_head.W.copy_(weights["lm_head.weight"])

    i = 0
    for layer in trans_lm.layers:
        prefix = f"layers.{i}."
        layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        with torch.no_grad():
            layer.ffn.w1.copy_(layer_weights["ffn.w1.weight"])
            layer.ffn.w2.copy_(layer_weights["ffn.w2.weight"])
            layer.ffn.w3.copy_(layer_weights["ffn.w3.weight"])
            layer.pre_norm1.gain.copy_(layer_weights["ln1.weight"])
            layer.pre_norm2.gain.copy_(layer_weights["ln2.weight"])
            layer.mha.q_proj.W.copy_(layer_weights["attn.q_proj.weight"])
            layer.mha.k_proj.W.copy_(layer_weights["attn.k_proj.weight"])
            layer.mha.v_proj.W.copy_(layer_weights["attn.v_proj.weight"])
            layer.mha.o_proj.W.copy_(layer_weights["attn.output_proj.weight"])
        i += 1

    return trans_lm(in_indices)

    #
    # embd = Embedding(vocab_size, d_model)
    # embd.emb.data = weights["token_embeddings.weight"]
    # x = embd(in_indices)
    # positional_encoder = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
    # bs, slen, _ = x.shape
    # token_positions = torch.arange(slen).repeat(bs, 1)
    # for i in range(num_layers):
    #     prefix = f"layers.{i}."
    #     layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    #     # 类实现
    #     block = TransformerBlock(d_model, num_heads, d_ff, positional_encoder)
    #     with torch.no_grad():
    #         block.ffn.w1.copy_(layer_weights["ffn.w1.weight"])
    #         block.ffn.w2.copy_(layer_weights["ffn.w2.weight"])
    #         block.ffn.w3.copy_(layer_weights["ffn.w3.weight"])
    #         block.pre_norm1.gain.copy_(layer_weights["ln1.weight"])
    #         block.pre_norm2.gain.copy_(layer_weights["ln2.weight"])
    #         block.mha.q_proj.W.copy_(layer_weights["attn.q_proj.weight"])
    #         block.mha.k_proj.W.copy_(layer_weights["attn.k_proj.weight"])
    #         block.mha.v_proj.W.copy_(layer_weights["attn.v_proj.weight"])
    #         block.mha.o_proj.W.copy_(layer_weights["attn.output_proj.weight"])
    #
    #     x = block(x, token_positions)
    #
    # ln_final = RMSNorm(d_model)
    # ln_final.gain.data = weights["ln_final.weight"]
    # x = ln_final(x)
    # lm_head = Linear(vocab_size, d_model)
    # lm_head.W.data = weights["lm_head.weight"]
    # logits = lm_head(x)
    # return logits


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.embd = Embedding(self.vocab_size, self.d_model)
        self.positional_encoder = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.positional_encoder) for _ in
             range(self.num_layers)])
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        bs, slen = x.shape
        x = self.embd(x)
        token_positions = torch.arange(slen).repeat(bs, 1)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


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
    model = RMSNorm(d_model, eps)
    state_dict = {"gain": weights}
    model.load_state_dict(state_dict)
    return model(in_features)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(self.d_model))
        nn.init.trunc_normal_(self.gain, mean=0., std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        result = (x / rms) * self.gain
        return result.to(in_dtype)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SfiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


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

    sl = len(dataset)
    x_list = []
    y_list = []
    for _ in range(batch_size):
        start = random.randint(0, sl - context_length - 1)
        x = torch.tensor(dataset[start:start + context_length], dtype=torch.long, device=device)
        y = torch.tensor(dataset[start + 1:start + context_length + 1], dtype=torch.long, device=device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list).to(device), torch.stack(y_list).to(device)


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
    in_features = in_features - in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


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
    bs = inputs.shape[0]
    x_scale = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum = torch.log(torch.exp(x_scale).sum(dim=-1, keepdim=True))
    return (log_sum - x_scale)[torch.arange(bs), targets].mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    l2_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        l2_norm += torch.linalg.vector_norm(p.grad.data, ord=2) ** 2

    l2_norm = torch.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    return AdamW


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            betas1, betas2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t, m, v = state["t"], state["m"], state["v"]
                grad = p.grad.data
                m.mul_(betas1).add_(grad, alpha=1 - betas1)
                v.mul_(betas2).add_(grad ** 2, alpha=1 - betas2)
                alpha_t = lr * math.sqrt(1 - betas2 ** t) / (1 - betas1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1


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
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it < cosine_cycle_iters:
        return min_learning_rate + (
                1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (
                max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate


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
    check_point = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(check_point, out)


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
    check_point = torch.load(src)
    model.load_state_dict(check_point["model"])
    optimizer.load_state_dict(check_point["optimizer"])
    return check_point["iteration"]


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
    return Tokenizer(vocab, merges, special_tokens)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merges_priority = {pair: i for i, pair in enumerate(merges)}
        current_max_id = max(self.vocab.keys()) if self.vocab else -1

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            special_pattern = "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            self.pattern = re.compile(f"(?:{special_pattern})|{PAT}")
        else:
            self.pattern = re.compile(PAT)

        self.vocab_inv = {v: k for k, v in vocab.items()}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.vocab_inv:
                current_max_id += 1
                self.vocab[current_max_id] = st_bytes
                self.vocab_inv[st_bytes] = current_max_id
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            v_raw = json.load(f)
            vocab = {int(k): v.encode("utf-8") for k, v in v_raw.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p1, p2 = line.strip().split()
                    merges.append((p1.encode("utf-8"), p2.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _bpe_merge(self, word_bytes):
        if word_bytes in self.cache:
            return self.cache[word_bytes]
        tokens = [word_bytes[i:i + 1] for i in range(len(word_bytes))]
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = min(pairs, key=lambda p: self.merges_priority.get(p, float("inf")))
            if best_pair not in self.merges_priority:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        ids = [self.vocab_inv[t] for t in tokens]
        self.cache[word_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for m in self.pattern.finditer(text):
            chunk = m.group()
            if chunk in self.special_tokens:
                ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                ids.extend(self._bpe_merge(chunk.encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        bytes_seg = [self.vocab[i] for i in ids]
        return b"".join(bytes_seg).decode("utf-8", errors="replace")


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
    return tokenizer.train_bpe(input_path, vocab_size, special_tokens)


    # # Vocabulary
    # vocab = {i: bytes([i]) for i in range(256)}
    # for s in special_tokens:
    #     vocab[len(vocab)] = s.encode("utf-8")
    #
    # # Pre-tokenization
    # with open(input_path, "r", encoding="utf-8") as f:
    #     training_data = f.read()
    #
    # special_patterns = f"{"|".join(re.escape(s) for s in special_tokens)}"
    # if special_tokens:
    #     # chunks = re.split(special_patterns, training_data)
    #     chunks = [c for c in re.split(special_patterns, training_data) if c and c not in special_tokens]
    # else:
    #     chunks = [training_data]
    #
    # counts = Counter()
    # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # for chunk in chunks:
    #     if chunk in special_tokens:
    #         continue
    #     for m in re.finditer(PAT, chunk):
    #         word = m.group()
    #         # byte_tuple = tuple(bytes([w]) for w in word.encode("utf-8"))
    #         byte_tuple = tuple(word.encode("utf-8")[i:i + 1] for i in range(len(word.encode("utf-8"))))
    #         counts[byte_tuple] += 1
    #
    # pair_frequency = Counter()
    # for token_tuple, freq in counts.items():
    #     for index in range(len(token_tuple) - 1):
    #         pair_frequency[(token_tuple[index], token_tuple[index + 1])] += freq
    #
    # # Merges
    # merges = []
    # while len(vocab) < vocab_size:
    #     if not pair_frequency: break
    #
    #     best_pair = max(pair_frequency, key=lambda x: (pair_frequency[x], x))
    #     merges.append(best_pair)
    #     new_token = best_pair[0] + best_pair[1]
    #     vocab[len(vocab)] = new_token
    #
    #     new_counts = Counter()
    #     for token_tuple, freq in counts.items():
    #         new_word = []
    #         i = 0
    #         while i < len(token_tuple):
    #             if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
    #                 new_word.append(new_token)
    #                 i += 2
    #             else:
    #                 new_word.append(token_tuple[i])
    #                 i += 1
    #         new_counts[tuple(new_word)] = freq  # 统一赋值
    #
    #     counts = new_counts
    #     pair_frequency = Counter()
    #     for token_tuple, freq in counts.items():
    #         for index in range(len(token_tuple) - 1):
    #             pair_frequency[(token_tuple[index], token_tuple[index + 1])] += freq
    #
    # return vocab, merges
