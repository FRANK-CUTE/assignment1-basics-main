from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor, nn


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
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


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0., std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.W.T
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.emb, mean=0., std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_ang = 1 / (theta ** (2 * torch.arange(d_k // 2, device=device) / d_k))
        index = torch.arange(max_seq_len, device=device)
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
            attn_output = scaled_dot_product_attention(Q=Q_rope, K=K_rope, V=V_mh, mask=causal_mask)
        else:
            attn_output = scaled_dot_product_attention(Q=Q_mh, K=K_mh, V=V_mh, mask=causal_mask)
        attn_output = rearrange(attn_output,
                                "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)",
                                num_heads=self.num_heads)
        result = self.o_proj(attn_output)
        return result


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        nn.init.trunc_normal_(self.w1, mean=0., std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0., std=0.02)
        nn.init.trunc_normal_(self.w3, mean=0., std=0.02)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        mid = silu(einsum(x, self.w1, "... d_model, d_ff d_model-> ... d_ff")) * (
            einsum(x, self.w3, "... d_model, d_ff d_model-> ... d_ff"))
        return einsum(mid, self.w2, "... d_ff, d_model d_ff -> ... d_model")


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

    def forward(self, in_features, token_positions):
        pre_norm_x1 = self.pre_norm1(in_features)
        mha_result = self.mha(pre_norm_x1, token_positions)
        sub1 = mha_result + in_features
        pre_norm_x2 = self.pre_norm2(sub1)
        result = self.ffn(pre_norm_x2) + sub1
        return result


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
        self.context_length = context_length
        self.embd = Embedding(self.vocab_size, self.d_model)
        self.positional_encoder = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, self.context_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.positional_encoder) for _ in
             range(self.num_layers)])
        self.ln_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        bs, slen = x.shape
        x = self.embd(x)
        token_positions = torch.arange(slen, device=x.device).repeat(bs, 1)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=None):
        """
        idx: (Batch, Seq_Len) 当前的 token indices, 也是 prompt
        max_new_tokens: 生成的最大新 token 数量
        temperature: 温度系数, 控制生成的随机性
        top_p: Nucleus sampling 的阈值
        """

        # 1. 循环生成 max_new_tokens 次
        for _ in range(max_new_tokens):
            # [cite_start]Step A: 裁剪 idx (Context Cropping) [cite: 767, 1062]
            # 如果 idx 长度超过了 context_length，必须截取最后 context_length 个 token。
            # 否则位置编码会越界报错。
            idx_cond = idx
            _, sl = idx_cond.shape
            if sl > self.context_length:
                idx_cond = idx_cond[:, -self.context_length:]

            # Step B: 前向传播获取 Logits
            # 调用 self(idx_cond) 获取所有 token 的 logits
            # 我们只需要最后一个时间步的 logits (预测下一个词)
            # logits = ... (请填空: 取最后一步, 形状变为 [Batch, Vocab_Size])
            logits = self.forward(idx_cond)[:, -1, :]

            # [cite_start]Step C: 温度采样 (Temperature Scaling) [cite: 1073-1076]
            # 如果 temperature != 1.0, 将 logits 除以 temperature
            # if temperature != 1.0:
            #     logits = ... (请填空)
            if temperature != 1.0:
                logits = logits / temperature

            # [cite_start]Step D: Top-p (Nucleus) Sampling [cite: 1077-1081]
            if top_p is not None and top_p < 1.0:
                # 1. 将 logits 转换为概率 (softmax)
                # probs = ...
                probs = softmax(logits, dim=-1)
                # 2. 对概率进行降序排序 (torch.sort)
                # sorted_probs, sorted_indices = ...
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # 3. 计算累积概率 (torch.cumsum)
                # cumulative_probs = ...
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 4. 创建移除掩码 (mask): 标记累积概率 > top_p 的部分
                # sorted_indices_to_remove = ...
                sorted_indices_to_remove = cumulative_probs > top_p

                # 5. 【关键】右移掩码 (Shift Right)
                # 保证至少保留第一个 token (即使它的概率 > top_p 也不应该被移除)
                # sorted_indices_to_remove[..., 1:] = ...
                # sorted_indices_to_remove[..., 0] = ...
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # 6. 恢复原索引顺序: 将 mask 映射回原始 vocab 的顺序 (scatter)
                # indices_to_remove = ...
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                # 7. 将被移除的 token 的 logits 设为负无穷 (-inf)
                # logits[...] = ...
                logits[indices_to_remove] = float('-inf')

            # [cite_start]Step E: 采样下一个 Token [cite: 1066-1068]
            # 1. 重新计算 softmax (因为 logits 可能变了)
            # probs = ...
            probs = softmax(logits, dim=-1)

            # 2. 使用 torch.multinomial 采样 1 个 token
            # idx_next = ...
            idx_next = torch.multinomial(probs, 1)
            # Step F: 拼接
            # 将新生成的 token 拼接到 idx 后面
            # idx = ...
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


def rmsnorm(
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
        self.gain = nn.Parameter(torch.empty(self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.gain, mean=0., std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        result = (x / rms) * self.gain
        return result.to(in_dtype)


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SfiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[
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


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
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


def get_lr_cosine_schedule(
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
        warmup_iters (int): T_w, the number of iterations to linearly warm up
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


def scaled_dot_product_attention(
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
        mask = mask.to(Q.device)
        logits = logits.masked_fill(~mask, float('-inf'))
    prob = softmax(logits, -1)
    return einsum(prob, V, "... queries keys, ... keys d_v -> ... queries d_v")
