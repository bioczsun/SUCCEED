import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import numpy as np
import math

# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

# Residual connection wrapper
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Convolution block    
def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2) # type: ignore
    )


class RMSNorm(nn.Module):
    """RMS Normalization module, normalizes input based on L2 norm without mean subtraction."""

    def __init__(self, dim=512, norm_eps=1e-5):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """RMS normalization: x / sqrt(mean(x^2) + eps)"""
        rms = torch.mean(x**2, dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex representation of rotary position embeddings.

    Args:
        dim: dimension of rotary embedding (head_dim)
        seq_len: sequence length
        device: computation device (cuda/cpu)
        theta: scaling factor for rotary embeddings

    Returns:
        [seq_len, head_dim] tensor of complex rotation matrices
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape `freqs_cis` to be broadcastable to input x.

    Args:
        freqs_cis: rotary embeddings [seq_len, head_dim]
        x: input tensor

    Returns:
        reshaped `freqs_cis`
    """
    assert x.ndim >= 2, "Input tensor must have ndim >= 2"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis must match [seq_len, head_dim]"
    
    shape = [1] * x.ndim
    shape[1], shape[-1] = freqs_cis.shape
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key vectors.

    Args:
        xq: query tensor (batch, seq_len, num_heads, head_dim)
        xk: key tensor (batch, seq_len, num_heads, head_dim)
        freqs_cis: precomputed rotary embeddings (seq_len, head_dim)

    Returns:
        rotated xq, xk
    """
    device = xq.device

    def to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    xq_, xk_ = to_complex(xq).to(device), to_complex(xk).to(device)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


# Expand key/value heads if fewer than query heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return x.unsqueeze(3).expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA):
    - Uses `n_kv_heads` key/value heads, while query heads = `num_heads`.
    - Query heads share the same K/V.
    - Efficient attention used in models like LLaMA2.
    """

    def __init__(self, dim, num_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads or self.num_heads
        self.head_dim = self.dim // self.num_heads
        self.repeat = self.num_heads // self.n_kv_heads

        # Linear projections (no bias for efficiency)
        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        # Precompute RoPE
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len, "cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Compute Q, K, V
        2. Apply RoPE
        3. Repeat K/V to match Q
        4. Compute scaled dot-product attention
        5. Project back to `dim`
        """
        bsz, seq_len, _ = x.shape
        device = x.device

        xq = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        freqs_cis = self.freqs_cis[:seq_len].to(device)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        keys = repeat_kv(xk, self.repeat)
        values = repeat_kv(xv, self.repeat)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, values)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """Gated feed-forward network (SwiGLU-style)."""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int) -> None:
        super().__init__()
        hidden_dim = int((2 / 3) * hidden_dim)
        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN: SiLU(x @ W1) ⊙ (x @ W3) @ W2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class TransformerBlock(nn.Module):
    """Standard Transformer Block with GQA and gated FFN."""

    def __init__(self, dim:int, num_heads:int, n_kv_heads:int, max_seq_len:int, norm_eps:float=1e-5, multiple_of:int=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of

        assert dim % multiple_of == 0, f"dim ({dim}) must be multiple of {multiple_of}"
        assert num_heads > 0
        assert num_heads % n_kv_heads == 0
        assert max_seq_len > 0
        assert 0 < norm_eps < 1

        self.attention_norm = RMSNorm(self.dim, norm_eps=self.norm_eps)
        self.ff_norm = RMSNorm(self.dim, norm_eps=self.norm_eps)

        self.attn = GroupedQueryAttention(self.dim, self.num_heads, self.n_kv_heads, max_seq_len)

        hidden_dim = (self.dim * 2)
        hidden_dim = (hidden_dim + self.multiple_of - 1) // self.multiple_of * self.multiple_of
        assert hidden_dim % multiple_of == 0

        self.ffn = FeedForward(
            dim=self.dim,
            hidden_dim=hidden_dim,
            multiple_of=self.multiple_of
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer block forward pass"""
        residual = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.ff_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x


def exponential_linspace_int(start, end, num, divisible_by=1):
    """
    Generate exponentially spaced integers between start and end.

    Args:
        start: start value
        end: end value
        num: number of values
        divisible_by: ensure each integer is divisible by this value
    """
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]
    

class Conv1dBlock(nn.Module):
    """1D Convolutional tower with residual connections and pooling."""
    def __init__(self, in_channels, out_channels=256, kernel_size=19):
        super().__init__()
        self.conv_start = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(8)
        )
        hidden = [128,32,64,256]
        pool_size = [2,2,2]
        conv_blocks = []
        for i in range(len(hidden)):
            if i < len(hidden) - 1:
                dim_in = hidden[i]
                dim_out = hidden[i+1]
                pool = pool_size[i]

                conv_blocks.append(
                    nn.Sequential(
                        ConvBlock(dim_in, dim_out, kernel_size=5),
                        Residual(ConvBlock(dim_out, dim_out, 1)),
                        nn.MaxPool1d(pool),
                    )
                )
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.avg_pool = nn.AdaptiveAvgPool1d(1024)

    def forward(self, x: torch.Tensor):
        # stem conv
        x = self.conv_start(x)
        # convolutional tower
        x = self.conv_blocks(x)
        x = self.avg_pool(x)
        return x
    

class TargetLengthCrop(nn.Module):
    """
    Crop input sequence to target length.

    Args:
        target_length (int): target length. If -1, no cropping.
    """
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')
        trim = (seq_len - target_len) // 2
        if trim == 0:
            return x
        return x[:, trim:-trim]
    

# Example test
# x = torch.randn(1, 1024, 256)
# model = TransformerBlock(256, 8, 8, 1024)
# out = model(x)
# print(out.shape)
