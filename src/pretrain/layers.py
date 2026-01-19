import torch
import torch.nn.functional as F

from typing import Optional, Tuple 
from torch import nn
import random
import numpy as np
import math
from einops.layers.torch import Rearrange
from pretrain.config import ModelArgs



# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

# residual connection wrapper
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# convolution block
def ConvBlock(dim,dim_out = None, kernel_size = 1):

    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2) # type: ignore
    )


# --------------- Efficient Transformer modules -----------------
class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.eps = args.norm_eps
        self.weight = nn.Parameter(torch.ones(args.dim))

    def _norm(self,x):
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x) * self.weight
        return output
    

def precompute_freqs_cis(dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # compute theta values for every pair of dims (dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

    # position indices along the sequence
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # outer product to get all (position, theta) combinations
    freqs = torch.outer(t, freqs)

    # convert to complex numbers in polar form for rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):  
    ndim = x.ndim  # number of dims of x
    # ensure x has at least 2 dims
    assert ndim >= 2, "x must have ndim >= 2"  
    
    # ensure freqs_cis matches x's second and last dims
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis must match x at [seq_len, head_dim]"
    
    # reshape for broadcasting:
    # keep sizes for the 2nd and last dims, set others to 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # view freqs_cis to the broadcastable shape
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # current device
    device = xq.device
    # apply rotary embeddings to queries and keys
    # 1) reshape last dim into pairs (because rotation acts on 2D subspaces)
    # 2) convert to complex numbers so rotations are simple multiplications
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)  # xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)  # xk_:[bsz, seq_len, n_heads, head_dim/2]
    
    # freqs_cis must match seq_len (dim=1) and head_dim (dim=3)
    # change from [seq_len, head_dim] to [1, seq_len, 1, head_dim] for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # rotate by complex multiplication, then convert back to real
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device)  # xq_out:[bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device)  # xk_out:[bsz, seq_len, n_heads, head_dim]
    
    # ensure dtype matches input
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.num_heads = args.num_heads
        self.n_kv_heads = self.num_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = self.dim // self.num_heads
        self.repeat = self.num_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)
        # # precompute RoPE (avoid recomputation)
        # self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, self.args.max_seq_len, "cpu"))

    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        device = x.device

        # compute Q, K, V
        xq = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # rotary positional embeddings (RoPE)
        # Important: freqs_cis must match the *current* seq_len, otherwise reshape_for_broadcast will assert.
        freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=seq_len, device=device, theta=self.args.rope_theta)  # type: ignore
        # freqs_cis = self.freqs_cis[:seq_len].to(device)  # ensure RoPE length matches current seq_len
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # repeat K/V to match number of Q heads
        keys = repeat_kv(xk, self.repeat)
        values = repeat_kv(xv, self.repeat)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, values)

        # restore shape & project back to `dim`
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(output)

# if the number of K/V heads is fewer than Q heads, repeat K/V along heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return x.unsqueeze(3).expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> None:
        super().__init__()
        # model embedding dimension
        self.dim = dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # ensure multiple-of alignment

        # feedforward weights
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # RMSNorm before attention
        self.attention_norm = RMSNorm(args)
        # attention module
        self.attn = GroupedQueryAttention(args)
        # RMSNorm before FFN
        self.ff_norm = RMSNorm(args)
        # feedforward network
        self.ffn = FeedForward(dim=args.dim, hidden_dim=args.multiple_of, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)

    def forward(self, x: torch.Tensor):
        # residual + norm + attention
        residual = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + residual  # residual connection

        # residual + norm + feedforward
        residual = x
        x = self.ff_norm(x)
        x = self.ffn(x)
        x = x + residual  # residual connection

        return x
    
# -------------- Transformer end ---------------------

def exponential_linspace_int(start, end, num, divisible_by=1):
    """
    Generate an exponentially spaced integer sequence.

    Args:
        start (int): start value
        end (int): end value
        num (int): number of integers
        divisible_by (int): each integer will be rounded to a multiple of this

    Returns:
        List[int]: exponentially spaced integers, each divisible by `divisible_by`.
    """
    def _round(x):
        # round to nearest multiple of `divisible_by`
        return int(round(x / divisible_by) * divisible_by)

    # compute the exponential base
    base = math.exp(math.log(end / start) / (num - 1))
    
    # generate the list from start to end
    return [_round(start * base**i) for i in range(num)]

    
class TargetLengthCrop(nn.Module):
    """
    Module to crop an input sequence to a target length.

    Args:
        target_length (int): desired length. If -1, no cropping.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Crop the last-but-one dim of x to `target_length`.
    """
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        # no cropping if target length is -1
        if target_len == -1:
            return x

        # if input is shorter than target, raise
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        # compute trim size
        trim = (seq_len - target_len) // 2

        # if no need to crop, return as-is
        if trim == 0:
            return x

        # crop and return
        return x[:, trim:-trim]

class SUCCEED(nn.Module):
    def __init__(self, args: ModelArgs,return_emb=False):
        super().__init__()
        self.args = args
        self.dim = args.dim
        half_dim = self.dim // 2
        twice_dim = self.dim * 2
        self.filter_size = 19
        self.return_emb=return_emb

        
        # stem convolutional block
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, self.filter_size, padding=self.filter_size // 2),
            nn.BatchNorm1d(half_dim),
            Residual(ConvBlock(half_dim)),
            nn.MaxPool1d(args.stem_windows),
        )
        # for 131k and 524k use 2; otherwise use 4
        # convolutional tower
        filter_list = exponential_linspace_int(half_dim, self.dim, num=args.num_downsamples, divisible_by=args.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = [
            nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                nn.MaxPool1d(self.args.pool_windows),
            )
            for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:])
        ]

        self.conv_tower = nn.Sequential(*conv_layers)

        if args.avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool1d(1024)

        transformer = [TransformerBlock(args) for _ in range(args.depth)]
        self.transformer = nn.Sequential(*transformer)

        # target cropping
        self.target_length = args.target_seq_len
        self.crop_final = TargetLengthCrop(self.target_length)

        # final pointwise conv
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.add_heads(**args.output_heads)

    def add_heads(self, **kwargs):
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    @property
    def heads(self):
        return self._heads

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.conv_tower(x)
        if self.args.avg_pool:
            x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        if self.return_emb:
            x = x.transpose(1, 2)
            return x
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return {name: head(x) for name, head in self.heads.items()}
