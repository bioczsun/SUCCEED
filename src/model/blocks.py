import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import numpy as np
import math

#辅助函数
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

#定义残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

##定义卷积块    
def ConvBlock(dim,dim_out = None, kernel_size = 1):

    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2) # type: ignore
    )



class RMSNorm(nn.Module):
    """RMS Normalization 模块, 避免层归一化的均值计算, 仅对输入的L2范数归一化"""
    
    def __init__(self,dim = 512,norm_eps = 1e-5):
        super().__init__()
        self.eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算RMS归一化: x / sqrt(mean(x^2) + eps)"""
        rms = torch.mean(x**2, dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return x / rms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    """
    计算旋转位置编码的复数表示
    :param dim:  旋转编码维度 (head_dim)
    :param seq_len: 序列长度
    :param device: 计算设备 (cuda/cpu)
    :param theta: 位置编码的缩放因子，默认10000
    :return: [seq_len, head_dim] 复数形式的旋转矩阵
    """
    # 计算频率: 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    # 位置索引
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # 计算旋转角度: [seq_len, head_dim/2]
    angles = torch.outer(positions, freqs)
    # 转换为复数形式: (cosθ + j sinθ)
    return torch.polar(torch.ones_like(angles), angles)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    调整 `freqs_cis` 形状以支持广播机制
    :param freqs_cis: 旋转编码的复数形式 [seq_len, head_dim]
    :param x: 需要应用旋转的输入张量
    :return: 调整形状后的 `freqs_cis`
    """
    assert x.ndim >= 2, "输入张量 `x` 维度必须 >= 2"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis 形状必须匹配 x 的 [seq_len, head_dim]"
    
    # 生成适用于广播的形状
    shape = [1] * x.ndim
    shape[1], shape[-1] = freqs_cis.shape  # 只在 seq_len 和 head_dim 维度上保留
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对查询向量 xq 和键向量 xk 应用旋转位置编码
    :param xq: 查询向量 (batch_size, seq_len, num_heads, head_dim)
    :param xk: 键向量 (batch_size, seq_len, num_heads, head_dim)
    :param freqs_cis: 预计算好的旋转位置编码 (seq_len, head_dim)
    :return: 旋转后的 xq, xk
    """
    # 确保设备匹配
    device = xq.device

    # 将 (head_dim) 拆分为 (head_dim/2, 2)，转换为复数形式
    def to_complex(x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    xq_, xk_ = to_complex(xq).to(device), to_complex(xk).to(device)

    # 适配广播
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 进行旋转
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out

# 如果键/值头的数量少于查询头,此函数使用所需的重复次数扩展键/值嵌入  
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return x.unsqueeze(3).expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) 机制:  
    - 使用 `n_kv_heads` 个 Key/Value 头，而 Query 头的数量是 `num_heads`。
    - Query 头会共享 `n_kv_heads` 计算得到的 Key/Value。
    - 适用于 LLaMA2 等高效 Transformer 结构。

    参数:
        args: 训练超参数 (包括 dim, num_heads, n_kv_heads, max_seq_len)
    """

    def __init__(self, dim, num_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads or self.num_heads  # 默认情况下 Key/Value 头的数量与 Query 头相同
        self.head_dim = self.dim // self.num_heads
        self.repeat = self.num_heads // self.n_kv_heads  # 每个 KV 头需要重复多少次

        # 线性投影层 (不使用 bias 以提高计算效率)
        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        # 预计算旋转编码 (避免重复计算)
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len, "cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
        1. 计算 Q, K, V
        2. 应用旋转位置编码 (RoPE)
        3. 复制 K, V 使其匹配 Q
        4. 计算注意力得分，并进行 Softmax 归一化
        5. 计算注意力输出，并通过 `wo` 投影回 `dim` 维度

        :param x: 输入张量 (batch_size, seq_len, dim)
        :return: 注意力输出 (batch_size, seq_len, dim)
        """
        bsz, seq_len, _ = x.shape
        device = x.device

        # 计算 Query, Key, Value
        xq = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 旋转位置编码 (RoPE)
        freqs_cis = self.freqs_cis[:seq_len].to(device)  # 确保 RoPE 计算的序列长度匹配当前 `seq_len`
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 复制 K/V 以匹配 Q 的数量
        keys = repeat_kv(xk, self.repeat)
        values = repeat_kv(xv, self.repeat)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力 (可选使用 `scaled_dot_product_attention` 提高效率)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, values)

        # 还原形状 & 投影回 `dim`
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int) -> None:
        super().__init__()
        # 计算 hidden_dim，确保是 multiple_of 的倍数
        hidden_dim = int((2 / 3) * hidden_dim)  # 2/3 缩减
        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of  # 取 nearest multiple_of

        # 线性层
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN: SiLU(x @ W1) ⊙ (x @ W3) @ W2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock(nn.Module):
    """标准 Transformer Block，包含 Grouped Query Attention 和 Gated Feedforward Network (Gated FFN)"""

    def __init__(self, dim:int, num_heads:int, n_kv_heads:int, max_seq_len:int,norm_eps:float=1e-5,multiple_of:int=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of

        # 参数检查
        assert dim % multiple_of == 0, f"dim ({dim}) 必须是 multiple_of ({multiple_of}) 的整数倍！"
        assert num_heads > 0, f"num_heads ({num_heads}) 必须大于 0！"
        assert num_heads % n_kv_heads == 0, f"num_heads ({num_heads}) 必须能整除 n_kv_heads ({n_kv_heads})！"
        assert max_seq_len > 0, f"max_seq_len ({max_seq_len}) 必须大于 0！"
        assert 0 < norm_eps < 1, f"norm_eps ({norm_eps}) 必须在 (0, 1) 范围内！"

        # 归一化层
        self.attention_norm = RMSNorm(self.dim,norm_eps=self.norm_eps)
        self.ff_norm = RMSNorm(self.dim,norm_eps=self.norm_eps)

        # 注意力机制（Grouped Query Attention）
        self.attn = GroupedQueryAttention(self.dim, self.num_heads, self.n_kv_heads, max_seq_len)

        # 计算 `hidden_dim`
        hidden_dim = (self.dim * 2)  # 默认 `2 × dim`
        hidden_dim = (hidden_dim + self.multiple_of - 1) // self.multiple_of * self.multiple_of  # 对齐 multiple_of
        assert hidden_dim % multiple_of == 0, f"hidden_dim ({hidden_dim}) 必须是 multiple_of ({multiple_of}) 的整数倍！"

        # 前馈网络 (FFN)
        self.ffn = FeedForward(
            dim=self.dim,
            hidden_dim=hidden_dim,
            multiple_of=self.multiple_of
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer Block 前向传播"""
        # 残差连接 + 归一化 + 注意力层
        residual = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + residual  # 残差连接

        # 残差连接 + 归一化 + 前馈网络
        residual = x
        x = self.ff_norm(x)
        x = self.ffn(x)
        x = x + residual  # 残差连接

        return x

def exponential_linspace_int(start, end, num, divisible_by=1):
    """
    生成一个在指数空间中均匀分布的整数序列。

    参数:
    start (int): 序列的起始值。
    end (int): 序列的结束值。
    num (int): 序列中整数的数量。
    divisible_by (int): 每个整数都必须是 divisible_by 的倍数，默认为 1。

    返回:
    List[int]: 一个在指数空间中均匀分布的整数序列，每个整数都是 divisible_by 的倍数。
    """
    def _round(x):
        # 将 x 除以 divisible_by，四舍五入后再乘以 divisible_by，以确保结果是 divisible_by 的倍数
        return int(round(x / divisible_by) * divisible_by)

    # 计算指数基数 base
    base = math.exp(math.log(end / start) / (num - 1))
    
    # 生成从 start 到 end 的 num 个整数，这些整数在指数空间中均匀分布，并且每个整数都是 divisible_by 的倍数
    return [_round(start * base**i) for i in range(num)]
    
class Conv1dBlock(nn.Module):
    def __init__(self,in_channels,out_channels=256,kernel_size=19):
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
        # stem 卷积层
        x = self.conv_start(x)
        # 卷积塔
        x = self.conv_blocks(x)
        x = self.avg_pool(x)
        return x
    
class TargetLengthCrop(nn.Module):
    """
    一个用于裁剪输入序列到目标长度的模块。

    参数:
    target_length (int): 目标长度。如果为 -1，则不进行裁剪。

    方法:
    forward(x: torch.Tensor) -> torch.Tensor:
        裁剪输入张量 x 到目标长度 target_length。
    """
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        # 如果目标长度为 -1，则不进行裁剪，直接返回输入
        if target_len == -1:
            return x

        # 如果输入序列长度小于目标长度，抛出异常
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        # 计算需要裁剪的长度
        trim = (seq_len - target_len) // 2

        # 如果不需要裁剪，直接返回输入
        if trim == 0:
            return x

        # 裁剪输入张量并返回
        return x[:, trim:-trim]
    
# x = torch.randn(1, 1024,256)
# model = TransformerBlock(256, 8, 8, 1024)
# out = model(x)
# print(out.shape)
    
