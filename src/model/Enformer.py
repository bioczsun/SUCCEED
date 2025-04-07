import torch
import torch.nn.functional as F

from typing import Optional, Tuple 
from torch import nn
import random
import numpy as np
import math
from einops.layers.torch import Rearrange
import sys
sys.path.append('/home/suncz/work/s02/Encode_epigenome/results/Enformer/src')
from model.config import ModelArgs


def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)

set_random_seed(1314)



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


##---------------定义高效Transformer模块-------------------------------    
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
    # 计算每对维度的Theta值，即dim/2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

    # 计算序列中位置(m)的范围
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # freqs给出序列中所有标记位置的Theta值范围
    freqs = torch.outer(t, freqs)

    # 转换为极坐标形式的旋转矩阵，以便对嵌入执行旋转
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):  
    ndim = x.ndim  # 获取x的维度数
    # 确保x的维度数大于等于2
    assert ndim >= 2, "x的维度必须大于等于2"  
    
    # 确保freqs_cis的形状与x的第二个和最后一个维度匹配
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), "freqs_cis的形状必须与x的第二个维度和最后一个维度匹配"
    
    # 调整形状以支持广播机制
    # 如果维度是第2维或最后一维，保留它们的大小，否则设置为1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状以支持广播
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 获取设备
    device = xq.device
    # 同时对查询和键嵌入应用旋转位置编码
    # 1. 首先：xq和xk嵌入的最后一个维度需要重塑为一对。因为旋转矩阵应用于每对维度。
    # 2. 其次：将xq和xk转换为复数，因为旋转矩阵只适用于复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)  # xq_:[bsz, seq_len, n_heads, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)  # xk_:[bsz, seq_len, n_heads, head_dim/2]
    
    # 旋转矩阵(freqs_cis)在seq_len(dim=1)和head_dim(dim=3)维度上应与嵌入匹配
    # 此外，freqs_cis的形状应与xq和xk相同，因此将freqs_cis的形状从[seq_len, head_dim]改变为[1, seq_len, 1, head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 最后，通过与freqs_cis相乘执行旋转操作。
    # 旋转完成后，将xq_out和xk_out转换回实数并返回
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device)  # xq_out:[bsz, seq_len, n_heads, head_dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device)  # xk_out:[bsz, seq_len, n_heads, head_dim]
    
    # 返回结果，使用type_as确保输出类型和输入一致
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
        # # 预计算旋转编码 (避免重复计算)
        # self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, self.args.max_seq_len, "cpu"))

    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape
        device = x.device

        # 计算 Query, Key, Value
        xq = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 旋转位置编码 (RoPE)
        freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len, device=device) # type: ignore
        # freqs_cis = self.freqs_cis[:seq_len].to(device)  # 确保 RoPE 计算的序列长度匹配当前 `seq_len`
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
        

# 如果键/值头的数量少于查询头,此函数使用所需的重复次数扩展键/值嵌入  
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    return x.unsqueeze(3).expand(bsz, seq_len, n_kv_heads, n_rep, head_dim).reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]) -> None:
        super().__init__()
        # 模型嵌入维度
        self.dim = dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 确保 hidden_dim 是 multiple_of 的倍数

        # 定义隐藏层权重
        self.w1 = nn.Linear(self.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        #定义RMSNorm
        self.attention_norm = RMSNorm(args)
        #初始化注意力
        self.attn = GroupedQueryAttention(args)
        #定义前馈网络的RMSNorm
        self.ff_norm = RMSNorm(args)
        #初始化前馈网络
        self.ffn = FeedForward(dim=args.dim, hidden_dim=args.multiple_of, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier)

    def forward(self, x: torch.Tensor):
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
    
####--------------Transformer end---------------------

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

class Enformer(nn.Module):
    def __init__(self, args: ModelArgs,return_emb=False):
        super().__init__()
        self.args = args
        self.dim = args.dim
        half_dim = self.dim // 2
        twice_dim = self.dim * 2
        self.filter_size = 19
        self.return_emb=return_emb

        
        # 创建 stem 卷积层
        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, self.filter_size, padding=self.filter_size // 2),
            nn.BatchNorm1d(half_dim),
            Residual(ConvBlock(half_dim)),
            nn.MaxPool1d(8),
        )
        ##对于131072，使用2,对于524k，使用4。其余分辨率皆用4
        # 创建卷积塔
        filter_list = exponential_linspace_int(half_dim, self.dim, num=args.num_downsamples, divisible_by=args.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        conv_layers = [
            nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                nn.MaxPool1d(2),
            )
            for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:])
        ]

        self.conv_tower = nn.Sequential(*conv_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1024)

        transformer = [TransformerBlock(args) for _ in range(args.depth)]
        self.transformer = nn.Sequential(*transformer)

        # 目标裁剪
        self.target_length = args.target_seq_len
        self.crop_final = TargetLengthCrop(self.target_length)

        # 最终逐点卷积
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
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        if self.return_emb:
            return x
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return {name: head(x) for name, head in self.heads.items()}
    


# for name, param in model.named_parameters():
#     if 'head' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

# print(f"Trainable parameters: {trainable_params}")
# print(f"Non-trainable parameters: {non_trainable_params}")

# # print(model)
# output = model(a)
# print(output["human"])
# np.savez('/home/suncz/work/s02/Basenji2/Figure/Results/1/emb.npz',emb=emb.cpu().detach().numpy())
# 冻结除了final_pointwise以外的所有层


# # 验证冻结的参数
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# print(model)
# output = model(a)
# print(output.shape)











            