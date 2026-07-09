import torch
from typing import Optional


class ModelArgs:
    def __init__(
        self,
        dim: int = 256,  # dimension of the model
        num_heads: int = 8,  # number of heads
        n_kv_heads: int = 8,  # number of key-value heads
        stem_windows: int = 8,  # stem window size
        pool_windows: int = 2,  #
        avg_pool: bool = True,  # whether to use average pooling
        output_heads: Optional[dict] = None,  # number of output heads
        depth: int = 11,  # depth of the model
        multiple_of: int = 512,  # dimension of feedforward network
        ffn_dim_multiplier: Optional[float] = None,  # multiplier for feedforward network dimension
        norm_eps: float = 1e-5,  # epsilon for RMSNorm
        dropout_rate: float = 0,  # dropout rate
        rope_theta: float = 10000.0,  # theta for ROPE
        max_batch_size: int = 64,  # maximum batch size
        max_seq_len: int = 1024,  # maximum sequence length
        target_seq_len: int = 896,  # target sequence length
        num_downsamples: int = 4,  # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        dim_divisible_by: int = 128, # dimension divisible by < dim
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',  # device type
        **kwargs,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.stem_windows = stem_windows
        self.pool_windows = pool_windows
        self.avg_pool = avg_pool
        # Avoid mutable default args (dict) being shared across instances.
        self.output_heads = output_heads if output_heads is not None else {"human": 5421}
        self.depth = depth
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.rope_theta = rope_theta
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.device = device
        self.target_seq_len = target_seq_len
        assert self.dim % self.dim_divisible_by == 0, 'dimension must be divisible by dim_divisible_by'
        assert self.dim > self.dim_divisible_by, 'dimension must be greater than dim_divisible_by'

        # Preserve forward-compatibility for any extra kwargs without crashing.
        for k, v in kwargs.items():
            setattr(self, k, v)