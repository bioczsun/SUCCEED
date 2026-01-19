import torch
from torch import nn
from einops.layers.torch import Rearrange

from pretrain.config import ModelArgs
from pretrain.layers import ConvBlock, TransformerBlock

class ATACDenoising(nn.Module):
    def __init__(self,args: ModelArgs):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.avgpool = nn.AdaptiveAvgPool1d(1024)
        self.track_trunk = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=19,padding=9),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(8),
            
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(4),

            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2), 

            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AvgPool1d(2),      
        )
        self.concat_linear = nn.Linear(512,256)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(1)])

        # Final pointwise convolution
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(256, 512, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.target_length = args.target_seq_len

        self.final_dense = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(512, 1, 1),
            Rearrange('b d n -> b n d'),
            nn.Softplus()
        )

        self.peak_dense = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(512, 1, 1),
            Rearrange('b d n -> b n d'),
            nn.Sigmoid()
        )

    def forward(self,x,emb):
        mean_x = self.avgpool(x)
        mean = mean_x.squeeze(1)
        x = self.track_trunk(x)
        x = x.transpose(1,2)
        emb = emb.transpose(1, 2)
        x = torch.cat([x,emb],dim=2)
        x = self.concat_linear(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_pointwise(x)
        out = self.final_dense(x)
        out = out.squeeze(-1)
        peak = self.peak_dense(x)
        peak = peak.squeeze(-1)
        return out,peak,mean
