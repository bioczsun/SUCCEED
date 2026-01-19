import torch
import torch.nn.functional as F

from typing import Optional, Tuple 
from torch import nn
import random
import numpy as np
import math
from einops.layers.torch import Rearrange
from config import ModelArgs

from pretrain.layers import TransformerBlock,TargetLengthCrop,ConvBlock,map_values
class SUCCEED_EPI(nn.Module):
    def __init__(self,args: ModelArgs):
        super().__init__()

        self.args = args
        self.dim = args.dim
        self.track_trunk = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=19,padding=9),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(8),
            
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),

            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4), 

            nn.Conv1d(in_channels=64,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4), 
        )
        self.conv_end = nn.Conv1d(512,256,1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(2)])

        # Final pointwise convolution
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(256, 512, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.add_heads(**args.output_heads)

        self.target_length = args.target_seq_len
        self.crop_final = TargetLengthCrop(self.target_length)

    def add_heads(self, **kwargs):
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim, features),
            nn.Softplus()
        ), kwargs))

    @property
    def heads(self):
        return self._heads

    def forward(self,x,emb):
        # Forward pass; embedding is passed explicitly.
        # emb = emb.transpose(1,2)
        x = self.track_trunk(x)
        x = torch.cat([x,emb],dim=1)
        x = self.conv_end(x).transpose(1,2)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.crop_final(x)
        out = map_values(lambda fn: fn(x), self._heads)
        return out

class LLM_EPI(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.args = ModelArgs()
        self.dim = 256
        self.emb_dim = emb_dim
        self.output_heads = {"human":46}

        self.track_trunk = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=19,padding=9),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(8),
            
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),

            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4), 

            nn.Conv1d(in_channels=64,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4), 
        )
        self.conv_end = nn.Conv1d(256+self.emb_dim,256,1)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.args) for _ in range(2)])
        self.add_heads(**self.output_heads)


    def add_heads(self, **kwargs):
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim, features),
            nn.Softplus()
        ), kwargs))

    @property
    def heads(self):
        return self._heads

    def forward(self,x,emb):
        # Forward pass; embedding is passed explicitly.
        x = self.track_trunk(x)
        # emb = emb.transpose(1,2)
        x = torch.cat([x,emb],dim=1)
        x = self.conv_end(x).transpose(1,2)
        for block in self.transformer_blocks:
            x = block(x)
        out = map_values(lambda fn: fn(x), self._heads)
        return out
