import sys
sys.path.append("/home/suncz/work/s02/Encode_epigenome/results/Enformer/src")
from model.layers import Enformer,exponential_linspace_int,TransformerBlock,TargetLengthCrop

import torch
import torch.nn as nn
from model.config import ModelArgs

args = ModelArgs()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##辅助函数
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

class HistonTF(nn.Module):
    def __init__(self,args,enformer_checkpoint=None, num_classes=46): # 19 classes for histone modification labels
        super(HistonTF, self).__init__()
        # Load the pre-trained Enformer model
        self.args = args
        args.output_heads = {"human": 5421}
        self.dim = args.dim
        self.filter_size = 19
        self.enformer = Enformer(args,return_emb= True)
        ##加载预训练的Enformer模型
        if enformer_checkpoint is not None:
            checkpoint = torch.load(enformer_checkpoint)
            self.enformer.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded Enformer checkpoint from {}".format(enformer_checkpoint))
        ##冻结Enformer的参数
        for param in self.enformer.parameters():
            param.requires_grad = False
        self.res_blocks_epi = nn.Sequential(
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1024)
        self.conv_end = nn.Conv1d(512,256,1)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(2)])
        self.add_heads(**{"num_classes": num_classes})
        


    def forward(self,seq,epi):
        seq = self.enformer(seq).transpose(1,2)
        epi = self.res_blocks_epi(epi)
        epi = self.avg_pool(epi)
        x = torch.cat([epi, seq], dim=1)
        x = self.conv_end(x).transpose(1,2)
        for block in self.transformer_blocks:
            x = block(x)
        out = map_values(lambda fn: fn(x), self._heads)
        return out
    
    # def get_res_blocks(self,dim,half_dim,filter_size):
    #     # 创建 stem 卷积层
    #     stem = nn.Sequential(
    #         nn.Conv1d(1, half_dim, filter_size, padding=filter_size // 2),
    #         nn.BatchNorm1d(half_dim),
    #         Residual(ConvBlock(half_dim)),
    #         nn.MaxPool1d(8),
    #     )
    #             # 创建卷积塔
    #     filter_list = exponential_linspace_int(half_dim, dim, num=args.num_downsamples, divisible_by=args.dim_divisible_by)
    #     filter_list = [half_dim, *filter_list]

    #     conv_layers = [
    #         nn.Sequential(
    #             ConvBlock(dim_in, dim_out, kernel_size=5),
    #             Residual(ConvBlock(dim_out, dim_out, 1)),
    #             nn.MaxPool1d(4),
    #         )
    #         for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:])
    #     ]

    #     # 将self.stem和self.conv_tower连接起来
    #     res_blocks_epi = nn.Sequential(stem,*conv_layers)

    #     return res_blocks_epi
    
    def add_heads(self, **kwargs):
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim, features),
            nn.Softplus()
        ), kwargs))

    @property
    def heads(self):
        return self._heads
seq = torch.randn(1,4,1048576)    
x = torch.randn(1,1,1048576)
model = HistonTF(args)
y = model(seq,x)["num_classes"]
print(y.shape)



        

        
