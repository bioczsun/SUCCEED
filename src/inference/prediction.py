import torch
import torch.nn as nn
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import Dataset, DataLoader

import random
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("/home/suncz/work/s02/Encode_epigenome/results/Enformer/src")

from data.GenomeDataset import EnformerDataset
from model.Enformer import Enformer
from training.metric import MeanPearsonCorrCoefPerChannel
from model.blocks import TargetLengthCrop

from model.config import ModelArgs
config_args = ModelArgs()

class EpiDataset(Dataset):
    def __init__(self,data):
        self.sequence = data["sequence"]
        self.target = data["target"]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return torch.Tensor(self.sequence[idx]),torch.Tensor(self.target[idx])

def main():
    params = argparse.ArgumentParser(description='train model')
    params.add_argument("--vd",help="valid data",required=True)
    params.add_argument("--name",help="model name",required=True)
    params.add_argument("--use_pth",help="model file",required=False)
    params.add_argument("--seed",help="random seed",default=1401,type=int)
    params.add_argument("--device",help="device cuda",default="cuda")
    params.add_argument("--batch",help="batch size",default=4,type=int)
    params.add_argument("--out_dir",help="output dir",required=True)
    args = params.parse_args()

    ## set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## parameters
    device = args.device
    out_dir = args.out_dir
    model_save_path = f"{out_dir}/{args.name}.h5"

    ## load model
        
    
    # define train and val data    
    batch_size = int(args.batch)
    device = args.device if torch.cuda.is_available() else "cpu"



    ## create model
    model = Enformer(ModelArgs(device=device))
    if args.use_pth:
        checkpoint = torch.load(args.use_pth,map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    ## create dataloader
    val_data = h5py.File(args.vd,"r",swmr=True)
    # val_data = np.load(args.vd, allow_pickle=True)
    val_dataset = EpiDataset(val_data)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=10,pin_memory=True)
    valid_par = tqdm(val_loader, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    ## create target crop
    

    ## create optimizer and loss function
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    ## create metric
    corr_coef_test = MeanPearsonCorrCoefPerChannel(n_channels=config_args.output_heads["human"]).to(device)

    ## evaluate
    avg_loss, avg_pearson_r = evaluate(model,valid_par,loss_fn,corr_coef_test,device,model_save_path)
    print(f"Valid loss: {avg_loss:.3f} Pearson R: {avg_pearson_r:.3f}")

def evaluate(model,valid_data,loss_fn,corr_coef,device,out_dir):
    model.eval()
    target_crop = TargetLengthCrop(config_args.target_seq_len)
    evaluate_loss = 0
    total_r = 0
    corr_coef.reset()
    target_ls = []
    prediction_ls = []
    with torch.no_grad():
        for idx, (seq,target) in enumerate(valid_data):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)
            outputs = model(seq)["human"]  # 前向传播
            loss = loss_fn(outputs, target)  # 计算损失
            # 计算 Pearson 相关系数
            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                valid_data.set_description(f"Valid--"
                                    f"valid loss : {loss.item():.3f} "
                                    f"valid pearson_r: {pearson_r:.3f} ")
            evaluate_loss += loss.item()
            total_r += pearson_r
            target_ls.append(target.cpu().detach().numpy().astype(np.float16))
            prediction_ls.append(outputs.cpu().detach().numpy().astype(np.float16))
    ##计算平均损失
    avg_loss = evaluate_loss / len(valid_data)
    avg_pearson_r = total_r / len(valid_data)
    #释放所有显存
    torch.cuda.empty_cache()
    ##写入H5文件
    target_ls = np.concatenate(target_ls, axis=0,dtype=np.float16)
    prediction_ls = np.concatenate(prediction_ls, axis=0,dtype=np.float16)
    with h5py.File(out_dir, "w") as f:
        f.create_dataset("target", data=target_ls,dtype="float16")
        f.create_dataset("prediction", data=prediction_ls,dtype="float16")
    return avg_loss, avg_pearson_r

if __name__ == "__main__":
    main()


