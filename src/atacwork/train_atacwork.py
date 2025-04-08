"""
python src/atacwork/train_atacwork.py \
    --project_dir /path/to/SUCCEED \
    --name ATACwork \
    --use_pth data/model/131k_corr_weight_10.10.1_best_network.pth \
    --batch 32 \
    --lr 0.0001 \
    --dataset_dir result/denoise/bulk \
    --outpath result/denoise/bulk/train_model/ \
    --cell_types "CD8-10" "Bcell-13" "CD4-9" "Nkcell-11"
"""
import sys
import argparse

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='train model')
    # params.add_argument("--td",help="train data",required=True)
    # params.add_argument("--vd",help="valid data",required=True)
    # params.add_argument("--atd",help="peaks file",required=True)
    # params.add_argument("--avd",help="peaks file",required=True)
    # params.add_argument("--res",help="model res",required=True,type=int)
    params.add_argument("--project_dir",help="project dir",required=True)
    params.add_argument("--name",help="model name",required=True)
    params.add_argument("--use_pth",help="Enformer model file",required=False)
    params.add_argument("--optimizer",help="optimizer file",required=False)
    params.add_argument("--seed",help="random seed",default=1401,type=int)
    params.add_argument("--device",help="device cuda",default="cuda")
    params.add_argument("--epoch",help="epochs",default=50000)
    params.add_argument("--lr",help="learn rata",default=0.0001)
    params.add_argument("--batch",help="batch size",default=4)
    params.add_argument("--outpath",help="model name")
    params.add_argument("--dataset_dir",help="directory containing data folders",required=True)
    params.add_argument("--cell_types",help="list of cell types to use",nargs='+',default=["CD8-10" "Bcell-13" "CD4-9" "Nkcell-11"])
    return params.parse_args()

# Parse arguments and add path before importing other modules
args = parse_arguments()
project_dir = args.project_dir
sys.path.append(project_dir + '/src')

from pathlib import Path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import Dataset, DataLoader
from torchmetrics import AUROC
import random
import h5py
import numpy as np
from tqdm import tqdm
import os
from model.Enformer import Enformer
from model.config import ModelArgs
from training.metric import EarlyStopping, compute_rowwise_pearson
import atacwork

import argparse


config_args = ModelArgs()

def set_random_seed(random_seed = 40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)

set_random_seed(args.seed)

class EpiDataset(Dataset):
    def __init__(self, data_files, train=True):
        self.data_files = data_files
        self.train = train

        self.cumulative_lengths = [0]  # cumulative lengths
        self.file_handles = []  # 存储打开的文件句柄

        # Calculate length and range for each file
        for data_file in self.data_files:
            # 打开文件并保存句柄
            h5_file = h5py.File(data_file, 'r')
            self.file_handles.append(h5_file)
            
            # 获取文件长度
            length = len(h5_file['sequence'])
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)

    def __len__(self):
        return self.cumulative_lengths[-1]  # Total length is the last cumulative length

    def locate_file_and_index(self, idx):
        """Locate specific file and line number within the file"""
        for file_idx, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                file_index = idx - start
                return file_idx, file_index
        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        # Locate specific file and line number
        file_idx, file_index = self.locate_file_and_index(idx)
        
        # 使用已打开的文件句柄直接读取数据
        h5_file = self.file_handles[file_idx]
        
        # Read data from file
        seq = h5_file['sequence'][file_index]
        clean = h5_file['clean'][file_index]
        noisy = h5_file['noisy'][file_index]
        label = h5_file['label'][file_index]
        
        return torch.Tensor(seq), torch.Tensor(clean), torch.Tensor(noisy), torch.Tensor(label)
    
    def __del__(self):
        # 关闭所有打开的文件句柄
        for handle in self.file_handles:
            try:
                handle.close()
            except:
                pass

# define train and val data    
batch_size = int(args.batch)
epochs = int(args.epoch)
device = args.device if torch.cuda.is_available() else "cpu"



## create Enformer
config_args = ModelArgs()
config_args.device = device
config_args.output_heads = {"human": 6389}
model_epi = Enformer(config_args, return_emb=True)
model_epi.to(device)


if args.use_pth:
    checkpoint = torch.load(args.use_pth,map_location="cpu",weights_only=True)
    model_epi.load_state_dict(checkpoint['model_state_dict'])
    model_epi.to(device)


## Freeze all parameters in model_epi
for param in model_epi.parameters():
    param.requires_grad = False

model = atacwork.ATACwork(config_args)

# for name, param in model.named_parameters():

#     if "stem" in name or "conv_tower" in name:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True

# # for name, param in model.named_parameters():
# #     if "_heads" in name:
# #         param.requires_grad = True
# #     else:
# #         param.requires_grad = False

# # Verify frozen parameters
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, requires_grad: {param.requires_grad}")





if args.optimizer is None:
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(args.lr))
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args.lr))
    # optimizer = torch.optim.AdamW(list(model.parameters()) + list(model_epi.parameters()), lr=float(args.lr))



# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model_epi = nn.DataParallel(model_epi)  # Wrap model in DataParallel
#     model = nn.DataParallel(model)  # Wrap model in DataParallel

model.to(device)  # Load model to the main device

# res = args.res

## create dataloader
dataset_dir = Path(args.dataset_dir)
train_data = []
val_data = []

# Collect train and validation data files for specified cell types
for cell_type in args.cell_types:
    cell_type_dir = dataset_dir / cell_type 
    if cell_type_dir.exists():
        train_data.extend([f for f in cell_type_dir.iterdir() if f.name.startswith("train_") and f.is_file()])
        val_data.extend([f for f in cell_type_dir.iterdir() if f.name.startswith("valid_") and f.is_file()])
    else:
        print(f"Warning: Directory {cell_type_dir} does not exist")

print(f"Found {len(train_data)} training files and {len(val_data)} validation files")

# Create datasets
train_dataset = EpiDataset(
    data_files=train_data,
    train=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  
    num_workers=4,  # 使用多进程加载数据
    pin_memory=True  # 加速GPU训练
)

val_dataset = EpiDataset(
    data_files=val_data,
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
# val_dataset = EpiDataset(val_data,atac_data=val_atac_data,augment=False,train=False)
# val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=20,pin_memory=True,persistent_workers=True)


## create optimizer and loss function
# optimizer = torch.optim.AdamW(model.parameters(),lr=float(args.lr))

## create metric
# loss_poss =  nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
loss_fn = nn.MSELoss()
loss_bce = nn.BCELoss()

## create metric
# corr_coef_train = PearsonR(num_targets=batch_size).to(device)

# corr_coef_test = PearsonR(num_targets=batch_size).to(device)
auroc = AUROC(task='multilabel', num_labels=1024).to(device)


model_save_path = args.name
early_stopping = EarlyStopping(save_path=args.outpath,model_name=model_save_path,patience=5)
# create save path
os.makedirs(args.outpath, exist_ok=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8, min_lr=1e-5)



r_tra = 0
r_val = 0

def evaluate(model, val_loader,epoch,epochs):
    model.eval()
    model_epi.eval()
    total_loss = 0
    total_r = 0
    tmp_r = 0
    total_auroc = 0
    par = tqdm(val_loader,bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    # corr_coef_test.reset()
    with torch.no_grad():
        for i, (seq, clean,noisy,label) in enumerate(par):
            seq = seq.to(device).float().transpose(1,2)
            sample = noisy.to(device).float().unsqueeze(1)
            target = clean.to(device)
            label = label.to(device)

            embed = model_epi(seq)
            out = model(sample,embed)

            signal = out[0]
            peak = out[1]
            origin_mean = out[2]

            # signal = torch.log2(signal + 1)
            # target = torch.log2(target + 1)
            
            # corr_coef_test.update(signal.to(device), target.to(device))
            r_val = compute_rowwise_pearson(signal, target).mean()
            r_val_tmp = compute_rowwise_pearson(target, origin_mean).mean()
            #corr_coef_test.compute().mean()#pearson_r(out.to(device), target.to(device))
            auroc_val = auroc(peak.detach(),label.detach().long())
            loss_mse = loss_fn(signal, target)
            loss_binary = loss_bce(peak,label)
            # loss_mse = loss_mse_fn(out, target)
            loss = loss_mse + loss_binary
            total_loss += loss.item()
            total_r += r_val
            tmp_r += r_val_tmp
            total_auroc += auroc_val
            if i % 50 == 0:
                par.set_description(f"Valid--Epoch [{epoch} / {epochs}] Loss_mse {loss_mse.item():.3f} Loss_bce {loss_binary.item():.3f} R {r_val:.3f} auc {auroc_val:.3f}",refresh=True)
        avg_loss = total_loss / len(val_loader)
        avg_r = total_r / len(val_loader)
        avg_tmp_r = tmp_r / len(val_loader)
        avg_auroc = total_auroc / len(val_loader)
        return avg_loss, avg_r, avg_auroc, avg_tmp_r



#train
for epoch in range(epochs):
    model.train()
    total_loss = 0
    par = tqdm(train_loader, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    for i, (seq, clean,noisy,label) in enumerate(par):
        # corr_coef_train.reset()
        target = clean.to(device)
        label = label.to(device)
        # ================================
        # 1. Train with original sequence
        # ================================
        seq = seq.to(device).float().transpose(1,2)
        sample = noisy.to(device).float().unsqueeze(1)
        optimizer.zero_grad()

        embed = model_epi(seq)
        out = model(sample,embed) # Forward pass
        signal = out[0]
        peak = out[1]
        origin_mean = out[2]

        loss_mse = loss_fn(signal, target)
        loss_binary = loss_bce(peak,label)
        r_tra_orig = compute_rowwise_pearson(signal, target).mean()
        r_train_tmp = compute_rowwise_pearson(target, origin_mean).mean()

        # Calculate Pearson correlation
        # corr_coef_train.update(signal.to(device), target.to(device))
        #corr_coef_train.compute().mean()

        loss = loss_mse + loss_binary

        loss.backward()  # Backpropagation
        # Gradient clipping
        # clip_grad.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()  # Update model parameters
        



        
        
        # Accumulate total loss (only for printing and observation, doesn't affect gradient updates)
        total_loss += loss.item()

        if i % 50 == 0:
            # Calculate AUROC
            train_auroc = auroc(peak.detach(),label.detach().long()).mean()
            current_lr = optimizer.param_groups[0]['lr']
            par.set_description(f"Train--Epoch: [{epoch} / {epochs}] "
                                f"Loss Orig: {loss.item():.3f} "
                                f"Loss bce: {loss_binary.item():.3f} "
                                f"R Orig: {r_tra_orig:.3f} "
                                f"auroc: {train_auroc:.3f} "
                                # f"R Shifted: {r_tra_shifted:.3f} "
                                f"tmp_r: {r_train_tmp:.3f}")

    # ================================
    # Validate the model
    # ================================
    val_loss, val_r,val_auroc,tmp_r = evaluate(model, val_loader, epoch, epochs)
    
    print(f"Epoch: [{epoch} / {epochs}] "
          f"Train Loss: {(total_loss / len(train_loader)):.3f} "
          f"Val Loss: {val_loss:.3f} "
          f"Val R: {val_r:.3f} "
          f"auroc: {val_auroc:.3f} "
          f"tmp_r: {tmp_r:.3f}"
          )

    # Save model
    if isinstance(model, nn.DataParallel):
        early_stopping(-val_r, model.module, optimizer)  # Save original model
    else:
        early_stopping(-val_r, model, optimizer)

    scheduler.step(-val_r)

    if early_stopping.early_stop:
        print("Early stopping")
        break