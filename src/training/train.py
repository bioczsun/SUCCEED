import sys
import argparse

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='train model')
    params.add_argument("--project_dir",help="project dir",required=True)
    params.add_argument("--td",help="train data",required=True)
    params.add_argument("--vd",help="valid data",required=True)
    params.add_argument("--name",help="model name",required=True)
    params.add_argument("--use_pth",help="model file",required=False)
    params.add_argument("--seed",help="random seed",default=1401,type=int)
    params.add_argument("--device",help="device cuda",default="cuda")
    params.add_argument("--epoch",help="epochs",default=50000)
    params.add_argument("--lr",help="learn rata",default=0.0001,type=float)
    params.add_argument("--batch",help="batch size",default=4,type=int)
    params.add_argument("--out_dir",help="output dir",required=True)
    params.add_argument("--output_head_size",help="size of output head",default=1063,type=int)
    return params.parse_args()

# 在所有其他导入之前解析参数并添加路径
args = parse_arguments()
project_dir = args.project_dir
sys.path.append(project_dir + '/src')

# 然后导入其他模块
import torch
import torch.nn as nn
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import Dataset, DataLoader

import random
import h5py
import numpy as np
from tqdm import tqdm
import os
import csv
import yaml

from data.GenomeDataset import GenomeDataset

from data.GenomeDataset import GenomeDataset,EnformerDataset
from model.Enformer import Enformer
from training.metric import MeanPearsonCorrCoefPerChannel,EarlyStopping
from model.blocks import TargetLengthCrop

from model.config import ModelArgs


def main():
    ## set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## parameters
    device = args.device
    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch
    out_dir = args.out_dir

    log_dir = os.path.join(out_dir,"csv/logs")
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Get the maximum version number of the current run
    existing_versions = [int(d.split("_")[-1]) for d in os.listdir(log_dir) if d.startswith("version_")]
    new_version = max(existing_versions) + 1 if existing_versions else 0

    # Create a new version folder
    run_folder = os.path.join(log_dir, f"version_{new_version}")
    os.makedirs(run_folder)
    ##Write hyperparameters to yaml file
        # Create super parameter dictionary
    hparams = {
        "train_data": args.td,
        "valid_data": args.vd,
        "model_name": args.name,
        "use_pth": args.use_pth,
        "seed": args.seed,
        "device": device,
        "epoch": epoch,
        "learning_rate": lr,
        "batch_size": batch_size,
        "output_dir": out_dir,
        "output_head_size": args.output_head_size
    }
    with open(os.path.join(run_folder, "hparams.yaml"), 'w') as f:
        yaml.dump(hparams, f,default_flow_style=False, sort_keys=False)


    ## create dataloader
    train_data = h5py.File(args.td,"r",swmr=True)
    val_data = h5py.File(args.vd,"r",swmr=True)

    train_dataset = EnformerDataset(train_data,augment=False)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=10,pin_memory=True)


    val_dataset = EnformerDataset(val_data,augment=False)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=10,pin_memory=True)

    ## define model
    # model = Enformer(dim=256, num_heads=8, n_kv_heads=8, max_seq_len=1024, num_layers=11)
    model = Enformer(ModelArgs(
        device=device,
        output_heads=dict(human=args.output_head_size)
    ))
    print(model)

    ## use pth
    if args.use_pth:
        checkpoint = torch.load(args.use_pth,map_location="cpu")
        # model.load_state_dict(checkpoint['model_state_dict']) ## Load all parameters
        
        ##-------------------- Load model parameters, but do not load parameters starting with '_heads'-----------------------------------
        # Create a new dictionary, only keep parameters that do not start with '_heads'
        filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('_heads.')}

        # Load to model (only update parameters that match)
        model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)

    for name, param in model.named_parameters():
        if "_heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ## define optimizer and loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    ## define metric
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=args.output_head_size).to(device)

    ## define early stopping
    model_save_path = args.name
    early_stopping = EarlyStopping(save_path=out_dir,model_name=model_save_path,patience=8)

    ## train
    train(run_folder,model,train_loader,val_loader,optimizer,loss_fn,corr_coef,device,epoch,early_stopping)



def train(run_folder,model,train_data,valid_data,optimizer,loss_fn,corr_coef,device,epochs,early_stopping):

    metrics_path = os.path.join(run_folder, "metrics.csv")
    # If metrics.csv does not exist, write the header
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_pearson_r", "valid_loss", "valid_pearson_r"])

    target_crop = TargetLengthCrop(896)
    model.train()
    for epoch in range(epochs):
        train_par = tqdm(train_data, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        total_loss = 0
        total_r = 0
        corr_coef.reset()
        for idx, (seq,target) in enumerate(train_par):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)
            optimizer.zero_grad()
            outputs = model(seq)["human"]  # Forward propagation
            loss = loss_fn(outputs, target)  # Calculate loss
            loss.backward()  # Backward propagation
            clip_grad.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()

            # Calculate Pearson correlation coefficient
            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                train_par.set_description(f"train--Epoch: [{epoch} / {epochs}] "
                                    f"train loss : {loss.item():.3f} "
                                    f"train pearson_r: {pearson_r:.3f} "
                                    f"lr: {current_lr}")

            total_loss += loss.item()
            total_r += pearson_r
        avg_loss = total_loss / len(train_data)
        avg_pearson_r = total_r / len(train_data)
        valid_par = tqdm(valid_data, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        valid_loss,valid_pearson_r = evaluate(model,valid_par,loss_fn,corr_coef,device)
        # Print training and validation loss for each epoch
        print(f"Epoch: [{epoch} / {epochs}] "
            f"train loss: {avg_loss:.3f} "
            f"train pearson_r: {avg_pearson_r:.3f} "
            f"valid loss: {valid_loss:.3f} "
            f"valid pearson_r: {valid_pearson_r:.3f}")
        
        # **Append to metrics.csv**
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_pearson_r.item(), valid_loss, valid_pearson_r.item()])
        
        early_stopping(valid_loss,model,optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def evaluate(model,valid_data,loss_fn,corr_coef,device):
    target_crop = TargetLengthCrop(896)
    evaluate_loss = 0
    total_r = 0
    corr_coef.reset()
    model.eval()
    with torch.no_grad():
        for idx, (seq,target) in enumerate(valid_data):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)
            outputs = model(seq)["human"]  # Forward propagation
            loss = loss_fn(outputs, target)  # Calculate loss
            # Calculate Pearson correlation coefficient
            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                valid_data.set_description(f"Valid--"
                                    f"valid loss : {loss.item():.3f} "
                                    f"valid pearson_r: {pearson_r:.3f} ")
            evaluate_loss += loss.item()
            total_r += pearson_r
        ##Calculate average loss
        avg_loss = evaluate_loss / len(valid_data)
        avg_pearson_r = total_r / len(valid_data)
        return avg_loss, avg_pearson_r
    

if __name__ == "__main__":
    main()
