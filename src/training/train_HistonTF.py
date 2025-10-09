"""
python src/training/train_HistonTF.py \
  --project_dir /path/to/project \
  --seq_dir /path/to/sequence/dir \
  --train_feature_dir /path/to/train/features \
  --valid_feature_dir /path/to/valid/features \
  --train_contig_bed /path/to/train/bed.npy \
  --valid_contig_bed /path/to/valid/bed.npy \
  --atac_path /path/to/atac \
  --atac_dict "GM12878.bigWig,HepG2.bigWig,K562.bigWig,MCF-7.bigWig" \
  --name model_name \
  --output_head_size 46 \
  --out_dir /path/to/output
"""
import sys
import argparse

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='train HistonTF model')
    params.add_argument("--project_dir", help="project dir", required=True)
    params.add_argument("--seq_dir", help="DNA sequence directory", required=True)
    params.add_argument("--train_feature_dir", help="training feature directory", required=True)
    params.add_argument("--valid_feature_dir", help="validation feature directory", required=True)
    params.add_argument("--train_contig_bed", help="training contig bed file", required=True)
    params.add_argument("--valid_contig_bed", help="validation contig bed file", required=True)
    params.add_argument("--atac_path", help="ATAC path", required=True)
    params.add_argument("--atac_dict", help="ATAC dictionary as comma-separated list", required=True)
    params.add_argument("--name", help="model name", required=True)
    params.add_argument("--use_pth", help="model file", required=False)
    params.add_argument("--seed", help="random seed", default=1401, type=int)
    params.add_argument("--device", help="device cuda", default="cuda")
    params.add_argument("--epoch", help="epochs", default=50000, type=int)
    params.add_argument("--lr", help="learn rata", default=0.0001, type=float)
    params.add_argument("--batch", help="batch size", default=4, type=int)
    params.add_argument("--out_dir", help="output dir", required=True)
    params.add_argument("--output_head_size", help="size of output head", default=46, type=int)
    params.add_argument("--genome_assembly", help="genome assembly", default="hg38")
    return params.parse_args()

# Parse arguments and add path before importing other modules
args = parse_arguments()
project_dir = args.project_dir
sys.path.append(project_dir + '/src')

# Then import other modules
import torch
import torch.nn as nn
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import Dataset, DataLoader

import random
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import yaml 
import csv
import math
from torch.optim.lr_scheduler import LambdaLR

from data.GenomeDataset import GenomeDataset, EnformerDataset, HistonTFDataset
from model.HistonTF import HistonTF
from training.metric import MeanPearsonCorrCoefPerChannel, EarlyStopping
from model.blocks0 import TargetLengthCrop

from model.config import ModelArgs

def warmup_cosine_annealing_lr(warmup_epochs, max_epochs, warmup_start_lr, base_lr, min_lr):
    """Return a Lambda function for calculating learning rate"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)) / base_lr
        else:
            # Cosine annealing
            cos_anneal = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
            return (min_lr + (base_lr - min_lr) * cos_anneal) / base_lr
    return lr_lambda

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
    # Get the maximum version number of current runs
    existing_versions = [int(d.split("_")[-1]) for d in os.listdir(log_dir) if d.startswith("version_")]
    new_version = max(existing_versions) + 1 if existing_versions else 0

    # Create folder for the new version
    run_folder = os.path.join(log_dir, f"version_{new_version}")
    os.makedirs(run_folder)
    ## Write hyperparameters to yaml file
    # Create hyperparameter dictionary
    hparams = {
        "seq_dir": args.seq_dir,
        "train_feature_dir": args.train_feature_dir,
        "valid_feature_dir": args.valid_feature_dir,
        "train_contig_bed": args.train_contig_bed,
        "valid_contig_bed": args.valid_contig_bed,
        "atac_path": args.atac_path,
        "atac_dict": args.atac_dict,
        "model_name": args.name,
        "use_pth": args.use_pth,
        "seed": args.seed,
        "device": device,
        "epoch": epoch,
        "learning_rate": lr,
        "batch_size": batch_size,
        "output_dir": out_dir,
        "output_head_size": args.output_head_size,
        "genome_assembly": args.genome_assembly
    }
    with open(os.path.join(run_folder, "hparams.yaml"), 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)

    # Convert ATAC dictionary from string to list
    atac_dict = args.atac_dict.split(',')

    # Load training and validation contig_bed
    contig_bed = np.load(args.train_contig_bed, allow_pickle=True)

    ## create dataloader
    train_dataset = HistonTFDataset(
                        contig_bed=contig_bed,
                        seq_dir=args.seq_dir,
                        feature_dir=args.train_feature_dir,
                        genome_assembly=args.genome_assembly,
                        atac_path=args.atac_path,
                        atac_dict=atac_dict,
                        mode="train",
                        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

    contig_bed = np.load(args.valid_contig_bed, allow_pickle=True)
    val_dataset = HistonTFDataset(
                    contig_bed=contig_bed,
                    seq_dir=args.seq_dir,
                    feature_dir=args.valid_feature_dir,
                    genome_assembly=args.genome_assembly,
                    atac_path=args.atac_path,
                    atac_dict=atac_dict,
                    mode="valid",
                    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

    ## define model
    model = HistonTF(ModelArgs(
        device=device,
        output_heads=dict(num_classes=args.output_head_size)
    ))

    ## use pth
    if args.use_pth:
        checkpoint = args.use_pth
        model = HistonTF(ModelArgs(
            device=device,
            output_heads=dict(num_classes=args.output_head_size)
        ), checkpoint)

    model.to(device)
    # ## Check if model parameters are frozen
    # for name,param in model.named_parameters():
    #     print(name,param.requires_grad)
    ## define optimizer and loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # define scheduler
    scheduler = LambdaLR(
        optimizer, 
        lr_lambda=warmup_cosine_annealing_lr(
            warmup_epochs=5,  # 5 epochs warmup
            max_epochs=epoch,
            warmup_start_lr=1e-5, 
            base_lr=1e-3, 
            min_lr=1e-5
        )
    )

    loss_fn = nn.PoissonNLLLoss(log_input=False)

    ## define metric
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=args.output_head_size).to(device)

    ## define early stopping
    model_save_path = args.name
    early_stopping = EarlyStopping(save_path=out_dir, model_name=model_save_path, patience=8)

    ## train
    train(run_folder, model, train_loader, val_loader, optimizer, scheduler, loss_fn, corr_coef, device, epoch, early_stopping)


def train(run_folder, model, train_data, valid_data, optimizer, scheduler, loss_fn, corr_coef, device, epochs, early_stopping):
    target_crop = TargetLengthCrop(896)
    model.train()

    metrics_path = os.path.join(run_folder, "metrics.csv")
    # If metrics.csv doesn't exist, write headers
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_pearson_r", "valid_loss", "valid_pearson_r"])

    for epoch in range(epochs):
        train_par = tqdm(train_data, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        total_loss = 0
        total_r = 0
        for idx, (seq, atac, target) in enumerate(train_par):
            corr_coef.reset()
            seq = seq.to(device).float().transpose(1, 2)
            atac = atac.to(device).float().unsqueeze(1)
            target = target.to(device).float()
            target = target_crop(target)
            optimizer.zero_grad()
            outputs = model(seq, atac)["num_classes"]  # Forward pass
            outputs = target_crop(outputs)
            loss = loss_fn(outputs, target)  # Calculate loss
            loss.backward()  # Backpropagation
            clip_grad.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()

            # Calculate Pearson correlation coefficient
            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                train_par.set_description(f"Train--Epoch: [{epoch} / {epochs}] "
                                    f"Train loss: {loss.item():.3f} "
                                    f"Train Pearson r: {pearson_r:.3f} "
                                    f"LR: {current_lr}")

            total_loss += loss.item()
            total_r += pearson_r
        avg_loss = total_loss / len(train_data)
        avg_pearson_r = total_r / len(train_data)
        valid_par = tqdm(valid_data, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        valid_loss, valid_pearson_r = evaluate(model, valid_par, loss_fn, corr_coef, device)
        # Print training and validation loss for each epoch
        print(f"Epoch: [{epoch} / {epochs}] "
            f"Train loss: {avg_loss:.3f} "
            f"Train Pearson r: {avg_pearson_r:.3f} "
            f"Valid loss: {valid_loss:.3f} "
            f"Valid Pearson r: {valid_pearson_r:.3f}")
        
        # **Append to metrics.csv**
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_pearson_r.item(), valid_loss, valid_pearson_r.item()])
        
        early_stopping(valid_loss, model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step()

def evaluate(model, valid_data, loss_fn, corr_coef, device):
    target_crop = TargetLengthCrop(896)
    evaluate_loss = 0
    total_r = 0
    model.eval()
    with torch.no_grad():
        for idx, (seq, atac, target) in enumerate(valid_data):
            corr_coef.reset()
            seq = seq.to(device).float().transpose(1, 2)
            atac = atac.to(device).float().unsqueeze(1)
            target = target.to(device)
            target = target_crop(target)
            outputs = model(seq, atac)["num_classes"]  # Forward pass
            outputs = target_crop(outputs)
            loss = loss_fn(outputs, target)  # Calculate loss
            # Calculate Pearson correlation coefficient
            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                valid_data.set_description(f"Valid--"
                                    f"Valid loss: {loss.item():.3f} "
                                    f"Valid Pearson r: {pearson_r:.3f} ")
            evaluate_loss += loss.item()
            total_r += pearson_r
        ## Calculate average loss
        avg_loss = evaluate_loss / len(valid_data)
        avg_pearson_r = total_r / len(valid_data)
        return avg_loss, avg_pearson_r
    

if __name__ == "__main__":
    main()
