"""
python src/inference/prediction_HistonTF.py \
  --project_dir /path/to/project \
  --name model_name \
  --model /path/to/model.pth \
  --seq_dir /path/to/sequence/dir \
  --feature_dir /path/to/feature/dir \
  --contig_bed /path/to/test_bed.npy \
  --atac_path /path/to/atac \
  --atac_dict "GM12878.bigWig,HepG2.bigWig,K562.bigWig,MCF-7.bigWig" \
  --batch 4 \
  --out_dir /path/to/output
"""
import sys
import argparse

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='Make predictions using the HistonTF model')
    params.add_argument("--project_dir", help="Project directory", required=True)
    params.add_argument("--name", help="Model name", required=True)
    params.add_argument("--use_pth", help="Model file path (mutually exclusive with --model)", required=False)
    params.add_argument("--model", help="Model weights file path", required=False)
    params.add_argument("--seq_dir", help="DNA sequence directory", required=True)
    params.add_argument("--feature_dir", help="Feature directory", required=True)
    params.add_argument("--contig_bed", help="Test contig bed file", required=True)
    params.add_argument("--atac_path", help="ATAC data path", required=True)
    params.add_argument("--atac_dict", help="ATAC file list, comma-separated", required=True)
    params.add_argument("--seed", help="Random seed", default=1401, type=int)
    params.add_argument("--device", help="Computing device", default="cuda")
    params.add_argument("--batch", help="Batch size", default=4, type=int)
    params.add_argument("--out_dir", help="Output directory", required=True)
    params.add_argument("--genome_assembly", help="Genome assembly version", default="hg38")
    params.add_argument("--output_head_size", help="Output head size", default=46, type=int)
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

from data.GenomeDataset import EnformerDataset, HistonTFDataset
from model.HistonTF import HistonTF
from training.metric import MeanPearsonCorrCoefPerChannel
from model.blocks import TargetLengthCrop

from model.config import ModelArgs

class EpiDataset(Dataset):
    def __init__(self, data):
        self.sequence = data["sequence"]
        self.target = data["target"]

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return torch.Tensor(self.sequence[idx]), torch.Tensor(self.target[idx])

def main():
    ## Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ## Parameters
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir
    model_save_path = f"{out_dir}/{args.name}.h5"
    
    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save configuration to yaml file
    hparams = vars(args)
    with open(f"{out_dir}/prediction_config.yaml", 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)
    
    # Convert ATAC dictionary from string to list
    atac_dict = args.atac_dict.split(',')
    
    # Load test data
    batch_size = int(args.batch)
    contig_bed = np.load(args.contig_bed, allow_pickle=True)
    
    val_dataset = HistonTFDataset(
                    contig_bed=contig_bed,
                    seq_dir=args.seq_dir,
                    feature_dir=args.feature_dir,
                    genome_assembly=args.genome_assembly,
                    atac_path=args.atac_path,
                    atac_dict=atac_dict,
                    mode="test",
                    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True)

    ## Define model
    model = HistonTF(ModelArgs(
        device=device,
        output_heads=dict(num_classes=args.output_head_size)
    ))

    ## Load model weights
    if args.use_pth:
        checkpoint = args.use_pth
        model = HistonTF(ModelArgs(
            device=device,
            output_heads=dict(num_classes=args.output_head_size)
        ), checkpoint)
    if args.model:
        checkpoint = torch.load(args.model, map_location=device)["model_state_dict"]
        model.load_state_dict(checkpoint)
    
    model.to(device)
    valid_par = tqdm(val_loader, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    ## Create loss function
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    ## Create evaluation metrics
    corr_coef_test = MeanPearsonCorrCoefPerChannel(n_channels=args.output_head_size).to(device)

    ## Evaluate
    avg_loss, avg_pearson_r = evaluate(model, valid_par, loss_fn, corr_coef_test, device, model_save_path)
    print(f"Validation loss: {avg_loss:.3f} Pearson correlation coefficient: {avg_pearson_r:.3f}")


def evaluate(model, valid_data, loss_fn, corr_coef, device, out_dir):
    target_crop = TargetLengthCrop(896)
    evaluate_loss = 0
    total_r = 0
    model.eval()
    target_ls = []
    prediction_ls = []
    label_ls = []
    chrom_ls = []
    with torch.no_grad():
        corr_coef.reset()
        for idx, (seq, atac, target, label, chrom) in enumerate(valid_data):
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
                valid_data.set_description(f"Validating--"
                                    f"Validation loss: {loss.item():.3f} "
                                    f"Validation Pearson coefficient: {pearson_r:.3f} ")
            evaluate_loss += loss.item()
            total_r += pearson_r
            prediction_ls.append(outputs.cpu().numpy())
            target_ls.append(target.cpu().numpy())
            label_ls.append(label)
            chrom_ls.append(chrom)
            torch.cuda.empty_cache()
    # Calculate average loss
    avg_loss = evaluate_loss / len(valid_data)
    avg_pearson_r = total_r / len(valid_data)
    # Free all GPU memory
    torch.cuda.empty_cache()
    # Write to H5 file
    target_ls = np.concatenate(target_ls, axis=0, dtype=np.float16)
    prediction_ls = np.concatenate(prediction_ls, axis=0, dtype=np.float16)
    with h5py.File(out_dir, "w") as f:
        f.create_dataset("target", data=target_ls, dtype="float16")
        f.create_dataset("prediction", data=prediction_ls, dtype="float16")
    
    # Save additional data
    out_dir_base = os.path.dirname(out_dir)
    np.save(f"{out_dir_base}/label.npy", np.concatenate(label_ls))
    np.save(f"{out_dir_base}/chrom.npy", np.concatenate(chrom_ls))
    
    return avg_loss, avg_pearson_r


if __name__ == "__main__":
    main()


