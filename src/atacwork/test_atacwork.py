"""
python src/atacwork/test_atacwork.py \
    --project_dir /path/to/SUCCEED \
    --name test_result \
    --use_pth data/model/131k_corr_weight_10.10.1_best_network.pth \
    --model result/denoise/bulk/train_model/ATACwork_best_model.pth \
    --batch 32 \
    --dataset_dir result/denoise/bulk \
    --outpath result/denoise/bulk/test_results/ \
    --cell_types "Bcell-13"
"""
import sys
import argparse

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='test model')
    params.add_argument("--project_dir", help="project dir", required=True)
    params.add_argument("--name", help="model name", required=True)
    params.add_argument("--use_pth", help="Enformer model file", required=True)
    params.add_argument("--model", help="ATACwork model file", required=True)
    params.add_argument("--seed", help="random seed", default=1401, type=int)
    params.add_argument("--device", help="device cuda", default="cuda")
    params.add_argument("--batch", help="batch size", default=64, type=int)
    params.add_argument("--outpath", help="output path", required=True)
    params.add_argument("--dataset_dir", help="directory containing data folders", required=True)
    params.add_argument("--cell_types", help="list of cell types to use", nargs='+', default=["Bcell-13"])
    return params.parse_args()

# Parse arguments and add path before importing other modules
args = parse_arguments()
project_dir = args.project_dir
sys.path.append(project_dir + '/src')

from pathlib import Path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchmetrics import AUROC
import random
import h5py
import numpy as np
from tqdm import tqdm
import os

from model.Enformer import Enformer
from model.config import ModelArgs
from training.metric import compute_rowwise_pearson
import atacwork

def set_random_seed(random_seed=40):
    # set random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.seed = random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed) 
    torch.cuda.manual_seed_all(random_seed)

set_random_seed(args.seed)

class EpiDataset(Dataset):
    def __init__(self, data_files, train=False):
        self.data_files = data_files
        self.train = train

        self.cumulative_lengths = [0]  # cumulative lengths

        # Calculate length and range for each file
        for data_file in self.data_files:
            with h5py.File(data_file, 'r') as d:
                length = len(d['sequence'])  # Get sequence length
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
        data_file = self.data_files[file_idx]

        # Read data from file
        with h5py.File(data_file, 'r') as d:
            seq = d['sequence'][file_index]
            clean = d['clean'][file_index]
            noisy = d['noisy'][file_index]
            label = d['label'][file_index]
        
        return torch.Tensor(seq), torch.Tensor(clean), torch.Tensor(noisy), torch.Tensor(label)

def evaluate(model, val_loader):
    model.eval()
    model_epi.eval()
    total_loss = 0
    total_r = 0
    tmp_r = 0
    total_auroc = 0
    par = tqdm(val_loader, bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    true_signal = []
    pred_signal = []
    sample_signal = []
    true_peak = []
    pred_peak = []
    
    with torch.no_grad():
        for i, (seq, clean, noisy, label) in enumerate(par):
            seq = seq.to(device).float().transpose(1,2)
            sample = noisy.to(device).float().unsqueeze(1)
            target = clean.to(device)
            label = label.to(device)

            embed = model_epi(seq)
            out = model(sample, embed)

            signal = out[0]
            peak = out[1]
            origin_mean = out[2]

            true_signal.append(target.cpu().numpy())
            pred_signal.append(signal.cpu().numpy())
            sample_signal.append(origin_mean.cpu().numpy())
            true_peak.append(label.cpu().numpy())
            pred_peak.append(peak.cpu().numpy())
            
            r_val = compute_rowwise_pearson(signal, target).mean()
            r_val_tmp = compute_rowwise_pearson(target, origin_mean).mean()
            auroc_val = auroc(peak.detach(), label.detach().long())
            
            loss_mse = loss_fn(signal, target)
            loss_binary = loss_bce(peak, label)
            loss = loss_mse + loss_binary
            
            total_loss += loss.item()
            total_r += r_val
            tmp_r += r_val_tmp
            total_auroc += auroc_val
            
            par.set_description(f"Test --> Loss_mse {loss_mse.item():.3f} Loss_bce {loss_binary.item():.3f} R {r_val:.3f} auc {auroc_val:.3f}", refresh=True)
            
        avg_loss = total_loss / len(val_loader)
        avg_r = total_r / len(val_loader)
        avg_tmp_r = tmp_r / len(val_loader)
        avg_auroc = total_auroc / len(val_loader)

        true_signal = np.concatenate(true_signal, axis=0)
        pred_signal = np.concatenate(pred_signal, axis=0)
        sample_signal = np.concatenate(sample_signal, axis=0)
        true_peak = np.concatenate(true_peak, axis=0)
        pred_peak = np.concatenate(pred_peak, axis=0)

        # Save results to npz file
        os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
        np.savez(f"{args.outpath}/{args.name}_signal.npz", 
                true_signal=true_signal, 
                pred_signal=pred_signal, 
                sample_signal=sample_signal, 
                true_peak=true_peak, 
                pred_peak=pred_peak)
        
        return avg_loss, avg_r, avg_auroc, avg_tmp_r

def main():
    global device, model_epi, model, loss_fn, loss_bce, auroc
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Create Enformer model
    config_args = ModelArgs()
    config_args.device = device
    config_args.output_heads = {"human": 6389}
    
    # Load Enformer model
    model_epi = Enformer(config_args, return_emb=True)
    model_epi.to(device)
    checkpoint = torch.load(args.use_pth, map_location="cpu", weights_only=True)
    model_epi.load_state_dict(checkpoint['model_state_dict'])
    model_epi.to(device)
    
    # Freeze all parameters in model_epi
    for param in model_epi.parameters():
        param.requires_grad = False
    
    # Load ATACwork model
    model = atacwork.ATACwork(config_args)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create loss functions
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    auroc = AUROC(task='multilabel', num_labels=1024).to(device)
    
    # Prepare test data
    dataset_dir = Path(args.dataset_dir)
    test_data = []
    
    # Collect test data files for specified cell types
    for cell_type in args.cell_types:
        cell_type_dir = dataset_dir / cell_type
        if cell_type_dir.exists():
            test_data.extend([str(f) for f in cell_type_dir.iterdir() if f.name.startswith("test_") and f.is_file()])
        else:
            print(f"Warning: Directory {cell_type_dir} does not exist")
    
    print(f"Found {len(test_data)} test files")
    
    # Create test dataset and loader
    test_dataset = EpiDataset(test_data, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=30,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    # Evaluate model
    test_loss, test_r, test_auroc, test_tmp_r = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.3f} R: {test_r:.3f} Auroc: {test_auroc:.3f} tmp_r: {test_tmp_r:.3f}")

if __name__ == "__main__":
    main()
