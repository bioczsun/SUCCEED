import subprocess
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import Dataset, DataLoader
from torchmetrics import AUROC
import random
import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from model.Enformer import Enformer
from model.config import ModelArgs
from training.metric import EarlyStopping, compute_rowwise_pearson
import atacwork

import argparse
params = argparse.ArgumentParser(description='inference model')
params.add_argument("--fasta",help="fasta",required=True)
params.add_argument("--chrom_size",help="chrom_size",required=True)
params.add_argument('--noisy_file', type=str, help='Path to the noisy BigWig file')
params.add_argument("--limit_bed", type=str, help="Path to the limit bed file")
params.add_argument("--limit_chrom", type=str, help="limit chromosome")
params.add_argument("--gaps_file", type=str, help="Path to the gaps file")
params.add_argument("--name",help="model name",required=True)
params.add_argument("--restart", action="store_true")
params.add_argument("--seed",help="random seed",default=1401)
params.add_argument("--model_epi",help="model file",required=False)
params.add_argument("--model",help="optimizer file",required=False)
params.add_argument("--device",help="device cuda",default="cuda:1")
params.add_argument("--outpath",help="model name")
args = params.parse_args()

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

# Run data preprocessing script through subprocess
def run_data_preprocessing(args):

    # Return generated HDF5 file path
    h5_file = f"{args.outpath}/inference_data_{args.name}.h5"
    if os.path.exists(h5_file) and not args.restart:
        print(f"Data file {h5_file} already exists, skipping data preprocessing.")
        return h5_file
    else:
        print("Running data preprocessing...")
        command = [
            "python", "src/atacwork/atacwork_data_inference.py",
            "--noisy_file", args.noisy_file,
            "--fasta_file", args.fasta,
            "--gaps_file", args.gaps_file,
            "--limit_chrom", args.limit_chrom,
            "--out_dir", args.outpath,
            "--name", args.name,
            "--restart"
        ]
        # Direct subprocess output to parent process
        result = subprocess.run(command, text=True)

        # Check subprocess return code
        if result.returncode != 0:
            raise RuntimeError(f"Data preprocessing failed: {result.stderr}")

        print("Data preprocessing completed successfully.")
        return h5_file
    
def run_bed_to_bigwig(bedGraph_file,type):
    command = [
        "bedGraphToBigWig", f"{bedGraph_file}",f"{args.chrom_size}",f"{args.outpath}/{args.name}_succeed_{type}.bw",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    while True:
        output = process.stdout.readline()  # type: ignore
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())  # Print subprocess output

    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read()  # type: ignore
        raise RuntimeError(f"bedGraph to bigWig failed: {stderr}")

    print("bedGraph to bigWig completed successfully.")

class EpiDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files


    def __len__(self):
        with h5py.File(self.data_files, 'r') as d:
            return len(d['sequence']) # type: ignore

    def __getitem__(self, idx):
        # Read data from file
        with h5py.File(self.data_files, 'r') as d:
            seq = d['sequence'][idx] # type: ignore
            noisy = d['noisy'][idx] # type: ignore
        return torch.Tensor(seq), torch.Tensor(noisy)




    
# define train and val data    
batch_size = 32
device = args.device if torch.cuda.is_available() else "cpu"



## create epimodel
config_args = ModelArgs()
config_args.device = device
config_args.output_heads = {"human": 6389}
model_epi = Enformer(config_args, return_emb=True)
model_epi.to(device)


checkpoint = torch.load(args.model_epi,map_location="cpu",weights_only=True)
model_epi.load_state_dict(checkpoint['model_state_dict'])
model_epi.to(device)


## Freeze all parameters in model_epi
for param in model_epi.parameters():
    param.requires_grad = False

model = atacwork.ATACwork(config_args)
checkpoint = torch.load(args.model,map_location="cpu",weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
test_data = f"{run_data_preprocessing(args)}"

## create test dataset
test_dataset = EpiDataset(
    data_files=test_data,
)

val_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=30,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

loss_fn = nn.MSELoss()
loss_bce = nn.BCELoss()

auroc = AUROC(task='multilabel', num_labels=1024).to(device)

inference_results_path = os.path.join(args.outpath, f"{args.name}_signal.npz")



def evaluate(model, val_loader):
    model.eval()
    model_epi.eval()
    par = tqdm(val_loader,bar_format='{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    # corr_coef_test.reset()
    pred_signal = []
    pred_peak = []
    with torch.no_grad():
        for i, (seq,noisy) in enumerate(par):
            seq = seq.to(device).float().transpose(1,2)
            sample = noisy.to(device).float().unsqueeze(1)

            embed = model_epi(seq)
            out = model(sample,embed)

            signal = out[0]
            peak = out[1]
            origin_mean = out[2]

            pred_signal.append(signal.cpu().numpy())
            pred_peak.append(peak.cpu().numpy())


        pred_signal = np.concatenate(pred_signal, axis=0)
        pred_peak = np.concatenate(pred_peak, axis=0)

        ## Save as npz file
        # np.savez(inference_results_path,pred_signal=pred_signal,pred_peak=pred_peak)
    print("Starting to generate bedGraph......")
    sequence_bed = open(f"{args.outpath}/sequence.bed").readlines()
    signal_ls = []
    peak_ls = []
    for seq,signal,peak in zip(sequence_bed,pred_signal,pred_peak):
        chr = seq.split("\t")[0]
        start = int(seq.split("\t")[1])
        end = int(seq.split("\t")[2])
        for index,res_start in enumerate(range(start,end,128)):
            res_end = res_start + 128
            signal_ls.append([chr,res_start,res_end,signal[index]])
            peak_ls.append([chr,res_start,res_end,peak[index]])
    tmp_df = pd.DataFrame(signal_ls,columns=["chr","start","end","signal"])
    tmp_df.sort_values(by=["chr","start"],inplace=True)
    tmp_df.drop_duplicates(subset=["chr","start","end"],inplace=True,keep="first")
    tmp_df.to_csv(f"{args.outpath}/{args.name}_succeed_signal.bedGraph",sep="\t",index=False,header=False)

    tmp_df = pd.DataFrame(peak_ls,columns=["chr","start","end","peak"])
    tmp_df.sort_values(by=["chr","start"],inplace=True)
    tmp_df.drop_duplicates(subset=["chr","start","end"],inplace=True,keep="first")
    tmp_df.to_csv(f"{args.outpath}/{args.name}_succeed_peak.bedGraph",sep="\t",index=False,header=False)
    print("Successfully generated bedGraph file!")

    print("Converting bedGraph to bigWig......")
    run_bed_to_bigwig(f"{args.outpath}/{args.name}_succeed_signal.bedGraph","signal")
    run_bed_to_bigwig(f"{args.outpath}/{args.name}_succeed_peak.bedGraph","peak")

evaluate(model, val_loader)
