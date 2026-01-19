import os
import sys
import math
import random
import argparse
from typing import List

import h5py
import pysam
import pyBigWig
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def parse_args():
    p = argparse.ArgumentParser("Train EPCOT-like model with CLI args")

    # ---- SUCCEED src path (NEW) ----
    p.add_argument(
        "--project_dir",
        required=True,
        help="Path to SUCCEED/src, e.g. /home/xxx/SUCCEED/src",
    )

    # paths
    p.add_argument("--fasta", required=True, help="Reference fasta, e.g. hg38.fa")
    p.add_argument("--bed", required=True, help="BED file with columns: chrom start end name")
    p.add_argument("--h5_dir", required=True, help="Directory containing per-key h5 files")
    p.add_argument("--bw_dir", required=True, help="Directory containing per-key bigWig files")
    p.add_argument("--keys", required=True, help="Comma-separated keys, e.g. GM12878,HepG2,K562,MCF-7")

    # keys inside h5
    p.add_argument("--train_target_key", default="train_target")
    p.add_argument("--test_target_key", default="test_target")
    p.add_argument("--index_key", default="index")

    # split
    p.add_argument("--test_chroms", default="chr2,chr10,chr21",
                   help="Comma-separated chroms used as test split")
    p.add_argument("--test_only_chrom", default="",
                   help="Optional: only evaluate this chrom in test (e.g. chr2). Empty means use all test_chroms.")

    # training hyperparams
    p.add_argument("--seed", type=int, default=1401)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)

    # output
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)

    # model settings
    p.add_argument("--output_head_size", type=int, default=46)
    p.add_argument("--target_seq_len", type=int, default=None,
                   help="If your EPCOT config uses it; otherwise ignore.")
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic mode (may slow down).")
    p.add_argument("--succeed_ckpt", required=True,
                   help="Path to SUCCEED checkpoint .pth")

    return p.parse_args()


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EpiDataset(Dataset):
    def __init__(self, contigs_array, targets_array, index_array, atac_path: str, fasta_path: str, dna_1hot_func):
        """
        contigs_array: ndarray of shape (N, 4): chrom,start,end,name
        targets_array: ndarray of shape (N, ...)
        index_array:  ndarray (whatever you stored)
        """
        self.contigs = contigs_array
        self.targets = targets_array
        self.index = index_array
        self.atac_path = atac_path
        self.fasta_path = fasta_path
        self._fasta = None
        self._atac = None
        self.dna_1hot = dna_1hot_func

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_path)
        return self._fasta

    @property
    def atac(self):
        if self._atac is None:
            self._atac = pyBigWig.open(self.atac_path)
        return self._atac

    def get_atac_bw(self, chrom, start, end):
        signals = self.atac.values(chrom, int(start), int(end))
        signals = np.nan_to_num(signals, 0.0)
        return np.asarray(signals)

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        chrom, start, end, _name = self.contigs[idx]
        seq = self.fasta.fetch(chrom, int(start), int(end))
        seq = self.dna_1hot(seq)

        target = self.targets[idx]
        atac_signal = self.get_atac_bw(chrom, start, end)

        return (
            torch.tensor(seq, dtype=torch.float16),
            torch.tensor(atac_signal, dtype=torch.float16),
            torch.tensor(target, dtype=torch.float16),
            torch.tensor(self.index, dtype=torch.long),
        )


def load_bed_split(bed_path: str, test_chroms: List[str], test_only_chrom: str = ""):
    bed = pd.read_csv(bed_path, sep="\t", header=None)
    bed.columns = ["chrom", "start", "end", "name"]

    train_df = bed[~bed["chrom"].isin(test_chroms)].reset_index(drop=True)
    test_df = bed[bed["chrom"].isin(test_chroms)].reset_index(drop=True)

    if test_only_chrom:
        test_df = test_df[test_df["chrom"] == test_only_chrom].reset_index(drop=True)

    return train_df, test_df


def build_datasets(args, train_df, test_df, dna_1hot_func):
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    h5_paths = [os.path.join(args.h5_dir, f"{k}.h5") for k in keys]
    atac_paths = [os.path.join(args.bw_dir, f"{k}.bigWig") for k in keys]

    for fp in h5_paths + atac_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

    train_datasets, test_datasets = [], []

    for h5_path, atac_path in zip(h5_paths, atac_paths):
        with h5py.File(h5_path, "r") as h5f:
            train_targets = h5f[args.train_target_key][:]
            test_targets_full = h5f[args.test_target_key][:]
            index_arr = h5f[args.index_key][:]

        train_ds = EpiDataset(
            train_df.values, train_targets, index_arr,
            atac_path, args.fasta, dna_1hot_func
        )
        # Note: test_targets_full must be aligned to the number of rows in test_df
        test_ds = EpiDataset(
            test_df.values, test_targets_full[:len(test_df)],
            index_arr, atac_path, args.fasta, dna_1hot_func
        )

        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    merged_train = ConcatDataset(train_datasets)

    train_loader = DataLoader(
        merged_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )

    test_loaders = [
        DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=seed_worker
        )
        for ds in test_datasets
    ]
    return train_loader, test_loaders


def build_models(args, device):
    # These imports must happen after sys.path.append(args.project_dir)
    from EFP.models import SUCCEED_EPI
    from pretrain import layers
    from EFP.config import ModelArgs as SucceedArgs

    succeed_cfg = SucceedArgs()
    succeed_cfg.device = device
    model_epi = layers.SUCCEED(succeed_cfg, return_emb=True)

    ckpt = torch.load(args.succeed_ckpt, map_location="cpu")
    filtered_state_dict = {k: v for k, v in ckpt['model_state_dict'].items() if not k.startswith('_heads.')}
    incompat = model_epi.load_state_dict(filtered_state_dict, strict=False)
    
    print("SUCCEED missing:", incompat.missing_keys)
    print("SUCCEED unexpected:", incompat.unexpected_keys)

    model_epi.to(device)
    for p in model_epi.parameters():
        p.requires_grad = False
    model_epi.eval()

    # EPCOT head
    from EFP.config import ModelArgs  # If this conflicts with your project, switch to EPCOT's config

    epcot_cfg = ModelArgs()
    epcot_cfg.output_heads = {"human": args.output_head_size}
    epcot_cfg.depth = 4
    epcot_cfg.device = device
    epcot_cfg.target_seq_len = 896
    if args.target_seq_len is not None:
        epcot_cfg.target_seq_len = args.target_seq_len

    model = SUCCEED_EPI(epcot_cfg).to(device)
    target_crop = layers.TargetLengthCrop(epcot_cfg.target_seq_len)

    return model_epi, model, target_crop


def train_one_run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, args.deterministic)

    # Import utilities from SUCCEED/src (dna_1hot/metric)
    from utils.metric import MeanPearsonCorrCoefPerChannel, EarlyStopping
    from utils.GenomeDataset import dna_1hot

    test_chroms = [x.strip() for x in args.test_chroms.split(",") if x.strip()]
    train_df, test_df = load_bed_split(args.bed, test_chroms, args.test_only_chrom)

    train_loader, test_loaders = build_datasets(args, train_df, test_df, dna_1hot)

    model_epi, model, target_crop = build_models(args, device)

    loss_fn = nn.PoissonNLLLoss(log_input=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    corr = MeanPearsonCorrCoefPerChannel(n_channels=args.output_head_size).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    model_save_name = f"{args.run_name}_seed{args.seed}"
    early_stopping = EarlyStopping(save_path=args.out_dir, model_name=model_save_name, patience=args.patience)

    for epoch in range(args.epochs):
        # -------- train --------
        model.train()
        corr.reset()
        par = tqdm(train_loader, total=len(train_loader))

        for step, batch in enumerate(par):
            seq, atac, target, index = batch
            seq = seq.to(device).float().transpose(1, 2)
            atac = atac.to(device).float().unsqueeze(1)
            target = target.to(device).float()
            target = target_crop(target)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                embed = model_epi(seq)
            out = model(atac, embed)["human"]

            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            corr.update(out.detach(), target.detach())

            if (step + 1) % 50 == 0:
                r_train = corr.compute().mean().item()
                par.set_description(
                    f"Train Epoch [{epoch}/{args.epochs}] "
                    f"Loss {loss.item():.3f} PCC {r_train:.3f} LR {optimizer.param_groups[0]['lr']:.2e}"
                )

        # -------- eval: average over keys --------
        model.eval()
        r_sum = 0.0
        with torch.no_grad():
            for tl in test_loaders:
                corr.reset()
                par_test = tqdm(tl, total=len(tl))
                for seq, atac, target, index in par_test:
                    seq = seq.to(device).float().transpose(1, 2)
                    atac = atac.to(device).float().unsqueeze(1)
                    target = target.to(device).float()
                    target = target_crop(target)

                    embed = model_epi(seq)
                    out = model(atac, embed)["human"]
                    corr.update(out, target)

                r_cell = corr.compute().mean().item()
                r_sum += r_cell

        r_avg = r_sum / len(test_loaders)
        print(f"\nEpoch {epoch}: TEST mean PCC over cells = {r_avg:.4f}")

        early_stopping(-r_avg, model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def main():
    args = parse_args()

    # Important: append SUCCEED/src to sys.path only after parsing args
    if not os.path.isdir(args.project_dir):
        raise NotADirectoryError(f"--project_dir is not a directory: {args.project_dir}")
    sys.path.append(os.path.join(args.project_dir, "src"))

    train_one_run(args)


if __name__ == "__main__":
    main()
