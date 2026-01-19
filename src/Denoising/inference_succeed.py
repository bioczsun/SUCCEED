#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import random
import sys
from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import pyBigWig
import pysam


def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
    """ dna_1hot

    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
      n_sample:  sample ACGT for N

    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim:seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2

    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    if n_uniform:
        seq_code = np.zeros((seq_len, 4), dtype='float16')
    else:
        seq_code = np.zeros((seq_len, 4), dtype='bool')
        
    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == 'A':
                seq_code[i, 0] = 1
            elif nt == 'C':
                seq_code[i, 1] = 1
            elif nt == 'G':
                seq_code[i, 2] = 1
            elif nt == 'T':
                seq_code[i, 3] = 1
            else:
                # Set N or unknown bases to zero
                seq_code[i, :] = 0

    return seq_code

# ======================
# Utils
# ======================
def ensure_src_on_path(project_dir: Optional[str]) -> None:
    if project_dir:
        src_path = Path(project_dir) / "src"
    else:
        src_path = Path(__file__).resolve().parents[1]
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)


def set_random_seed(seed: int = 1401) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def bedgraph_to_bigwig(bg, chrom_size, bw):
    if shutil.which("bedGraphToBigWig") is None:
        raise FileNotFoundError("bedGraphToBigWig not found in PATH")
    subprocess.run(
        ["bedGraphToBigWig", bg, chrom_size, bw],
        check=True
    )


def resolve_overlaps(df, mode="mean"):
    if mode == "mean":
        return df.groupby(["chr","start","end"], as_index=False).mean()
    else:
        return df.groupby(["chr","start","end"], as_index=False).max()


# ======================
# Dataset
# ======================
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, fasta, noisy_bw):
        self.contigs = contigs
        self.fasta_file = fasta
        self.noisy_bw_path = noisy_bw
        self._fasta = None
        self._bw = None

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_file)
        return self._fasta

    @property
    def bw(self):
        if self._bw is None:
            self._bw = pyBigWig.open(self.noisy_bw_path)
        return self._bw

    def close(self) -> None:
        if self._bw is not None:
            try:
                self._bw.close()
            finally:
                self._bw = None
        if self._fasta is not None:
            try:
                self._fasta.close()
            finally:
                self._fasta = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        chrom, start, end = self.contigs[idx]
        start, end = int(start), int(end)

        seq = self.fasta.fetch(chrom, start, end)
        seq_1hot = dna_1hot(seq)

        noisy = self.bw.values(chrom, start, end)
        noisy = np.nan_to_num(np.array(noisy), nan=0)

        return (
            torch.tensor(seq_1hot, dtype=torch.float32),
            torch.tensor(noisy, dtype=torch.float32),
            chrom,
            start,
            end
        )


# ======================
# Inference
# ======================
@torch.no_grad()
def run_inference(model, model_epi, loader, device, outpath, prefix, step_bp: int = 128):
    signal_rec = []
    peak_rec = []

    model.eval()
    model_epi.eval()

    for seq, noisy, chroms, starts, ends in tqdm(loader):
        seq = seq.to(device).transpose(1, 2)
        noisy = noisy.to(device).unsqueeze(1)

        embed = model_epi(seq)
        signal, peak, _ = model(noisy, embed)

        signal = signal.cpu().numpy()
        peak = peak.cpu().numpy()

        for b in range(signal.shape[0]):
            start = int(starts[b])
            end = int(ends[b])
            chrom = chroms[b]
            n_bins = signal.shape[1]
            for i in range(n_bins):
                pos = start + i * step_bp
                if pos >= end:
                    break
                signal_rec.append([chrom, pos, pos + step_bp, float(signal[b, i])])
                peak_rec.append([chrom, pos, pos + step_bp, float(peak[b, i])])

    sig_df = pd.DataFrame(signal_rec, columns=["chr","start","end","value"])
    peak_df = pd.DataFrame(peak_rec, columns=["chr","start","end","value"])

    sig_df = resolve_overlaps(sig_df, "mean")
    peak_df = resolve_overlaps(peak_df, "max")

    sig_bg = f"{outpath}/{prefix}_signal.bedGraph"
    peak_bg = f"{outpath}/{prefix}_peak.bedGraph"

    sig_df.sort_values(["chr","start"]).to_csv(sig_bg, sep="\t", index=False, header=False)
    peak_df.sort_values(["chr","start"]).to_csv(peak_bg, sep="\t", index=False, header=False)

    return sig_bg, peak_bg


# ======================
# Main
# ======================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SUCCEED ATAC denoising inference")
    parser.add_argument(
        "--project-dir",
        "--project_dir",
        dest="project_dir",
        default=None,
        help="Project root; adds <root>/src to PYTHONPATH",
    )

    parser.add_argument("--model", required=True, help="ATAC denoising checkpoint (.pth)")
    parser.add_argument(
        "--succeed-ckpt",
        "--model-ckpt",
        "--model_epi",
        dest="succeed_ckpt",
        required=True,
        help="SUCCEED checkpoint used to build embeddings",
    )

    parser.add_argument("--sequence-bed", "--sequence_bed", dest="sequence_bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--noisy-bw", "--noisy_bw", dest="noisy_bw", required=True)
    parser.add_argument("--chrom-size", "--chrom_size", dest="chrom_size", required=True)

    parser.add_argument("--seed", type=int, default=1401, help="Random seed and run tag")
    parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--step-bp", type=int, default=128, help="Bin size in bp for bedGraph output")
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--run-name", default=None, help="Output file prefix override")
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        return "cpu"
    return device


def _read_sequence_bed(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] == 3:
        df.columns = ["chrom", "start", "end"]
    elif df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["chrom", "start", "end", "name"]
    else:
        raise ValueError(f"sequence bed must have >=3 columns, got {df.shape[1]}")
    return df[["chrom", "start", "end"]].values


def _load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_succeed_embedder(config_args, ckpt_path: str, device: str):
    from pretrain import layers

    model_epi = layers.SUCCEED(config_args, return_emb=True)
    state_dict = _strip_module_prefix(_load_checkpoint_state_dict(ckpt_path))
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_heads.")}
    incompat = model_epi.load_state_dict(filtered_state_dict, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print("SUCCEED load_state_dict missing:", incompat.missing_keys)
        print("SUCCEED load_state_dict unexpected:", incompat.unexpected_keys)

    model_epi.to(device)
    for param in model_epi.parameters():
        param.requires_grad = False
    model_epi.eval()
    return model_epi


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.project_dir)

    from pretrain.config import ModelArgs

    os.makedirs(args.outpath, exist_ok=True)
    set_random_seed(args.seed)

    device = _resolve_device(args.device)

    contigs = _read_sequence_bed(args.sequence_bed)

    dataset = InferenceDataset(contigs, args.fasta, args.noisy_bw)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    config_args = ModelArgs()
    config_args.device = device

    model_epi = _build_succeed_embedder(config_args, args.succeed_ckpt, device)

    from Denoising.models.atacwork import ATACDenoising

    model = ATACDenoising(config_args)
    model_state_dict = _load_checkpoint_state_dict(args.model)
    model.load_state_dict(_strip_module_prefix(model_state_dict))
    model.to(device)
    model.eval()

    run_name = args.run_name or f"SUCCEED_seed_{args.seed}"

    sig_bg, peak_bg = run_inference(
        model, model_epi, loader, device, args.outpath, run_name, step_bp=args.step_bp
    )

    bedgraph_to_bigwig(sig_bg, args.chrom_size, f"{args.outpath}/{run_name}_signal.bw")
    bedgraph_to_bigwig(peak_bg, args.chrom_size, f"{args.outpath}/{run_name}_peak.bw")

    dataset.close()
    print("Inference finished")


if __name__ == "__main__":
    main()
