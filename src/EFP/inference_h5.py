#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EPCOT inference -> HDF5 (cropped bins)
- contigs generated from FASTA chrom sizes (pysam .fai)
- window_len = 1,048,576 bp
- resolution = 1,024 bp -> 1,024 bins per window
- channels = 46
- crop (bins) default = 64 -> valid bins = 1024 - 2*64 = 896
- outputs:
    pred: float32 (N, valid_bins, channels)  # default (N, 896, 46)
    chrom: string (N,)
    start: int64 (N,)   # cropped start
    end: int64 (N,)     # cropped end

NOTE: pysam requires FASTA index (.fai). Create it with:
  samtools faidx genome.fa[.gz]
"""

import os
import re
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import pysam
import pyBigWig
import h5py
from tqdm import tqdm

# =========================
# Project imports (EDIT if needed)
# =========================
import models
import sys
sys.path.append('/home/suncz/work/s02/SUCCEED/src')
from pretrain import layers
from pretrain.config import ModelArgs as SucceedArgs
from config import ModelArgs as EPCOTArgs
from utils import dna_1hot


# =========================
# Helpers
# =========================
def set_seed(seed: int = 1401):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chrom_sizes_from_fasta(fasta_path: str) -> List[Tuple[str, int]]:
    fa = pysam.FastaFile(fasta_path)
    out = [(c, fa.get_reference_length(c)) for c in fa.references]
    fa.close()
    return out


def filter_chroms(
    chrom_sizes: List[Tuple[str, int]],
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
) -> List[Tuple[str, int]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    res = []
    for c, L in chrom_sizes:
        if inc and not inc.search(c):
            continue
        if exc and exc.search(c):
            continue
        res.append((c, L))
    return res


def make_contigs(
    chrom_sizes: List[Tuple[str, int]],
    window_len: int,
    stride: int,
    drop_last: bool = True,
) -> List[Tuple[str, int, int]]:
    contigs: List[Tuple[str, int, int]] = []
    for chrom, L in chrom_sizes:
        s = 0
        while s < L:
            e = s + window_len
            if e > L:
                if drop_last:
                    break
                e = L
            contigs.append((chrom, s, e))
            s += stride
    return contigs


def open_bw(path: str) -> pyBigWig.pyBigWig:
    bw = pyBigWig.open(path)
    if bw is None:
        raise RuntimeError(f"Failed to open bigWig: {path}")
    return bw


def fetch_atac_vector(
    bw: pyBigWig.pyBigWig,
    chrom: str,
    start: int,
    end: int,
    window_len: int,
) -> np.ndarray:
    vals = bw.values(chrom, start, end)
    vals = np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0)
    # pad / trim to window_len
    if len(vals) < window_len:
        vals = np.concatenate([vals, np.zeros((window_len - len(vals),), dtype=np.float32)], axis=0)
    elif len(vals) > window_len:
        vals = vals[:window_len]
    return vals


def fetch_seq_1hot(
    fa: pysam.FastaFile,
    chrom: str,
    start: int,
    end: int,
    window_len: int,
) -> np.ndarray:
    seq = fa.fetch(chrom, start, end).upper()
    if len(seq) < window_len:
        seq = seq + ("N" * (window_len - len(seq)))
    elif len(seq) > window_len:
        seq = seq[:window_len]
    return dna_1hot(seq).astype(np.float32)  # (L,4)


# =========================
# Dataset
# =========================
class EPCOTInferDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, fasta_path, atac_bw_path, window_len):
        self.contigs = contigs
        self.fasta_path = fasta_path
        self.atac_bw_path = atac_bw_path
        self.window_len = window_len
        self._fa = None
        self._bw = None

    def _get_fa(self):
        if self._fa is None:
            self._fa = pysam.FastaFile(self.fasta_path)
        return self._fa

    def _get_bw(self):
        if self._bw is None:
            self._bw = open_bw(self.atac_bw_path)
        return self._bw

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        chrom, start, end = self.contigs[idx]
        fa = self._get_fa()
        bw = self._get_bw()

        seq_1hot = fetch_seq_1hot(fa, chrom, start, end, self.window_len)  # (L,4)
        atac = fetch_atac_vector(bw, chrom, start, end, self.window_len)   # (L,)

        return (
            torch.from_numpy(seq_1hot),          # (L,4)
            torch.from_numpy(atac),              # (L,)
            str(chrom),
            np.int64(start),
            np.int64(end),
        )


# =========================
# Model loading
# =========================
def load_succeed(ckpt_path: str, device: str):
    s_args = SucceedArgs()
    s_args.device = device
    model = layers.SUCCEED(s_args, return_emb=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    model.to(device).eval()
    return model


def load_epcot(ckpt_path: str, device: str, out_channels: int, head_name: str):
    e_args = EPCOTArgs()
    e_args.device = device
    e_args.output_heads = {head_name: out_channels}

    model = models.EPCOT(e_args).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================
# HDF5 writing
# =========================
def create_h5(
    out_h5: str,
    n_samples: int,
    n_bins: int,
    n_channels: int,
    compression: str = "gzip",
    compression_opts: int = 4,
    pred_chunk: Tuple[int, int, int] = (8, 896, 46),
):
    """
    Create an HDF5 file with datasets:
      pred: (N, n_bins, n_channels) float32
      chrom: (N,) variable-length utf-8 string
      start/end: (N,) int64
    """
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)

    f = h5py.File(out_h5, "w")

    # predictions
    f.create_dataset(
        "pred",
        shape=(n_samples, n_bins, n_channels),
        dtype=np.float32,
        chunks=pred_chunk,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )

    # bed order / coordinates
    str_dt = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("chrom", shape=(n_samples,), dtype=str_dt, compression=compression, compression_opts=compression_opts)
    f.create_dataset("start", shape=(n_samples,), dtype=np.int64, compression=compression, compression_opts=compression_opts)
    f.create_dataset("end", shape=(n_samples,), dtype=np.int64, compression=compression, compression_opts=compression_opts)

    return f


@torch.no_grad()
def run_inference_to_h5(
    model_epcot,
    model_succeed,
    loader,
    h5f,
    device,
    head_name,
    window_len,
    resolution,
    out_channels,
    crop=64,
):
    total_bins = window_len // resolution
    valid_bins = total_bins - 2 * crop
    if valid_bins <= 0:
        raise ValueError(f"Invalid crop={crop}: total_bins={total_bins} -> valid_bins={valid_bins}")

    write_idx = 0
    pred_ds = h5f["pred"]

    for seq_1hot, atac, chroms, starts, ends in tqdm(loader, desc="EPCOT inference -> H5"):
        # (B,L,4) -> (B,4,L)
        seq = seq_1hot.to(device).transpose(1, 2)
        atac = atac.to(device).unsqueeze(1)

        emb = model_succeed(seq)
        out = model_epcot(atac, emb)[head_name]     # (B, T, C)

        out = out.detach().cpu().numpy().astype(np.float32)

        B, T, C = out.shape
        if C != out_channels:
            raise ValueError(f"Channel mismatch: out_channels={out_channels}, model_out_C={C}")

        # Expect T == valid_bins (e.g. 896). If model outputs full 1024, crop it here.
        if T == total_bins:
            out = out[:, crop:crop + valid_bins, :]
            T = out.shape[1]

        if T != valid_bins:
            raise ValueError(f"Time/bin mismatch: got T={T}, expected valid_bins={valid_bins} (total_bins={total_bins}, crop={crop})")

        if pred_ds.shape[1] != valid_bins:
            raise ValueError(f"H5 pred bins mismatch: h5={pred_ds.shape[1]}, expected valid_bins={valid_bins}")

        pred_ds[write_idx:write_idx + B, :, :] = out

        # === write cropped coordinates ===
        for b in range(B):
            h5f["chrom"][write_idx + b] = chroms[b]
            h5f["start"][write_idx + b] = starts[b] + crop * resolution
            h5f["end"][write_idx + b]   = starts[b] + (crop + valid_bins) * resolution

        write_idx += B

    assert write_idx == pred_ds.shape[0]
    return write_idx


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser("EPCOT inference -> HDF5 (cropped bins) + bed order")

    p.add_argument("--fasta", required=True, help="Reference genome FASTA (.fa or .fa.gz) with .fai present")
    p.add_argument("--atac_bw", required=True, help="ATAC bigWig (bp-resolution values)")
    p.add_argument("--succeed_ckpt", required=True, help="SUCCEED checkpoint (.pth)")
    p.add_argument("--epcot_ckpt", required=True, help="EPCOT checkpoint (.pth)")
    p.add_argument("--out_h5", required=True, help="Output HDF5 path")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1401)

    p.add_argument("--window_len", type=int, default=1048576)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--stride", type=int, default=1048576, help="Tiling stride; default non-overlap")
    p.add_argument("--drop_last", action="store_true", help="Drop tail shorter than window_len (recommended)")
    p.set_defaults(drop_last=True)

    # chrom filtering: useful to exclude alt/random/chrM
    p.add_argument("--include_regex", default=None, help='e.g. "^chr([0-9]+|X|Y)$"')
    p.add_argument("--exclude_regex", default="_|Un|random|alt|chrM", help='default excludes alt/random/chrM-like contigs')

    # model head + channels
    p.add_argument("--head_name", default="human", help='EPCOT head key, e.g. "human" or "mouse"')
    p.add_argument("--out_channels", type=int, default=46)

    # crop
    p.add_argument("--crop", type=int, default=64, help="Crop bins at both ends. valid_bins = total_bins - 2*crop")

    # performance
    p.add_argument("--batch_size", type=int, default=1, help="Keep small (1MB windows are big)")
    p.add_argument("--num_workers", type=int, default=4)

    # H5 options
    p.add_argument("--h5_compression", default="gzip")
    p.add_argument("--h5_level", type=int, default=4)
    p.add_argument("--chunk_bs", type=int, default=8, help="H5 pred chunk batch dim")

    args = p.parse_args()
    set_seed(args.seed)

    device = args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"

    # 1) chrom sizes from FASTA
    chrom_sizes = load_chrom_sizes_from_fasta(args.fasta)
    chrom_sizes = filter_chroms(chrom_sizes, args.include_regex, args.exclude_regex)

    # 2) contigs
    contigs = make_contigs(
        chrom_sizes=chrom_sizes,
        window_len=args.window_len,
        stride=args.stride,
        drop_last=args.drop_last,
    )

    n_samples = len(contigs)

    if args.window_len % args.resolution != 0:
        raise ValueError("window_len must be divisible by resolution.")

    total_bins = args.window_len // args.resolution  # expected 1024
    valid_bins = total_bins - 2 * args.crop          # expected 896

    if total_bins != 1024:
        print(f"[WARN] window_len/resolution = {total_bins} bins (expected 1024).")
    if valid_bins <= 0:
        raise ValueError(f"Invalid crop={args.crop}: total_bins={total_bins} -> valid_bins={valid_bins}")

    print(
        f"[INFO] chroms={len(chrom_sizes)} contigs(samples)={n_samples} "
        f"total_bins={total_bins} valid_bins={valid_bins} channels={args.out_channels}"
    )

    # 3) data loader
    ds = EPCOTInferDataset(contigs, args.fasta, args.atac_bw, args.window_len)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in device),
    )

    # 4) load models
    model_succeed = load_succeed(args.succeed_ckpt, device=device)
    model_epcot = load_epcot(args.epcot_ckpt, device=device, out_channels=args.out_channels, head_name=args.head_name)

    # 5) create h5 and write (STORE CROPPED BINS)
    pred_chunk = (max(1, args.chunk_bs), valid_bins, args.out_channels)
    h5f = create_h5(
        out_h5=args.out_h5,
        n_samples=n_samples,
        n_bins=valid_bins,
        n_channels=args.out_channels,
        compression=args.h5_compression,
        compression_opts=args.h5_level,
        pred_chunk=pred_chunk,
    )

    # metadata
    h5f.attrs["fasta"] = args.fasta
    h5f.attrs["atac_bw"] = args.atac_bw
    h5f.attrs["window_len"] = args.window_len
    h5f.attrs["resolution"] = args.resolution
    h5f.attrs["stride"] = args.stride
    h5f.attrs["drop_last"] = bool(args.drop_last)
    h5f.attrs["head_name"] = args.head_name
    h5f.attrs["out_channels"] = args.out_channels
    h5f.attrs["crop_bins"] = int(args.crop)
    h5f.attrs["total_bins"] = int(total_bins)
    h5f.attrs["valid_bins"] = int(valid_bins)

    try:
        written = run_inference_to_h5(
            model_epcot=model_epcot,
            model_succeed=model_succeed,
            loader=loader,
            h5f=h5f,
            device=device,
            head_name=args.head_name,
            window_len=args.window_len,
            resolution=args.resolution,
            out_channels=args.out_channels,
            crop=args.crop,
        )
        print(f"✅ Wrote H5: {args.out_h5}  samples={written}  shape=({written},{valid_bins},{args.out_channels})")
    finally:
        h5f.close()


if __name__ == "__main__":
    main()
