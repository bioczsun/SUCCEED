#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genome-wide SUCCEED_EPI inference from FASTA + one cell-specific bigWig.

This script follows the data/model path used by train_HistonTF.py and the EPCOT
evaluation notebook:
  seq  : dna_1hot -> (B, 4, 1048576)
  track: bigWig values -> (B, 1, 1048576)
  emb  : frozen SUCCEED(seq)
  pred : SUCCEED_EPI(track, emb)["human"]

Unlike the older inference_h5.py, genome tiling is based on the model's valid
central output span. The first window is left-padded so the first written bin is
chr:start=0, and chromosome tails are padded so terminal bins are not silently
dropped.
"""

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pysam
import pyBigWig
import torch
from tqdm import tqdm


_EFP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _EFP_DIR.parent
for _path in (str(_EFP_DIR), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from EFP import models
from EFP.config import ModelArgs as EFPArgs
from pretrain import layers
from pretrain.config import ModelArgs as SucceedArgs
from utils.GenomeDataset import dna_1hot


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chrom_sizes_from_fasta(fasta_path: str) -> List[Tuple[str, int]]:
    fasta = pysam.FastaFile(fasta_path)
    try:
        return [(chrom, fasta.get_reference_length(chrom)) for chrom in fasta.references]
    finally:
        fasta.close()


def filter_chrom_sizes(
    chrom_sizes: Sequence[Tuple[str, int]],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
) -> List[Tuple[str, int]]:
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    out: List[Tuple[str, int]] = []
    for chrom, size in chrom_sizes:
        if include and not include.search(chrom):
            continue
        if exclude and exclude.search(chrom):
            continue
        out.append((chrom, int(size)))
    return out


def make_windows(
    chrom_sizes: Sequence[Tuple[str, int]],
    crop_bp: int,
    valid_bp: int,
    stride_bp: int,
) -> List[Tuple[str, int, int]]:
    windows: List[Tuple[str, int, int]] = []
    for chrom, chrom_len in chrom_sizes:
        valid_start = 0
        while valid_start < chrom_len:
            window_start = valid_start - crop_bp
            windows.append((chrom, window_start, window_start + crop_bp * 2 + valid_bp))
            valid_start += stride_bp
    return windows


def _copy_with_padding(
    values: np.ndarray,
    requested_start: int,
    requested_end: int,
    chrom_len: int,
    fill_shape: Tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(fill_shape, dtype=np.float32)
    clipped_start = max(0, requested_start)
    clipped_end = min(chrom_len, requested_end)
    if clipped_end <= clipped_start:
        return out

    dst_start = clipped_start - requested_start
    dst_end = dst_start + (clipped_end - clipped_start)
    out[dst_start:dst_end] = values
    return out


def fetch_seq_1hot(
    fasta: pysam.FastaFile,
    chrom: str,
    start: int,
    end: int,
    chrom_len: int,
    window_len: int,
) -> np.ndarray:
    clipped_start = max(0, start)
    clipped_end = min(chrom_len, end)
    if clipped_end > clipped_start:
        seq = fasta.fetch(chrom, clipped_start, clipped_end).upper()
        seq_arr = dna_1hot(seq).astype(np.float32)
    else:
        seq_arr = np.zeros((0, 4), dtype=np.float32)
    return _copy_with_padding(seq_arr, start, end, chrom_len, (window_len, 4))


def fetch_bw_values(
    bw: pyBigWig.pyBigWig,
    chrom: str,
    start: int,
    end: int,
    chrom_len: int,
    window_len: int,
) -> np.ndarray:
    clipped_start = max(0, start)
    clipped_end = min(chrom_len, end)
    if clipped_end > clipped_start:
        values = bw.values(chrom, clipped_start, clipped_end)
        values_arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
    else:
        values_arr = np.zeros((0,), dtype=np.float32)
    return _copy_with_padding(values_arr, start, end, chrom_len, (window_len,))


class GenomeInferDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        windows: Sequence[Tuple[str, int, int]],
        chrom_len_dict: Dict[str, int],
        fasta_path: str,
        bigwig_path: str,
        window_len: int,
    ):
        self.windows = list(windows)
        self.chrom_len_dict = dict(chrom_len_dict)
        self.fasta_path = fasta_path
        self.bigwig_path = bigwig_path
        self.window_len = window_len
        self._fasta = None
        self._bw = None

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_path)
        return self._fasta

    @property
    def bw(self):
        if self._bw is None:
            self._bw = pyBigWig.open(self.bigwig_path)
            if self._bw is None:
                raise RuntimeError(f"Failed to open bigWig: {self.bigwig_path}")
        return self._bw

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        chrom, start, end = self.windows[idx]
        chrom_len = self.chrom_len_dict[chrom]
        seq = fetch_seq_1hot(self.fasta, chrom, start, end, chrom_len, self.window_len)
        track = fetch_bw_values(self.bw, chrom, start, end, chrom_len, self.window_len)
        return torch.from_numpy(seq), torch.from_numpy(track), chrom, np.int64(start)


def load_succeed(ckpt_path: str, device: torch.device):
    args = SucceedArgs()
    args.device = device
    model = layers.SUCCEED(args, return_emb=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state = {key: value for key, value in state.items() if not key.startswith("_heads.")}
    incompat = model.load_state_dict(state, strict=False)
    print("SUCCEED missing:", incompat.missing_keys)
    print("SUCCEED unexpected:", incompat.unexpected_keys)

    for param in model.parameters():
        param.requires_grad = False
    return model.to(device).eval()


def load_efp(
    ckpt_path: str,
    device: torch.device,
    head_name: str,
    out_channels: int,
    target_seq_len: int,
):
    args = EFPArgs()
    args.output_heads = {head_name: out_channels}
    args.depth = 4
    args.device = device
    args.target_seq_len = target_seq_len

    model = models.SUCCEED_EPI(args).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model.eval()


def create_h5(
    out_h5: str,
    n_channels: int,
    chrom_sizes: Sequence[Tuple[str, int]],
    args,
    chunk_bins: int,
) -> h5py.File:
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    h5f = h5py.File(out_h5, "w")
    str_dt = h5py.string_dtype("utf-8")

    h5f.create_dataset(
        "pred",
        shape=(0, n_channels),
        maxshape=(None, n_channels),
        dtype=np.float32,
        chunks=(max(1, chunk_bins), n_channels),
        compression=args.h5_compression,
        compression_opts=args.h5_level,
        shuffle=True,
    )
    for name, dtype in (("chrom", str_dt), ("start", np.int64), ("end", np.int64), ("coverage", np.int32)):
        h5f.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(max(1, chunk_bins),),
            compression=args.h5_compression,
            compression_opts=args.h5_level,
        )

    h5f.create_dataset("chrom_sizes_chrom", data=np.asarray([c for c, _ in chrom_sizes], dtype=object), dtype=str_dt)
    h5f.create_dataset("chrom_sizes_length", data=np.asarray([l for _, l in chrom_sizes], dtype=np.int64))

    h5f.attrs["fasta"] = args.fasta
    h5f.attrs["bigwig"] = args.bigwig
    h5f.attrs["succeed_ckpt"] = args.succeed_ckpt
    h5f.attrs["efp_ckpt"] = args.efp_ckpt
    h5f.attrs["window_len"] = args.window_len
    h5f.attrs["resolution"] = args.resolution
    h5f.attrs["stride_bp"] = args.stride_bp
    h5f.attrs["target_seq_len"] = args.target_seq_len
    h5f.attrs["head_name"] = args.head_name
    h5f.attrs["out_channels"] = args.out_channels
    h5f.attrs["crop_bins"] = args.crop_bins
    h5f.attrs["crop_bp"] = args.crop_bins * args.resolution
    h5f.attrs["valid_bp"] = args.target_seq_len * args.resolution
    h5f.attrs["output_mode"] = "genome_wide_averaged_bins"
    h5f.attrs["overlap_aggregation"] = "mean"
    return h5f


def append_chrom_predictions(
    h5f: h5py.File,
    chrom: str,
    pred_sum: np.ndarray,
    pred_count: np.ndarray,
    chrom_len: int,
    resolution: int,
) -> int:
    covered = pred_count > 0
    if not np.any(covered):
        return 0

    idx = np.nonzero(covered)[0]
    pred = pred_sum[idx] / pred_count[idx, None].astype(np.float32)
    starts = idx.astype(np.int64) * np.int64(resolution)
    ends = np.minimum(starts + np.int64(resolution), np.int64(chrom_len))
    keep = starts < ends
    idx = idx[keep]
    pred = pred[keep]
    starts = starts[keep]
    ends = ends[keep]
    if len(idx) == 0:
        return 0

    old = h5f["pred"].shape[0]
    new = old + len(idx)
    h5f["pred"].resize((new, h5f["pred"].shape[1]))
    h5f["pred"][old:new] = pred.astype(np.float32, copy=False)
    for key in ("chrom", "start", "end", "coverage"):
        h5f[key].resize((new,))
    h5f["chrom"][old:new] = [chrom] * len(idx)
    h5f["start"][old:new] = starts
    h5f["end"][old:new] = ends
    h5f["coverage"][old:new] = pred_count[idx].astype(np.int32, copy=False)
    return len(idx)


def iter_h5_chunks(h5f: h5py.File, chunk_size: int):
    n_rows = h5f["pred"].shape[0]
    for offset in range(0, n_rows, chunk_size):
        stop = min(offset + chunk_size, n_rows)
        chroms = h5f["chrom"][offset:stop]
        chroms = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in chroms]
        yield chroms, h5f["start"][offset:stop], h5f["end"][offset:stop], h5f["pred"][offset:stop]


def parse_channels(spec: str, n_channels: int) -> List[int]:
    if spec.lower() == "all":
        return list(range(n_channels))
    channels: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            channels.extend(range(int(left), int(right) + 1))
        else:
            channels.append(int(part))
    unique = []
    seen = set()
    for channel in channels:
        if channel < 0 or channel >= n_channels:
            raise ValueError(f"Channel {channel} is outside [0, {n_channels - 1}]")
        if channel not in seen:
            seen.add(channel)
            unique.append(channel)
    if not unique:
        raise ValueError("No bigWig channels selected.")
    return unique


def write_bigwigs_from_h5(
    h5f: h5py.File,
    chrom_sizes: Sequence[Tuple[str, int]],
    out_dir: str,
    channels: Sequence[int],
    chunk_size: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for channel in channels:
        out_bw = os.path.join(out_dir, f"channel_{channel}.bw")
        bw = pyBigWig.open(out_bw, "w")
        if bw is None:
            raise RuntimeError(f"Failed to create bigWig: {out_bw}")
        try:
            bw.addHeader(list(chrom_sizes))
            written = 0
            for chroms, starts, ends, pred in iter_h5_chunks(h5f, chunk_size):
                values = pred[:, channel].astype(float)
                finite = np.isfinite(values)
                if not np.any(finite):
                    continue
                kept_chroms = [chrom for chrom, ok in zip(chroms, finite) if ok]
                bw.addEntries(
                    kept_chroms,
                    starts[finite].astype(int).tolist(),
                    ends=ends[finite].astype(int).tolist(),
                    values=values[finite].astype(float).tolist(),
                )
                written += int(finite.sum())
            print(f"[OK] wrote channel {channel}: {out_bw} intervals={written}")
        finally:
            bw.close()


@torch.no_grad()
def run_inference(
    loader: torch.utils.data.DataLoader,
    model_succeed,
    model_efp,
    h5f: h5py.File,
    chrom_len_dict: Dict[str, int],
    device: torch.device,
    args,
) -> int:
    n_chrom_bins = 0
    current_chrom = None
    pred_sum = None
    pred_count = None
    total_written = 0

    for seq, track, chroms, window_starts in tqdm(loader, desc="Genome inference"):
        seq = seq.to(device).float().transpose(1, 2)
        track = track.to(device).float().unsqueeze(1)

        emb = model_succeed(seq)
        out = model_efp(track, emb)[args.head_name]
        if out.shape[1] != args.target_seq_len or out.shape[2] != args.out_channels:
            raise ValueError(
                f"Unexpected output shape {tuple(out.shape)}; "
                f"expected (B, {args.target_seq_len}, {args.out_channels})"
            )
        out_np = out.detach().cpu().numpy().astype(np.float32)

        for i, chrom in enumerate(chroms):
            chrom = str(chrom)
            if chrom != current_chrom:
                if current_chrom is not None:
                    total_written += append_chrom_predictions(
                        h5f, current_chrom, pred_sum, pred_count,
                        chrom_len_dict[current_chrom], args.resolution,
                    )
                current_chrom = chrom
                n_chrom_bins = int(math.ceil(chrom_len_dict[chrom] / args.resolution))
                pred_sum = np.zeros((n_chrom_bins, args.out_channels), dtype=np.float32)
                pred_count = np.zeros((n_chrom_bins,), dtype=np.int32)

            valid_start_bp = int(window_starts[i]) + args.crop_bins * args.resolution
            chrom_bin_start = math.floor(valid_start_bp / args.resolution)
            out_start = 0
            if chrom_bin_start < 0:
                out_start = -chrom_bin_start
                chrom_bin_start = 0

            keep_bins = min(args.target_seq_len - out_start, n_chrom_bins - chrom_bin_start)
            if keep_bins <= 0:
                continue
            pred_sum[chrom_bin_start:chrom_bin_start + keep_bins] += out_np[i, out_start:out_start + keep_bins]
            pred_count[chrom_bin_start:chrom_bin_start + keep_bins] += 1

    if current_chrom is not None:
        total_written += append_chrom_predictions(
            h5f, current_chrom, pred_sum, pred_count,
            chrom_len_dict[current_chrom], args.resolution,
        )
    return total_written


def main() -> None:
    parser = argparse.ArgumentParser("Genome-wide SUCCEED_EPI inference from FASTA + cell-specific bigWig")
    parser.add_argument("--fasta", required=True, help="Reference FASTA with .fai index")
    parser.add_argument("--bigwig", required=True, help="Cell-specific input bigWig, e.g. ATAC RPGC")
    parser.add_argument("--succeed_ckpt", required=True, help="Frozen SUCCEED checkpoint")
    parser.add_argument("--efp_ckpt", required=True, help="Trained SUCCEED_EPI/EFP checkpoint")
    parser.add_argument("--out_h5", required=True, help="Output HDF5 with pred/chrom/start/end/coverage")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--window_len", type=int, default=1048576)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--target_seq_len", type=int, default=896)
    parser.add_argument(
        "--stride_bp",
        type=int,
        default=0,
        help="Stride between valid output spans. Default: target_seq_len * resolution, no valid-bin overlap.",
    )
    parser.add_argument("--head_name", default="human")
    parser.add_argument("--out_channels", type=int, default=46)

    parser.add_argument("--include_regex", default=None, help='Example: "^chr([0-9]+|X|Y)$"')
    parser.add_argument("--exclude_regex", default="_|Un|random|alt|chrM", help="Default excludes alt/random/chrM-like contigs")

    parser.add_argument("--h5_compression", default="gzip")
    parser.add_argument("--h5_level", type=int, default=4)
    parser.add_argument("--chunk_bins", type=int, default=8192)

    parser.add_argument("--out_bw_dir", default="", help="Optional directory to also write selected channels as bigWig")
    parser.add_argument("--bw_channels", default="all", help='For --out_bw_dir: "all", "0", "0,3,5", or "0-4"')
    parser.add_argument("--bw_chunk_size", type=int, default=200000)
    args = parser.parse_args()

    if args.window_len % args.resolution != 0:
        raise ValueError("--window_len must be divisible by --resolution")
    total_bins = args.window_len // args.resolution
    crop_total = total_bins - args.target_seq_len
    if crop_total < 0 or crop_total % 2 != 0:
        raise ValueError(
            f"window_len/resolution={total_bins} and target_seq_len={args.target_seq_len} "
            "must leave an even non-negative crop."
        )
    args.crop_bins = crop_total // 2
    if args.stride_bp <= 0:
        args.stride_bp = args.target_seq_len * args.resolution
    if args.stride_bp % args.resolution != 0:
        raise ValueError("--stride_bp must be divisible by --resolution")
    if args.stride_bp > args.target_seq_len * args.resolution:
        raise ValueError("--stride_bp cannot exceed target_seq_len * resolution, or bins will be skipped")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    chrom_sizes = load_chrom_sizes_from_fasta(args.fasta)
    chrom_sizes = filter_chrom_sizes(chrom_sizes, args.include_regex, args.exclude_regex)
    if not chrom_sizes:
        raise ValueError("No chromosomes remain after filtering.")
    chrom_len_dict = dict(chrom_sizes)

    windows = make_windows(
        chrom_sizes=chrom_sizes,
        crop_bp=args.crop_bins * args.resolution,
        valid_bp=args.target_seq_len * args.resolution,
        stride_bp=args.stride_bp,
    )
    print(
        f"[INFO] chroms={len(chrom_sizes)} windows={len(windows)} "
        f"window_bins={total_bins} crop_bins={args.crop_bins} "
        f"target_seq_len={args.target_seq_len} stride_bp={args.stride_bp}"
    )

    dataset = GenomeInferDataset(windows, chrom_len_dict, args.fasta, args.bigwig, args.window_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in str(device)),
    )

    model_succeed = load_succeed(args.succeed_ckpt, device)
    model_efp = load_efp(args.efp_ckpt, device, args.head_name, args.out_channels, args.target_seq_len)

    h5f = create_h5(args.out_h5, args.out_channels, chrom_sizes, args, args.chunk_bins)
    try:
        written = run_inference(loader, model_succeed, model_efp, h5f, chrom_len_dict, device, args)
        h5f.flush()
        print(f"[OK] wrote H5: {args.out_h5} rows={written} channels={args.out_channels}")

        if args.out_bw_dir:
            channels = parse_channels(args.bw_channels, args.out_channels)
            write_bigwigs_from_h5(h5f, chrom_sizes, args.out_bw_dir, channels, args.bw_chunk_size)
    finally:
        h5f.close()


if __name__ == "__main__":
    main()
