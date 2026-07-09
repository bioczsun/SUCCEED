#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert averaged EPCOT/SUCCEED HDF5 predictions to bigWig.

Expected HDF5 layout from inference_h5.py:
  pred:  float32 (M, channels)
  chrom: string  (M,)
  start: int64   (M,)
  end:   int64   (M,)

Examples:
  python src/EFP/h5_to_bigwig.py \
    --h5 pred.h5 \
    --channels 0,3,5-7 \
    --out_prefix pred_channel

  python src/EFP/h5_to_bigwig.py \
    --h5 pred.h5 \
    --channels 0 \
    --out_bw pred_channel0.bw
"""

import argparse
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pyBigWig


def parse_channels(spec: str, n_channels: int) -> List[int]:
    channels: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid channel range: {part}")
            channels.extend(range(start, end + 1))
        else:
            channels.append(int(part))

    seen = set()
    unique = []
    for ch in channels:
        if ch < 0 or ch >= n_channels:
            raise ValueError(f"Channel {ch} out of range [0, {n_channels - 1}]")
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    if not unique:
        raise ValueError("No channels selected.")
    return unique


def decode_chrom_array(values: Sequence) -> List[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def read_fai(fai_path: str) -> List[Tuple[str, int]]:
    chrom_sizes = []
    with open(fai_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            chrom_sizes.append((fields[0], int(fields[1])))
    return chrom_sizes


def read_chrom_sizes(path: str) -> List[Tuple[str, int]]:
    chrom_sizes = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = re.split(r"\s+", line.strip())
            if len(fields) < 2:
                raise ValueError(f"Invalid chrom sizes line: {line.rstrip()}")
            chrom_sizes.append((fields[0], int(fields[1])))
    return chrom_sizes


def chrom_sizes_from_h5(h5f: h5py.File) -> List[Tuple[str, int]]:
    chroms = decode_chrom_array(h5f["chrom"][:])
    ends = h5f["end"][:]
    max_end: Dict[str, int] = {}
    order: List[str] = []
    for chrom, end in zip(chroms, ends):
        if chrom not in max_end:
            order.append(chrom)
            max_end[chrom] = int(end)
        elif int(end) > max_end[chrom]:
            max_end[chrom] = int(end)
    return [(chrom, max_end[chrom]) for chrom in order]


def load_chrom_sizes(args, h5f: h5py.File) -> List[Tuple[str, int]]:
    if args.chrom_sizes:
        return read_chrom_sizes(args.chrom_sizes)

    fasta = args.fasta
    if fasta is None and "fasta" in h5f.attrs:
        fasta_attr = h5f.attrs["fasta"]
        fasta = fasta_attr.decode("utf-8") if isinstance(fasta_attr, bytes) else str(fasta_attr)

    if fasta:
        fai = fasta + ".fai"
        if os.path.exists(fai):
            return read_fai(fai)
        if fasta.endswith(".fai") and os.path.exists(fasta):
            return read_fai(fasta)
        raise FileNotFoundError(f"Could not find FASTA index: {fai}")

    if args.infer_chrom_sizes:
        return chrom_sizes_from_h5(h5f)

    raise ValueError("Provide --chrom_sizes or --fasta, or use --infer_chrom_sizes.")


def filter_chrom_sizes(
    chrom_sizes: List[Tuple[str, int]],
    include_regex: str = None,
    exclude_regex: str = None,
) -> List[Tuple[str, int]]:
    inc = re.compile(include_regex) if include_regex else None
    exc = re.compile(exclude_regex) if exclude_regex else None
    filtered = []
    for chrom, size in chrom_sizes:
        if inc and not inc.search(chrom):
            continue
        if exc and exc.search(chrom):
            continue
        filtered.append((chrom, size))
    return filtered


def output_path_for_channel(args, channel: int, n_selected: int) -> str:
    if args.out_bw:
        if n_selected != 1:
            raise ValueError("--out_bw can only be used when exactly one channel is selected.")
        return args.out_bw
    if not args.out_prefix:
        raise ValueError("Provide --out_prefix, or --out_bw for a single channel.")
    return f"{args.out_prefix}.ch{channel}.bw"


def iter_h5_chunks(h5f: h5py.File, chunk_size: int) -> Iterable[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]]:
    n = h5f["pred"].shape[0]
    for offset in range(0, n, chunk_size):
        stop = min(offset + chunk_size, n)
        chroms = decode_chrom_array(h5f["chrom"][offset:stop])
        starts = h5f["start"][offset:stop].astype(np.int64)
        ends = h5f["end"][offset:stop].astype(np.int64)
        pred = h5f["pred"][offset:stop, :]
        yield chroms, starts, ends, pred


def write_channel_bigwig(
    h5f: h5py.File,
    chrom_sizes: List[Tuple[str, int]],
    channel: int,
    out_bw: str,
    chunk_size: int,
    skip_nan: bool,
):
    chrom_size_dict = dict(chrom_sizes)
    os.makedirs(os.path.dirname(out_bw) or ".", exist_ok=True)

    bw = pyBigWig.open(out_bw, "w")
    if bw is None:
        raise RuntimeError(f"Failed to create bigWig: {out_bw}")

    try:
        bw.addHeader(chrom_sizes)
        written = 0
        skipped = 0
        for chroms, starts, ends, pred in iter_h5_chunks(h5f, chunk_size):
            values = pred[:, channel].astype(float)
            keep = np.ones(values.shape[0], dtype=bool)

            if skip_nan:
                keep &= np.isfinite(values)
            keep &= starts < ends
            keep &= np.array([chrom in chrom_size_dict for chrom in chroms], dtype=bool)

            if not np.any(keep):
                skipped += int(len(values))
                continue

            kept_chroms = [chrom for chrom, ok in zip(chroms, keep) if ok]
            kept_starts = starts[keep].astype(int).tolist()
            kept_ends = ends[keep].astype(int).tolist()
            kept_values = values[keep].astype(float).tolist()

            bw.addEntries(kept_chroms, kept_starts, ends=kept_ends, values=kept_values)
            written += len(kept_values)
            skipped += int(len(values) - len(kept_values))
    finally:
        bw.close()

    print(f"[OK] channel={channel} wrote {written} intervals -> {out_bw} (skipped={skipped})")


def main():
    p = argparse.ArgumentParser("Convert averaged H5 predictions to selected-channel bigWig files")
    p.add_argument("--h5", required=True, help="Input HDF5 from src/EFP/inference_h5.py")
    p.add_argument("--channels", required=True, help='Channel list/ranges, e.g. "0", "0,3,5", or "0-4"')
    p.add_argument("--out_prefix", default=None, help="Output prefix; writes <prefix>.ch<channel>.bw")
    p.add_argument("--out_bw", default=None, help="Output bigWig path for a single selected channel")

    p.add_argument("--chrom_sizes", default=None, help="Two-column chrom sizes file")
    p.add_argument("--fasta", default=None, help="Reference FASTA or FASTA .fai. Defaults to H5 attr 'fasta' when present")
    p.add_argument("--infer_chrom_sizes", action="store_true", help="Infer chrom sizes from max H5 end coordinate if no FASTA/chrom sizes are available")
    p.add_argument("--include_regex", default=None, help='Only include chromosomes matching this regex, e.g. "^chr([0-9]+|X|Y)$"')
    p.add_argument("--exclude_regex", default=None, help="Exclude chromosomes matching this regex")

    p.add_argument("--chunk_size", type=int, default=200000, help="Number of H5 rows to stream per chunk")
    p.add_argument("--keep_nan", action="store_true", help="Write NaN/Inf values instead of skipping them")
    args = p.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")

    with h5py.File(args.h5, "r") as h5f:
        required = ["pred", "chrom", "start", "end"]
        missing = [key for key in required if key not in h5f]
        if missing:
            raise KeyError(f"Missing required H5 dataset(s): {missing}")
        if h5f["pred"].ndim != 2:
            raise ValueError("Expected pred to have shape (M, channels). Re-run inference_h5.py with averaged-bin output.")

        n_channels = h5f["pred"].shape[1]
        channels = parse_channels(args.channels, n_channels)
        chrom_sizes = load_chrom_sizes(args, h5f)
        chrom_sizes = filter_chrom_sizes(chrom_sizes, args.include_regex, args.exclude_regex)
        if not chrom_sizes:
            raise ValueError("No chromosomes remain after filtering.")

        print(f"[INFO] rows={h5f['pred'].shape[0]} channels={n_channels} selected={channels}")
        print(f"[INFO] bigWig header chromosomes={len(chrom_sizes)}")

        for channel in channels:
            out_bw = output_path_for_channel(args, channel, len(channels))
            write_channel_bigwig(
                h5f=h5f,
                chrom_sizes=chrom_sizes,
                channel=channel,
                out_bw=out_bw,
                chunk_size=args.chunk_size,
                skip_nan=not args.keep_nan,
            )


if __name__ == "__main__":
    main()
