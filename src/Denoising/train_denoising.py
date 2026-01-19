import argparse
import os
import random
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchmetrics import AUROC



def ensure_src_on_path(project_dir: Optional[str]) -> None:
    if project_dir:
        src_path = Path(project_dir) / "src"
    else:
        src_path = Path(__file__).resolve().parents[1]
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ATACwork denoising model")
    parser.add_argument("--project-dir", default=None, help="Project root; adds <root>/src to PYTHONPATH")
    parser.add_argument("--sequence-bed", required=True, help="Path to sequence.bed")
    parser.add_argument("--fasta", required=True, help="Reference FASTA path")
    parser.add_argument("--val-chroms", nargs="+", default=["chr20"], help="Chroms for validation split")
    parser.add_argument(
        "--train-exclude-chroms",
        nargs="+",
        default=["chr20", "chr10"],
        help="Chroms to exclude from training",
    )
    parser.add_argument("--res", default=200000, help="Read depth tag for naming")
    parser.add_argument("--seed", type=int, default=1401, help="Random seed")
    parser.add_argument("--device", default="cuda:0", help="Device string")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--outpath", required=True, help="Output directory for checkpoints")
    parser.add_argument("--run-name", default=None, help="Checkpoint name override")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--model-ckpt", required=True, help="SUCCEED checkpoint path")
    parser.add_argument(
        "--cell-type-spec",
        action="append",
        required=True,
        help="Repeatable: name,noisy_bw_or_dir,clean_bw,peak_bed",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_cell_specs(specs):
    parsed = []
    for spec in specs:
        parts = spec.split(",")
        if len(parts) != 4:
            raise ValueError("--cell-type-spec must be: name,noisy_bw_or_dir,clean_bw,peak_bed")
        name, noisy_bw, clean_bw, peak_bed = [p.strip() for p in parts]
        parsed.append({"name": name, "noisy": noisy_bw, "clean": clean_bw, "peak": peak_bed})
    return parsed





def build_datasets(train_contigs, val_contigs, fasta, cell_specs):
    from utils.GenomeDataset import ATACDenoisingDataset, NoisyDirATACDataset
    train_sets = []
    val_sets = []
    for spec in cell_specs:
        if os.path.isdir(spec["noisy"]):
            train_sets.append(
                NoisyDirATACDataset(
                    contigs=train_contigs,
                    fasta=fasta,
                    noisy_dir=spec["noisy"],
                    clean_file=spec["clean"],
                    peak_file=spec["peak"],
                )
            )
            val_sets.append(
                NoisyDirATACDataset(
                    contigs=val_contigs,
                    fasta=fasta,
                    noisy_dir=spec["noisy"],
                    clean_file=spec["clean"],
                    peak_file=spec["peak"],
                )
            )
        else:
            train_sets.append(
                ATACDenoisingDataset(
                    contigs=train_contigs,
                    fasta=fasta,
                    noisy_file=spec["noisy"],
                    clean_file=spec["clean"],
                    peak_file=spec["peak"],
                )
            )
            val_sets.append(
                ATACDenoisingDataset(
                    contigs=val_contigs,
                    fasta=fasta,
                    noisy_file=spec["noisy"],
                    clean_file=spec["clean"],
                    peak_file=spec["peak"],
                )
            )
    return train_sets, val_sets


def main() -> None:
    args = parse_args()
    ensure_src_on_path(args.project_dir)
    from utils.metric import compute_rowwise_pearson, EarlyStopping
    import src.Denoising.models as models
    set_random_seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"

    sequence_bed = pd.read_csv(args.sequence_bed, sep="\t", header=None)
    sequence_bed.columns = ["chrom", "start", "end", "name"]

    train_index = sequence_bed[~sequence_bed["chrom"].isin(args.train_exclude_chroms)]
    val_index = sequence_bed[sequence_bed["chrom"].isin(args.val_chroms)]

    cell_specs = parse_cell_specs(args.cell_type_spec)
    train_sets, val_sets = build_datasets(train_index.values, val_index.values, args.fasta, cell_specs)

    from torch.utils.data import ConcatDataset

    dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    from pretrain import layers
    from EFP.config import ModelArgs

    config_args = ModelArgs()
    config_args.device = device

    model_epi = layers.SUCCEED(config_args, return_emb=True)
    checkpoint = torch.load(args.model_ckpt, map_location="cpu")
    filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('_heads.')}
    model_epi.load_state_dict(filtered_state_dict, strict=False)
    model_epi.to(config_args.device)

    for param in model_epi.parameters():
        param.requires_grad = False

    model = models.ATACDenoising(config_args)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    auroc = AUROC(task="multilabel", num_labels=1024).to(device)

    run_name = args.run_name or f"SUCCEED_{args.res}_seed_{args.seed}"
    early_stopping = EarlyStopping(save_path=args.outpath, model_name=run_name, patience=args.patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=8, min_lr=1e-5
    )

    def evaluate(epoch):
        model.eval()
        model_epi.eval()
        total_loss = 0
        total_r = 0
        tmp_r = 0
        total_auroc = 0
        par = tqdm(val_loader, bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        with torch.no_grad():
            for i, (seq, clean, noisy, label) in enumerate(par):
                seq = seq.to(device).float().transpose(1, 2)
                sample = noisy.to(device).float().unsqueeze(1)
                target = clean.to(device)
                label = label.to(device)

                embed = model_epi(seq)
                signal, peak, origin_mean = model(sample, embed)

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
                if i % 50 == 0:
                    par.set_description(
                        f"Valid--Epoch [{epoch} / {args.epochs}] Loss_mse {loss_mse.item():.3f} "
                        f"Loss_bce {loss_binary.item():.3f} R {r_val:.3f} auc {auroc_val:.3f}",
                        refresh=True,
                    )

        avg_loss = total_loss / len(val_loader)
        avg_r = total_r / len(val_loader)
        avg_tmp_r = tmp_r / len(val_loader)
        avg_auroc = total_auroc / len(val_loader)
        return avg_loss, avg_r, avg_auroc, avg_tmp_r

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        par = tqdm(train_loader, bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for i, (seq, clean, noisy, label) in enumerate(par):
            target = clean.to(device)
            label = label.to(device)

            seq = seq.to(device).float().transpose(1, 2)
            sample = noisy.to(device).float().unsqueeze(1)
            optimizer.zero_grad()

            embed = model_epi(seq)
            signal, peak, origin_mean = model(sample, embed)

            loss_mse = loss_fn(signal, target)
            loss_binary = loss_bce(peak, label)
            r_tra_orig = compute_rowwise_pearson(signal, target).mean()
            r_train_tmp = compute_rowwise_pearson(target, origin_mean).mean()

            loss = loss_mse + loss_binary
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 50 == 0:
                train_auroc = auroc(peak.detach(), label.detach().long()).mean()
                par.set_description(
                    f"Train--Epoch: [{epoch} / {args.epochs}] Loss {loss.item():.3f} "
                    f"Loss_bce {loss_binary.item():.3f} R {r_tra_orig:.3f} "
                    f"auroc {train_auroc:.3f} tmp_r {r_train_tmp:.3f}",
                    refresh=True,
                )

        val_loss, val_r, val_auroc, tmp_r = evaluate(epoch)
        print(
            f"Epoch: [{epoch} / {args.epochs}] "
            f"Train Loss: {(total_loss / len(train_loader)):.3f} "
            f"Val Loss: {val_loss:.3f} Val R: {val_r:.3f} "
            f"auroc: {val_auroc:.3f} tmp_r: {tmp_r:.3f}"
        )

        if isinstance(model, nn.DataParallel):
            early_stopping(-val_r, model.module, optimizer)
        else:
            early_stopping(-val_r, model, optimizer)

        scheduler.step(-val_r)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
