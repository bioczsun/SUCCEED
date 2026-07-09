import argparse
import csv
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.clip_grad as clip_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train SUCCEED pretrain model")
    parser.add_argument("--project_dir", required=True, help="Project root directory (must contain `src/`)")
    parser.add_argument("--td", required=True, help="Training data H5 path")
    parser.add_argument("--vd", required=True, help="Validation data H5 path")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--use_pth", required=False, help="Optional pretrained checkpoint (.pth)")
    parser.add_argument("--seed", default=1401, type=int, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda/cpu")
    parser.add_argument("--epoch", default=5000, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    return parser.parse_args()


def evaluate(model, valid_data, loss_fn, corr_coef, target_crop, device):
    model.eval()
    evaluate_loss = 0.0
    total_r = 0.0

    with torch.no_grad():
        corr_coef.reset()
        for idx, (seq, target) in enumerate(valid_data):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)

            outputs = model(seq)["human"]
            loss = loss_fn(outputs, target)

            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()

            if idx % 5 == 0:
                valid_data.set_description(f"Valid -- loss: {loss.item():.3f} pearson_r: {pearson_r:.3f}")

            evaluate_loss += loss.item()
            total_r += pearson_r

    avg_loss = evaluate_loss / len(valid_data)
    avg_pearson_r = total_r / len(valid_data)
    return avg_loss, avg_pearson_r


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def train(run_folder, model, train_data, valid_data, optimizer, loss_fn, corr_coef, target_crop, device, epochs, early_stopping):
    metrics_path = os.path.join(run_folder, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_pearson_r", "valid_loss", "valid_pearson_r"])

    model.train()
    for epoch in range(epochs):
        train_par = tqdm(
            train_data,
            bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        total_loss = 0.0
        total_r = 0.0

        corr_coef.reset()
        train_data.dataset.set_epoch(epoch)

        for idx, (seq, target) in enumerate(train_par):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(seq)["human"]
            loss = loss_fn(outputs, target)
            loss.backward()
            clip_grad.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()

            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()
            if idx % 5 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                train_par.set_description(
                    f"Train -- epoch: [{epoch}/{epochs}] loss: {loss.item():.3f} pearson_r: {pearson_r:.3f} lr: {current_lr:.2e}"
                )

            total_loss += loss.item()
            total_r += pearson_r

        avg_loss = total_loss / len(train_data)
        avg_pearson_r = total_r / len(train_data)

        valid_par = tqdm(
            valid_data,
            bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        valid_loss, valid_pearson_r = evaluate(model, valid_par, loss_fn, corr_coef, target_crop, device)

        print(
            f"Epoch: [{epoch}/{epochs}] "
            f"train loss: {avg_loss:.3f} train pearson_r: {avg_pearson_r:.3f} "
            f"valid loss: {valid_loss:.3f} valid pearson_r: {valid_pearson_r:.3f}"
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_pearson_r.item(), valid_loss, valid_pearson_r.item()])

        early_stopping(valid_loss, model, optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def main():
    args = parse_arguments()
    sys.path.append(os.path.join(args.project_dir, "src"))

    from pretrain.config import ModelArgs
    from pretrain.layers import SUCCEED, TargetLengthCrop
    from utils.GenomeDataset import PretrainDataset
    from utils.metric import EarlyStopping, MeanPearsonCorrCoefPerChannel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    output_head_size = ModelArgs().output_heads["human"]

    log_dir = os.path.join(out_dir, "csv", "logs")
    os.makedirs(log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1]) for d in os.listdir(log_dir) if d.startswith("version_")
    ]
    new_version = max(existing_versions) + 1 if existing_versions else 0
    run_folder = os.path.join(log_dir, f"version_{new_version}")
    os.makedirs(run_folder, exist_ok=True)

    if yaml is not None:
        hparams = {
            "train_data": args.td,
            "valid_data": args.vd,
            "model_name": args.name,
            "use_pth": args.use_pth,
            "seed": args.seed,
            "device": device,
            "epoch": args.epoch,
            "learning_rate": args.lr,
            "batch_size": args.batch,
            "output_dir": out_dir,
            "output_head_size": output_head_size,
        }
        with open(os.path.join(run_folder, "hparams.yaml"), "w") as f:
            yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)

    train_h5 = h5py.File(args.td, "r", swmr=True)
    val_h5 = h5py.File(args.vd, "r", swmr=True)

    train_dataset = PretrainDataset(train_h5, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=10, pin_memory=True)

    val_dataset = PretrainDataset(val_h5, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=10, pin_memory=True)

    model = SUCCEED(ModelArgs(device=device))
    if args.use_pth:
        checkpoint_obj = torch.load(args.use_pth, map_location="cpu")
        state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj)
        state_dict = strip_module_prefix(state_dict)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_heads.")}
        model.load_state_dict(filtered_state_dict, strict=False)

    model.to(device)
    if torch.cuda.device_count() > 1 and device.startswith("cuda"):
        print(f"Using {torch.cuda.device_count()} GPUs with torch.nn.DataParallel")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.PoissonNLLLoss(log_input=False)

    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=output_head_size).to(device)
    target_crop = TargetLengthCrop(ModelArgs().target_seq_len)

    early_stopping = EarlyStopping(save_path=out_dir, model_name=args.name, patience=8)

    train(
        run_folder,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        corr_coef,
        target_crop,
        device,
        args.epoch,
        early_stopping,
    )


if __name__ == "__main__":
    main()
