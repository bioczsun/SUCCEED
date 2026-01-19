import argparse
import csv
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate SUCCEED pretrain model")
    parser.add_argument("--project_dir", required=True, help="Project root directory (must contain `src/`)")
    parser.add_argument("--data", required=True, help="Evaluation data H5 path")
    parser.add_argument("--name", required=True, help="Model name (for logging only)")
    parser.add_argument("--use_pth", required=True, help="Model checkpoint path (.pth)")
    parser.add_argument("--seed", default=1401, type=int, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda/cpu")
    parser.add_argument("--batch", default=4, type=int, help="Batch size")
    parser.add_argument("--out_dir", required=True, help="Output directory for evaluation artifacts")
    return parser.parse_args()


def evaluate(model, data_loader, loss_fn, corr_coef, target_crop, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        corr_coef.reset()
        eval_par = tqdm(
            data_loader,
            bar_format="{l_bar}{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        for idx, (seq, target) in enumerate(eval_par):
            seq = seq.to(device).float().transpose(1, 2)
            target = target.to(device).transpose(1, 2)
            target = target_crop(target)

            outputs = model(seq)["human"]
            loss = loss_fn(outputs, target)

            corr_coef(outputs, target)
            pearson_r = corr_coef.compute().mean()

            if idx % 5 == 0:
                eval_par.set_description(f"Eval -- loss: {loss.item():.3f} pearson_r: {pearson_r:.3f}")

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    avg_pearson_r = corr_coef.compute().mean()
    return avg_loss, avg_pearson_r


def main():
    args = parse_arguments()

    project_dir = args.project_dir
    sys.path.append(os.path.join(project_dir, "src"))

    # Project imports (after sys.path is updated)
    from pretrain.config import ModelArgs
    from pretrain.layers import SUCCEED, TargetLengthCrop
    from utils.GenomeDataset import PretrainDataset
    from utils.metric import MeanPearsonCorrCoefPerChannel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    output_head_size = ModelArgs().output_heads["human"]

    # Create a versioned run folder
    log_dir = os.path.join(out_dir, "csv", "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1]) for d in os.listdir(log_dir) if d.startswith("version_")
    ]
    new_version = max(existing_versions) + 1 if existing_versions else 0
    run_folder = os.path.join(log_dir, f"version_{new_version}")
    os.makedirs(run_folder, exist_ok=True)

    # Save hyperparameters (optional)
    if yaml is not None:
        hparams = {
            "eval_data": args.data,
            "model_name": args.name,
            "use_pth": args.use_pth,
            "seed": args.seed,
            "device": device,
            "batch_size": args.batch,
            "output_dir": out_dir,
            "output_head_size": output_head_size,
        }
        with open(os.path.join(run_folder, "hparams.yaml"), "w") as f:
            yaml.dump(hparams, f, default_flow_style=False, sort_keys=False)

    eval_data_h5 = h5py.File(args.data, "r", swmr=True)
    eval_dataset = PretrainDataset(eval_data_h5, augment=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    model = SUCCEED(ModelArgs(device=device))
    checkpoint_obj = torch.load(args.use_pth, map_location="cpu")
    state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj)
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print("Missing keys:", incompat.missing_keys)
        print("Unexpected keys:", incompat.unexpected_keys)

    model.to(device)

    loss_fn = nn.PoissonNLLLoss(log_input=False)
    corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=output_head_size).to(device)
    target_crop = TargetLengthCrop(ModelArgs().target_seq_len)

    avg_loss, avg_pearson_r = evaluate(model, eval_loader, loss_fn, corr_coef, target_crop, device)

    print("========== Evaluation Result ==========")
    print(f"Average loss      : {avg_loss:.6f}")
    print(f"Average Pearson r : {avg_pearson_r:.6f}")

    metrics_path = os.path.join(run_folder, "eval_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["avg_loss", "avg_pearson_r"])
        writer.writerow([avg_loss, avg_pearson_r.item()])

    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
