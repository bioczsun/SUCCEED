import sys
import os
from pathlib import Path

# Ensure `src/` is on sys.path so `import hic...` works when running this file directly.
# File location: <repo>/src/hic/training/main_alter.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import scipy.stats
import math
import hic.model.corigami_models as corigami_models
import hic.model.blocks as blocks
from hic.data import genome_dataset
from scipy.stats import pearsonr, spearmanr
import insulation as insu
from tqdm import tqdm
import h5py

# Metrics
def mse(preds, targets):
    mse = ((preds - targets) ** 2).mean(axis = (1, 2))
    results = list(mse.astype(float))
    return results

def insulation_pearson(preds, targets):
    scores = []
    for pred, target in zip(preds, tqdm(targets)):
        pred_insu = np.array(insu.chr_score(pred))
        label_insu = np.array(insu.chr_score(target))
        nas = np.logical_or(np.isnan(pred_insu), np.isnan(label_insu))
        if nas.sum() == len(pred):
            scores.append(np.nan)
        else:
            metric, p_val = pearsonr(pred_insu[~nas], label_insu[~nas])
            scores.append(metric)
    results = scores
    return results

def observed_vs_expected_with_means(preds, targets, preds_mean, targets_mean):
    scores = []
    for pred, target in zip(preds - preds_mean, tqdm(targets - targets_mean)):
        metric, p_val = pearsonr(pred.reshape(-1), target.reshape(-1))
        scores.append(metric)
    results = scores
    return results

def observed_vs_expected(preds, targets):
    scores = []
    preds_mean = preds.mean(axis = 0, keepdims = True)
    targets_mean = targets.mean(axis = 0, keepdims = True)
    for pred, target in zip(preds - preds_mean, tqdm(targets - targets_mean)):
        metric, p_val = pearsonr(pred.reshape(-1), target.reshape(-1))
        scores.append(metric)
    results = scores
    return results

def distance_stratified_correlation(preds, targets):
    scores = []
    for pred, target in zip(preds, tqdm(targets)):
        distance_list = []
        for dis_i in range(len(pred)):
            pred_diag_i = np.diagonal(pred, offset = dis_i)
            target_diag_i = np.diagonal(target, offset = dis_i)
            if len(pred_diag_i) < 2: break
            metric, p_val = pearsonr(pred_diag_i, target_diag_i)
            distance_list.append(metric)
        scores.append(distance_list)
    results = scores
    return results
# Main entrypoint
def main():
    args = init_parser()
    init_training(args)

def init_parser():
  parser = argparse.ArgumentParser(description='C.Origami Training Module.')

  parser.add_argument('--project-dir', dest='project_dir',
                        default=str(_REPO_ROOT),
                        help='Project root directory (should contain `src/`).')

  # Data and Run Directories
  parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

  # Data directories
  parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
  parser.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                        help='Genome assembly for training data')
  parser.add_argument('--celltype', dest='dataset_celltype', default='imr90',
                        help='Sample cell type for prediction, used for output separation')
  parser.add_argument('--sequence-embedding-h5', dest='sequence_embedding_h5', default=None,
                        help='Optional path to an H5 file with precomputed sequence embeddings ordered by the provided BED file.')
  parser.add_argument('--sequence-embedding-key', dest='sequence_embedding_key', default='embeddings_tensor',
                        help='Dataset key inside the H5 file that stores embeddings.')
  parser.add_argument('--sequence-intervals-bed', dest='sequence_intervals_bed', default='corigami_intervals.bed',
                        help='BED file that specifies the order of intervals corresponding to the embeddings.')

  # Optional SUCCEED sequence encoder (used when succeed=True in ConvTransModel and sequence embeddings are not precomputed)
  parser.add_argument('--use-succeed-seq-model', dest='use_succeed_seq_model',
                        action='store_true',
                        help='Use a pretrained SUCCEED sequence encoder inside the Hi-C model (requires --succeed-ckpt).')
  parser.add_argument('--succeed-ckpt', dest='succeed_ckpt',
                        default='',
                        help='Path to pretrained SUCCEED checkpoint (.pth) used as sequence encoder.')
  parser.add_argument('--succeed-output-heads', dest='succeed_output_heads', default=6389, type=int,
                        help='Output head size used when instantiating SUCCEED encoder (human head).')
  parser.add_argument('--succeed-stem-windows', dest='succeed_stem_windows', default=32, type=int,
                        help='SUCCEED config: stem_windows.')
  parser.add_argument('--succeed-pool-windows', dest='succeed_pool_windows', default=4, type=int,
                        help='SUCCEED config: pool_windows.')
  parser.add_argument('--succeed-max-seq-len', dest='succeed_max_seq_len', default=256, type=int,
                        help='SUCCEED config: max_seq_len.')
  parser.add_argument('--succeed-target-seq-len', dest='succeed_target_seq_len', default=256, type=int,
                        help='SUCCEED config: target_seq_len.')

  # Model parameters
  parser.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                        help='CNN with Transformer')

  # Training Parameters
  parser.add_argument('--patience', dest='trainer_patience', default=80,
                        type=int,
                        help='Epoches before early stopping')
  parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=80,
                        type=int,
                        help='Max epochs')
  parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                        type=int,
                        help='Top n models to save')
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=2,
                        type=int,
                        help='Number of GPUs to use')

  # Dataloader Parameters
  parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8,
                        type=int,
                        help='Batch size')
  parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
  parser.add_argument('--num-workers', dest='dataloader_num_workers', default=16,
                        type=int,
                        help='Dataloader workers')


  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  return args

def resolve_seq_channels(args):
  if not args.sequence_embedding_h5:
    return 4
  h5_path = os.path.abspath(os.path.expanduser(args.sequence_embedding_h5))
  if not os.path.exists(h5_path):
    raise FileNotFoundError(f'Sequence embedding file not found: {h5_path}')
  with h5py.File(h5_path, 'r') as h5_file:
    if args.sequence_embedding_key not in h5_file:
      raise KeyError(f'Dataset key {args.sequence_embedding_key} not found in {h5_path}')
    dataset = h5_file[args.sequence_embedding_key]
    if dataset.ndim < 2:
      raise ValueError(f'Embedding dataset must be at least 2D, found shape {dataset.shape}')
    if dataset.ndim == 2:
      return dataset.shape[1]
    return dataset.shape[-1]

def init_training(args):
    if args.use_succeed_seq_model and args.sequence_embedding_h5 is None:
        if not args.succeed_ckpt:
            raise ValueError("--use-succeed-seq-model requires --succeed-ckpt")

        project_dir = args.project_dir
        src_dir = os.path.join(project_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from pretrain import layers as succeed_layers
        from hic.model.config import ModelArgs as SucceedArgs

        config_args = SucceedArgs()
        config_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_args.output_heads = {"human": args.succeed_output_heads}
        config_args.stem_windows = args.succeed_stem_windows
        config_args.pool_windows = args.succeed_pool_windows
        config_args.max_seq_len = args.succeed_max_seq_len
        config_args.target_seq_len = args.succeed_target_seq_len


        model_epi = succeed_layers.SUCCEED(config_args, return_emb=True)
        ckpt = torch.load(args.succeed_ckpt, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_heads.')}
        incompat = model_epi.load_state_dict(filtered_state_dict, strict=False)
        
        print("SUCCEED missing:", incompat.missing_keys)
        print("SUCCEED unexpected:", incompat.unexpected_keys)

        model_epi.to(config_args.device)
        model_epi.eval()
        for p in model_epi.parameters():
            p.requires_grad = False

        blocks.set_succeed_model(model_epi)

    seq_input_channels = resolve_seq_channels(args)

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
    all_loggers = csv_logger
    
    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args, seq_input_channels)
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = all_loggers,
                            callbacks = [early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs
                            )
    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')
    testloader = pl_module.get_dataloader(args, 'test')
    pl_trainer.fit(pl_module, trainloader, valloader)

class TrainModule(pl.LightningModule):
    
    def __init__(self, args, seq_input_channels = 4):
        super().__init__()
        self.seq_input_channels = seq_input_channels
        self.use_precomputed_seq_embeddings = args.sequence_embedding_h5 is not None
        self.model = self.get_model(args, preencoded_seq = self.use_precomputed_seq_embeddings)
        print(self.model)
        self.args = args
        self.save_hyperparameters()
        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2).float()
        seq = seq.float()
        if self.use_precomputed_seq_embeddings:
            inputs = (seq, features)
        else:
            inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss  # PL uses this for backward

    def validation_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)

        batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        self._val_preds.append(outputs.detach().cpu())
        self._val_targets.append(mat.detach().cpu())

    def test_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)

        batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_start(self):
        self._val_preds = []
        self._val_targets = []

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        all_preds = torch.cat(self._val_preds, dim=0).numpy()
        all_targets = torch.cat(self._val_targets, dim=0).numpy()

        mse_scores = mse(all_preds, all_targets)
        ins_pearson = insulation_pearson(all_preds, all_targets)
        ove_scores = observed_vs_expected(all_preds, all_targets)
        dist_corr = distance_stratified_correlation(all_preds, all_targets)

        self.log("val_mse", float(np.nanmean(mse_scores)), prog_bar=True)
        self.log("val_ins_pearson", float(np.nanmean(ins_pearson)), prog_bar=True)
        self.log("val_ove", float(np.nanmean(ove_scores)), prog_bar=True)
        self.log("val_dist_corr", float(np.nanmean([np.nanmean(x) for x in dist_corr])), prog_bar=True)

        self._val_preds = []
        self._val_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        # NOTE: `pl_bolts` is not compatible with PyTorch Lightning v2 in many environments
        # (e.g., `LightningLoggerBase` removal). Use a pure-torch warmup+cosine schedule instead.
        warmup_epochs = 10
        max_epochs = int(self.args.trainer_max_epochs)

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            if max_epochs <= warmup_epochs:
                return 1.0
            progress = float(current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "name": "WarmupCosineAnnealing",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def get_dataset(self, args, mode):

        celltype_root = f'{args.dataset_data_root}/{args.dataset_celltype}'
        genomic_features = {'ctcf_log2fc' : {'file_name' : 'ctcf_log2fc.bw',
                                             'norm' : None },
                            'atac' : {'file_name' : 'atac.bw',
                                             'norm' : 'log' }}
        # genomic_features = {
        #                     'atac' : {'file_name' : 'atac.bw',
        #                                     'norm' : 'log' }}
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                genomic_features, 
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True,
                                sequence_embedding_h5 = args.sequence_embedding_h5,
                                sequence_embedding_key = args.sequence_embedding_key,
                                intervals_bed = args.sequence_intervals_bed)

        # Record length for printing validation image
        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if mode == 'train':
            shuffle = True
        else: # validation and test settings
            shuffle = False
        
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,

            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader

    def get_model(self, args, preencoded_seq = False):
        model_name =  args.model_type
        num_genomic_features = 2
        ModelClass = getattr(corigami_models, model_name)
        use_succeed = bool(args.use_succeed_seq_model) and (not preencoded_seq)
        model = ModelClass(
            num_genomic_features,
            seq_channels=self.seq_input_channels,
            mid_hidden=256,
            succeed=use_succeed,
            preencoded_seq=preencoded_seq,
        )
        return model

if __name__ == '__main__':
    main()
