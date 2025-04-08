import sys
import argparse
import os

# Parse command line arguments
def parse_arguments():
    params = argparse.ArgumentParser(description='C.Origami Training Module.')

    # Project and Run Directories
    params.add_argument('--project_dir', dest='project_dir', required=True,
                        help='Project root directory')
    params.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
    params.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')
    params.add_argument('--use_pth', dest='use_pth', default=None,
                        help='Path to the pretrained model checkpoint')

    # Data directories
    params.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
    params.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                        help='Genome assembly for training data')
    params.add_argument('--celltype', dest='dataset_celltype', default='imr90',
                        help='Sample cell type for prediction, used for output separation')

    # Model parameters
    params.add_argument('--model-type', dest='model_type', default='ConvTransModel',
                        help='CNN with Transformer')

    # Training Parameters
    params.add_argument('--patience', dest='trainer_patience', default=80,
                        type=int,
                        help='Epoches before early stopping')
    params.add_argument('--max-epochs', dest='trainer_max_epochs', default=80,
                        type=int,
                        help='Max epochs')
    params.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                        type=int,
                        help='Top n models to save')
    params.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
                        type=int,
                        help='Number of GPUs to use')

    # Dataloader Parameters
    params.add_argument('--batch-size', dest='dataloader_batch_size', default=8, 
                        type=int,
                        help='Batch size')
    params.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
    params.add_argument('--num-workers', dest='dataloader_num_workers', default=16,
                        type=int,
                        help='Dataloader workers')

    return params.parse_args()

# 在所有其他导入之前解析参数并添加路径
args = parse_arguments()
sys.path.append(args.project_dir+'/src')
sys.path.append(args.project_dir+'/src/3d_genome/sc')

# 然后导入其他模块
import torch
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import scipy.stats
import corigami.model.corigami_models as corigami_models
from corigami.data import genome_dataset
import numpy as np
from scipy.stats import pearsonr, spearmanr
import insulation as insu
from tqdm import tqdm

# 评估指标
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

# main开始
def main():
    args = parse_arguments()
    init_training(args)

def init_training(args):

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
    pl_module = TrainModule(args)
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu", devices=[args.trainer_num_gpu],
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
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        print(self.model)
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat
    
    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size = inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        
        # 计算 MSE loss
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        
        # 转换为numpy数组以计算其他指标
        outputs_np = outputs.detach().cpu().numpy()
        mat_np = mat.detach().cpu().numpy()
        
        return {
            'loss': loss,
            'preds': outputs_np,
            'targets': mat_np
        }

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        # 对于训练步骤，我们只关心损失值
        losses = [out['loss'] if isinstance(out, dict) else out for out in step_outputs]
        avg_loss = torch.stack(losses).mean()
        metrics = {'train_loss': avg_loss}
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        
        # 收集所有预测和目标值
        all_preds = np.concatenate([x['preds'] for x in step_outputs])
        all_targets = np.concatenate([x['targets'] for x in step_outputs])
        
        # 计算各种评估指标
        mse_scores = mse(all_preds, all_targets)
        ins_pearson = insulation_pearson(all_preds, all_targets)
        ove_scores = observed_vs_expected(all_preds, all_targets)
        dist_corr = distance_stratified_correlation(all_preds, all_targets)
        
        # 计算平均值
        metrics = {
            'val_loss': ret_metrics['loss'],
            'val_mse': np.nanmean(mse_scores),
            'val_ins_pearson': np.nanmean(ins_pearson),
            'val_ove': np.nanmean(ove_scores),
            'val_dist_corr': np.nanmean([np.nanmean(x) for x in dist_corr])
        }
        
        self.log_dict(metrics, prog_bar=True)
        
    def _shared_epoch_end(self, step_outputs):
        # 从字典中提取损失值
        losses = [out['loss'] for out in step_outputs]
        avg_loss = torch.stack(losses).mean()
        return {'loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):

        celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{args.dataset_celltype}'
        genomic_features = {
                            'atac' : {'file_name' : 'atac_500_1.bw',
                                             'norm' : None }}
        dataset = genome_dataset.GenomeDataset(celltype_root, 
                                args.dataset_assembly,
                                genomic_features, 
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True)

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

    def get_model(self, args):
        model_name =  args.model_type
        num_genomic_features = 1 ##num features
        ModelClass = getattr(corigami_models, model_name)
        model = ModelClass(num_genomic_features, mid_hidden = 256)
        return model

if __name__ == '__main__':
    main()
