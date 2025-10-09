import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

import pandas as pd
import numpy as np

import sys
sys.path.append("/home/suncz/work/s02/SUCCEED/src")

from utils.GenomeDataset import GenomeDataset
from model.layers import SUCCEED
from model.config import ModelArgs
from model.blocks0 import TargetLengthCrop
from training.metric import MeanPearsonCorrCoefPerChannel

run_save_path = '/home/suncz/work/s02/Encode_epigenome/results/preprocessed/131k/model'
trainer_save_top_n = 20
run_seed = 1401

# Early_stopping
early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                    min_delta=0.00, 
                                    patience=50,
                                    verbose=False,
                                    mode="min")
# Checkpoints
checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{run_save_path}/models',
                                    save_top_k=trainer_save_top_n, 
                                    monitor='val_loss')

# LR monitor
lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

# Logger
csv_logger = pl.loggers.CSVLogger(save_dir = f'{run_save_path}/csv')
all_loggers = csv_logger

# Assign seed
pl.seed_everything(run_seed, workers=True)
contig_bed = pd.read_csv("/home/suncz/work/s02/Encode_epigenome/results/preprocessed/131k-Basenji/human/sequences.bed", sep="\t", header=None)
contig_bed.columns = ["chrom", "start", "end","label"]
seq_dir = "/home/suncz/work/s02/Encode_epigenome/results/downstream/C.Origami/corigami_data/data/hg38/sc_imr90_gm12878/dna_sequence"

## train dataloader
train_bed = contig_bed[contig_bed["label"] == "train"].values
train_features = "/home/suncz/work/s02/Encode_epigenome/results/preprocessed/131k-Basenji/human/train_sequence_target.h5"
train_dataset = GenomeDataset(train_bed,seq_dir,train_features,"hg38","train")
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=10,shuffle=False)


## valid dataloader
valid_bed = contig_bed[contig_bed["label"] == "valid"].values
valid_features = "/home/suncz/work/s02/Encode_epigenome/results/preprocessed/131k-Basenji/human/valid_sequence_target.h5"
valid_dataset = GenomeDataset(valid_bed,seq_dir,valid_features,"hg38","valid")
valid_dataloader = DataLoader(valid_dataset, batch_size=16, num_workers=10,shuffle=False)

import math

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    线性预热 + 余弦退火学习率调度器
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, base_lr=1e-3, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热阶段
            return [self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
                    for _ in self.base_lrs]
        else:
            # 余弦退火阶段
            cos_anneal = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            return [self.min_lr + (self.base_lr - self.min_lr) * cos_anneal for _ in self.base_lrs]
        

pl_trainer = pl.Trainer(accumulate_grad_batches=1,
                        accelerator="gpu", devices=1,
                        logger = all_loggers,
                        callbacks = [early_stop_callback,
                                        checkpoint_callback,
                                        lr_monitor],
                        max_epochs = 500,
                        )

target_crop = TargetLengthCrop(896)

from model.config import ModelArgs
    
# define train and val data    

## 定义 lighting 模型
class TrainModule(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model =  SUCCEED(ModelArgs(
    ))#EpiModel(ModelArgs(device="cuda"))
        #
        self.save_hyperparameters()

        self.loss_fn = nn.PoissonNLLLoss(log_input=False)

        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.corr_coef_train = MeanPearsonCorrCoefPerChannel(n_channels=6389)
        self.corr_coef_test = MeanPearsonCorrCoefPerChannel(n_channels=6389)



    def forward(self, x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x,y = batch
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        y = target_crop(y)
        outs = self(x.float())["human"]
        loss = self.loss_fn(outs, y)
        # 计算 Pearson 相关系数
        
        
        self.corr_coef_train(outs.detach(), y)
        r = self.corr_coef_train.compute().mean()

        metrics = {'train_step_loss': loss, 'r': r}
        self.log_dict(metrics, batch_size = x.shape[0], prog_bar=True)
        self.training_step_outputs.append({"loss": loss, "r": r})
        return {"loss": loss, "r": r}
    
    # def on_train_epoch_start(self):
    #     """在每个 epoch 开始时重置 Pearson 相关系数计算对象"""
    #     self.corr_coef_train.reset()
    #     self.corr_coef_test.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        y = target_crop(y)
        outs = self(x.float())["human"]
        loss = self.loss_fn(outs, y)

        # 计算 Pearson 相关系数
        self.corr_coef_test(outs.detach(), y)
        r = self.corr_coef_test.compute().mean()
        self.validation_step_outputs.append({"loss": loss, "r": r})

        return {"loss": loss, "r": r}

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     outputs = self(x)["human"]
    #     loss = self.loss_fn(outputs, y)

    #     # 计算 Pearson 相关系数
    #     self.corr_coef_test(x, y)
    #     r = self.corr_coef_test.compute().mean()

    #     self.test_step_outputs.append({"loss": loss, "r": r})

    #     return {"loss": loss, "r": r}

    # Collect epoch statistics
    def on_train_epoch_end(self):
        """在 epoch 结束时计算平均 loss """
        losses = [out["loss"] for out in self.training_step_outputs]
        avg_loss = torch.stack(losses).mean()

        r_ls = [out["r"] for out in self.training_step_outputs]
        avg_r = torch.stack(r_ls).mean()

        metrics = {"train_loss": avg_loss, "train_r": avg_r}
        self.log_dict(metrics, prog_bar=True)

        # 清空存储的 batch 结果，防止累积到下个 epoch
        self.training_step_outputs.clear()



    def on_validation_epoch_end(self):
        """在 epoch 结束时计算平均 loss """
        losses = [out["loss"] for out in self.validation_step_outputs]
        r_ls = [out["r"] for out in self.validation_step_outputs]

        #让 Lightning 自动收集所有 GPU 结果
        losses = self.all_gather(torch.stack(losses)).mean()
        avg_r = self.all_gather(torch.stack(r_ls)).mean()

        metrics = {"val_loss": losses, "val_pearson": avg_r}
        self.log_dict(metrics, prog_bar=True)

        self.validation_step_outputs.clear()

        
    # def _shared_epoch_end(self, step_outputs):
    #     losses = [out['loss'] if isinstance(out, dict) else out for out in step_outputs]
    #     avg_loss = torch.stack(losses).mean()
    #     R = [out['r'] if isinstance(out, dict) else out for out in step_outputs]
    #     avg_r = torch.stack(R).mean()
    #     return avg_loss, avg_r

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr = 1e-3)

        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=80)
        # scheduler_config = {
        #     'scheduler': scheduler,
        #     'interval': 'epoch',
        #     'frequency': 1,
        #     'monitor': 'val_loss',
        #     'strict': True,
        #     'name': 'WarmupCosineAnnealing',
        # }
        return{'optimizer' : optimizer}#, 'lr_scheduler' : scheduler_config}
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """手动裁剪梯度"""
        optimizer_closure()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.2)
        optimizer.step()


pl_module = TrainModule()
pl_trainer.fit(
    model=pl_module, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader
)
