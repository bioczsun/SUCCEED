from typing import Optional

try:
    from torchmetrics import Metric
except ModuleNotFoundError:  # pragma: no cover
    import torch.nn as nn

    class Metric(nn.Module):
        """
        Minimal fallback for `torchmetrics.Metric`.

        This keeps `src/` importable even when `torchmetrics` isn't installed.
        It implements only the subset used by this repository.
        """

        is_differentiable: Optional[bool] = False
        higher_is_better: Optional[bool] = True

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._state_defaults = {}

        def add_state(self, name, default, dist_reduce_fx=None):
            if not hasattr(default, "clone"):
                raise TypeError("Fallback Metric only supports tensor states.")
            self.register_buffer(name, default.clone())
            self._state_defaults[name] = default.clone()

        def reset(self):
            for name, default in self._state_defaults.items():
                setattr(self, name, default.clone())

        def forward(self, *args, **kwargs):
            self.update(*args, **kwargs)
            return self.compute()
import torch
import numpy as np
import os

class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        clean_correlation = correlation[torch.isfinite(correlation)]
        return clean_correlation
    
class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path,model_name,patience=3,verbose=False, delta=0):
        """
        Args:
            save_path : model save dir
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        super(EarlyStopping, self).__init__()
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.model_name = model_name

    def __call__(self, val_loss, model,optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, '%s_best_network.pth'%self.model_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, path)
        # torch.save(model.state_dict(), path)  # This would store the best model weights so far.
        self.val_loss_min = val_loss

def compute_rowwise_pearson(y_true, y_pred):
    """
    Compute Pearson correlation coefficient row-wise.
    
    Args:
        y_true (torch.Tensor): Ground truth values with shape [N, D].
        y_pred (torch.Tensor): Predicted values with shape [N, D].
    
    Returns:
        torch.Tensor: Pearson correlation for each row with shape [N].
    """
    # Cast to float
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Row-wise means
    true_mean = torch.mean(y_true, dim=1, keepdim=True)
    pred_mean = torch.mean(y_pred, dim=1, keepdim=True)
    
    # Row-wise covariance
    covariance = torch.sum((y_true - true_mean) * (y_pred - pred_mean), dim=1)
    
    # Row-wise variances
    true_var = torch.sum((y_true - true_mean) ** 2, dim=1)
    pred_var = torch.sum((y_pred - pred_mean) ** 2, dim=1)
    
    # Avoid division by zero
    denominator = torch.sqrt(true_var) * torch.sqrt(pred_var)
    denominator = torch.where(denominator > 1e-12, denominator, torch.tensor(float('inf')).to(denominator.device))
    
    # Pearson correlation
    pearson_r = covariance / denominator
    
    return pearson_r
