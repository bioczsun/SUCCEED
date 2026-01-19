import os
import numpy as np
import pandas as pd
import sys
import torch

from hic.inference.utils.model_utils import load_default

def preprocess_default(seq, ctcf, atac):
    # Process sequence
    seq = torch.tensor(seq).unsqueeze(0) 
    # Normalize signals (replace NaN with 0)
    atac_sig = torch.tensor(np.nan_to_num(atac, 0))
    features = [atac_sig]
    if ctcf is not None:
        ctcf_sig = torch.tensor(np.nan_to_num(ctcf, 0))
        features = [ctcf_sig, atac_sig]
    features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim = 2)
    inputs = torch.cat([seq, features], dim = 2)
    # Move input to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    return inputs

## Load data ##
def load_region(chr_name, start, seq_path, ctcf_path, atac_path, window = 2097152):
    ''' Single loading method for one region '''
    end = start + window
    seq, ctcf, atac = load_data_default(chr_name, seq_path, ctcf_path, atac_path)
    seq_region, ctcf_region, atac_region = get_data_at_interval(chr_name, start, end, seq, ctcf, atac)
    return seq_region, ctcf_region, atac_region


def load_data_default(chr_name, seq_path, ctcf_path, atac_path):
    from hic.data.data_feature import SequenceFeature, GenomicFeature
    seq_chr_path = os.path.join(seq_path, f'{chr_name}.fa.gz')
    seq = SequenceFeature(path = seq_chr_path)
    ctcf = GenomicFeature(path=ctcf_path, norm=None) if ctcf_path else None
    atac = GenomicFeature(path = atac_path, norm = None)

    return seq, ctcf, atac

def get_data_at_interval(chr_name, start, end, seq, ctcf, atac):
    '''
    Slice data from arrays with transformations
    '''
    seq_region = seq.get(start, end)
    ctcf_region = ctcf.get(chr_name, start, end) if ctcf is not None else None
    atac_region = atac.get(chr_name, start, end)
    return seq_region, ctcf_region, atac_region

## Load Model ##
def prediction(
    seq_region,
    ctcf_region,
    atac_region,
    model_path,
    *,
    use_atac_only=False,
    use_succeed_seq_model=False,
    project_dir="",
    succeed_ckpt=""
):
    if use_atac_only:
        ctcf_region = None
        num_genomic_features = 1
    else:
        num_genomic_features = 2

    model = load_default(
        model_path,
        num_genomic_features=num_genomic_features,
        succeed=use_succeed_seq_model,
        project_dir=project_dir,
        succeed_ckpt=succeed_ckpt
    )
    inputs = preprocess_default(seq_region, ctcf_region, atac_region)
    pred = model(inputs)[0].detach().cpu().numpy()
    return pred
