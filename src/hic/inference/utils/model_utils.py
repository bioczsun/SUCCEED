import sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import hic.model.blocks as blocks
import hic.model.corigami_models as corigami_models

def load_default(
    model_path,
    *,
    num_genomic_features=2,
    succeed=False,
    project_dir="",
    succeed_ckpt=""
):
    model_name = 'ConvTransModel'
    mid_hidden = 256
    if succeed:
        if not succeed_ckpt:
            raise ValueError("`succeed=True` requires `succeed_ckpt` to be set.")
        _init_succeed_seq_model(
            project_dir=project_dir,
            succeed_ckpt=succeed_ckpt
        )

    model = get_model(model_name, mid_hidden, num_genomic_features=num_genomic_features, succeed=succeed)
    load_checkpoint(model, model_path)
    return model


def _init_succeed_seq_model(
    *,
    project_dir="",
    succeed_ckpt=""
):
    if project_dir:
        src_dir = os.path.join(project_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

    from pretrain import layers as succeed_layers
    from hic.model.config import ModelArgs as SucceedArgs

    config_args = SucceedArgs()
    config_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_epi = succeed_layers.SUCCEED(config_args, return_emb=True)
    ckpt = torch.load(succeed_ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_heads.")}
    incompat = model_epi.load_state_dict(filtered_state_dict, strict=False)
    print("SUCCEED missing:", incompat.missing_keys)
    print("SUCCEED unexpected:", incompat.unexpected_keys)

    model_epi.to(config_args.device)
    model_epi.eval()
    for p in model_epi.parameters():
        p.requires_grad = False

    blocks.set_succeed_model(model_epi)


def get_model(model_name, mid_hidden, num_genomic_features=2, succeed=False):
    ModelClass = getattr(corigami_models, model_name)
    model = ModelClass(num_genomic_features, mid_hidden=mid_hidden, succeed=succeed)
    return model

def load_checkpoint(model, model_path):
    print('Loading weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    model_weights = checkpoint['state_dict']

    # Edit keys
    for key in list(model_weights):
        # model_weights[key.replace('model.', '')] = model_weights.pop(key)
        model_weights[key[6:]] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    return model

if __name__ == '__main__':
    main()
