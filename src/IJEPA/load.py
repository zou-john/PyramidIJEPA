# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torch.nn.init import trunc_normal_
import src.ijepa_paper.vision_transformer as vit


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        for k in ['encoder', 'predictor', 'target_encoder']:
            for _k in list(checkpoint[k].keys()):
                checkpoint[k][_k.replace('module.', '')] = checkpoint[k][_k]
                del checkpoint[k][_k]

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f'loaded optimizers from epoch {epoch}')
        print(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    print(encoder)
    return encoder, predictor

def load_pretrained_model(
    cfg_path='src/IJEPA/in1k_vith14_ep300.yaml',
    model_path='/users/IN1K-vit.h.14-300e.pth.tar',
    device=None,
    freeze_encoder=True,
    freeze_target=True,
    verbose=True
):
    """Load and initialize a pretrained JEPA model.
    
    Args:
        cfg_path (str): Path to model config YAML
        model_path (str): Path to pretrained weights
        device (torch.device): Device to load model to. Defaults to CUDA if available
        freeze_encoder (bool): Whether to freeze encoder weights
        freeze_target (bool): Whether to freeze target encoder weights  
        verbose (bool): Whether to print model architectures
    
    Returns:
        tuple: (encoder, predictor, target_encoder) models
    """
    import yaml
    import copy
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Initialize models
    encoder, predictor = init_model(
        device,
        patch_size=cfg['mask']['patch_size'],
        model_name=cfg['meta']['model_name'], 
        crop_size=cfg['data']['crop_size'],
        pred_depth=cfg['meta']['pred_depth'],
        pred_emb_dim=cfg['meta']['pred_emb_dim']
    )
    
    if verbose:
        print(encoder)
        print(predictor)
        
    # Load checkpoint
    encoder, predictor, target_encoder, _, _, _ = load_checkpoint(
        device,
        model_path,
        encoder,
    predictor,
        target_encoder=copy.deepcopy(encoder),
        opt=None,
        scaler=None,
    )
    
    # Set gradients
    for param in encoder.parameters():
        param.requires_grad = not freeze_encoder
    for param in predictor.parameters():
        param.requires_grad = True  
    for param in target_encoder.parameters():
        param.requires_grad = not freeze_target

    if verbose:
        print(encoder)
        print(predictor) 
        print(target_encoder)
        # Print the number of parameters in the encoder, predictor, and target_encoder
        print(f'Encoder parameters: {sum(p.numel() for p in encoder.parameters())}')
        print(f'Predictor parameters: {sum(p.numel() for p in predictor.parameters())}')
        print(f'Target encoder parameters: {sum(p.numel() for p in target_encoder.parameters())}')
        
    return encoder, predictor, target_encoder


if __name__ == "__main__":
    # NOTE To download the pretrained weights, run the following command:
    # wget https://dl.fbaipublicfiles.com/jepa/IN1K-vit.h.14-300e.pth.tar
    # ideally in the scratch directory. This is 630M params.
    encoder, predictor, target_encoder = load_pretrained_model()
