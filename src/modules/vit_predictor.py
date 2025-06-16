import math
import torch, torch.nn as nn
from functools import partial
from torch.nn.init import trunc_normal_


from src.utils import apply_mask, repeat_interleave_batch
from src.pos_embeds import get_2d_sincos_pos_embed
from src.modules import Block
from src.vit_encoder import VIT_EMBED_DIMS

VIT_PREDICTOR_EMBED_DIMS = {
    'vit_pico': 32,
    'vit_nano': 64,
    'vit_tiny': 128,
    'vit_small': 192,
    'vit_base': 384,
    'vit_large': 512,
    'vit_huge': 512,
    'vit_giant': 768,
    'vit_gigantic': 768,
}

VIT_PREDICTOR_PATCH_SIZE = {
    'vit_pico': 2,
    'vit_nano': 2,
    'vit_tiny': 2,
    'vit_small': 16,
    'vit_base': 16,
    'vit_large': 16,
    'vit_huge': 16,
    'vit_giant': 14,
    'vit_gigantic': 14,
}

VIT_PREDICTOR_DEPTH = {
    'vit_pico': 6,
    'vit_nano': 6,
    'vit_tiny': 6,
    'vit_small': 6,
    'vit_base': 6,
    'vit_large': 12,
    'vit_huge': 16,
    'vit_giant': 24,
    'vit_gigantic': 20,
}

def vit_pico_predictor(num_patches: int, patch_size=None, predictor_embed_dim=None, depth=None, **kwargs):
    patch_size = patch_size or VIT_PREDICTOR_PATCH_SIZE['vit_pico']
    predictor_embed_dim = predictor_embed_dim or VIT_PREDICTOR_EMBED_DIMS['vit_pico']
    depth = depth or VIT_PREDICTOR_DEPTH['vit_pico']
    # TODO: Make this configurable and passed in when instantiating the predictor as above
    encoder_embed_dim = VIT_EMBED_DIMS['vit_pico']

    return VisionTransformerPredictor(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        num_patches=num_patches,
        patch_size=patch_size, embed_dim=encoder_embed_dim, predictor_embed_dim=predictor_embed_dim, depth=depth, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_nano_predictor(num_patches: int, patch_size=None, predictor_embed_dim=None, depth=None, **kwargs):
    patch_size = patch_size or VIT_PREDICTOR_PATCH_SIZE['vit_nano']
    predictor_embed_dim = predictor_embed_dim or VIT_PREDICTOR_EMBED_DIMS['vit_nano']
    depth = depth or VIT_PREDICTOR_DEPTH['vit_nano']
    # TODO: Make this configurable and passed in when instantiating the predictor as above
    encoder_embed_dim = VIT_EMBED_DIMS['vit_nano']

    return VisionTransformerPredictor(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        num_patches=num_patches,
        patch_size=patch_size, embed_dim=encoder_embed_dim, predictor_embed_dim=predictor_embed_dim, depth=depth, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_tiny_predictor(num_patches: int, patch_size=None, predictor_embed_dim=None, depth=None, **kwargs):
    patch_size = patch_size or VIT_PREDICTOR_PATCH_SIZE['vit_tiny']
    predictor_embed_dim = predictor_embed_dim or VIT_PREDICTOR_EMBED_DIMS['vit_tiny']
    depth = depth or VIT_PREDICTOR_DEPTH['vit_tiny']
    # TODO: Make this configurable and passed in when instantiating the predictor as above
    encoder_embed_dim = VIT_EMBED_DIMS['vit_tiny']

    return VisionTransformerPredictor(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        num_patches=num_patches,
        patch_size=patch_size, embed_dim=encoder_embed_dim, predictor_embed_dim=predictor_embed_dim, depth=depth, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_small_predictor(num_patches: int, patch_size=None, predictor_embed_dim=None, depth=None, **kwargs):
    patch_size = patch_size or VIT_PREDICTOR_PATCH_SIZE['vit_small']
    predictor_embed_dim = predictor_embed_dim or VIT_PREDICTOR_EMBED_DIMS['vit_small']
    depth = depth or VIT_PREDICTOR_DEPTH['vit_small']
    # TODO: Make this configurable and passed in when instantiating the predictor as above
    encoder_embed_dim = VIT_EMBED_DIMS['vit_small']

    return VisionTransformerPredictor(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        num_patches=num_patches,
        patch_size=patch_size, embed_dim=encoder_embed_dim, predictor_embed_dim=predictor_embed_dim, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        # --
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
                                                requires_grad=False)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    self.predictor_pos_embed.shape[-1], 
                    int(num_patches**.5), 
                    cls_token=False
                )
            ).float().unsqueeze(0)
        )
        # --
        _drop_path_scheduler = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=_drop_path_scheduler[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_norm = norm_layer(predictor_embed_dim)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, overall_patches, mask_ctxt, mask_tgt):
        assert (mask_tgt is not None) and (mask_ctxt is not None), 'Cannot run predictor without mask indices'
        B, N_ctxt, N_tgt = overall_patches.shape[0], mask_ctxt.shape[1], mask_tgt.shape[1]
        pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        # -- map from encoder-dim to predictor-dim and add pos-embeds
        context_patches = apply_mask(overall_patches, mask_ctxt)
        context_patches = self.predictor_embed(context_patches)
        context_patches += apply_mask(pos_embed, mask_ctxt)

        # -- concat mask tokens to x
        target_patches_pos_embed = apply_mask(pos_embed, mask_tgt)
        # -- generate as many mask tokens as we have target patches
        pred_tokens = self.mask_token.repeat(B, N_tgt, 1)
        pred_tokens += target_patches_pos_embed

        # -- concat the context patches and the target mask-token patches to form the final input to the predictor
        context_patches = torch.cat([context_patches, pred_tokens], dim=1)
        predicted_patches = context_patches
        # -- fwd prop
        for blk in self.predictor_blocks:
            predicted_patches = blk(predicted_patches)

        predicted_patches = self.predictor_norm(predicted_patches)

        # -- return preds for mask tokens
        predicted_target_patches = predicted_patches[:, N_ctxt:]
        predicted_target_patches = self.predictor_proj(predicted_target_patches)

        return predicted_target_patches