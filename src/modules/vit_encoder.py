import math
import torch, torch.nn as nn
from functools import partial
from torch.nn.init import trunc_normal_

from src.utils import apply_mask
from src.modules import PatchEmbed2D, Block
from src.pos_embeds import get_2d_sincos_pos_embed

# NOTE There's a very particular reason why we have the defaults in a dict, but I forgot.
# Will keep it this way for now!
VIT_EMBED_DIMS = {
    'vit_pico': 64,
    'vit_nano': 128,
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}

VIT_PATCH_SIZE = {
    'vit_pico': 2,  # NOTE: This assumes we are using 32x32 images as per CIFAR10
    'vit_nano': 2,
    'vit_tiny': 2,
    'vit_small': 16,
    'vit_base': 16,
    'vit_large': 16,
    'vit_huge': 16,
    'vit_giant': 14,
    'vit_gigantic': 14,
}

VIT_DEPTH = {
    'vit_pico': 12,
    'vit_nano': 12,
    'vit_tiny': 12,
    'vit_small': 12,
    'vit_base': 12,
    'vit_large': 24,
    'vit_huge': 32,
    'vit_gigantic': 48,
    'vit_giant': 40,
}


def vit_pico(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_pico']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_pico']
    depth       = depth         or VIT_DEPTH     ['vit_pico']
  
    return VisionTransformer(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=2, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_nano(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_nano']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_nano']
    depth       = depth         or VIT_DEPTH     ['vit_nano']
      
    return VisionTransformer(
        # NOTE embed_dim needs to be a multiple of num_heads. 
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_tiny(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_tiny']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_tiny']
    depth       = depth         or VIT_DEPTH     ['vit_tiny']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=4, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_small(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_small']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_small']
    depth       = depth         or VIT_DEPTH     ['vit_small']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_base(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_base']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_base']
    depth       = depth         or VIT_DEPTH     ['vit_base']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_large(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_large']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_large']
    depth       = depth         or VIT_DEPTH     ['vit_large']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_huge(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_huge']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_huge']
    depth       = depth         or VIT_DEPTH     ['vit_huge']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_giant(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_giant']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_giant']
    depth       = depth         or VIT_DEPTH     ['vit_giant']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_gigantic(patch_size=None, embed_dim=None, depth=None, **kwargs):
    patch_size  = patch_size    or VIT_PATCH_SIZE['vit_gigantic']
    embed_dim   = embed_dim     or VIT_EMBED_DIMS['vit_gigantic']
    depth       = depth         or VIT_DEPTH     ['vit_gigantic']
    
    return VisionTransformer(
        patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=16, mlp_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=28,
        patch_size=4,
        in_chans=3,
        embed_dim=64,
        depth=12,
        num_heads=2,
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
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
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

    def forward(self, x, mask_ctxt=None):
        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if mask_ctxt is not None:
            x = apply_mask(x, mask_ctxt)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

if __name__ == "__main__":
    vit = vit_pico()
    x = torch.randn(1, 3, 28, 28)
    print(vit(x).shape)
