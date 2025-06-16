import torch
import torch.nn as nn
import einops
from typing import Optional

from src.modules.vit_encoder import ViTEncoder
from src.modules.pixel import Pixel

class Reconstructor(nn.Module):
    def __init__(self, 
                 *,
                 image_size: int, # size of the image
                 patch_size: int, # size of the patch
                 input_dim: int, # dimension of the input
                 depth: int = 12, # depth of the transformer
                 model_dim: Optional[int] = None, # dimension of the model
                 n_heads: int = 2): # number of attention heads in the transformer
        super().__init__()
        # image size and patch size
        self.patch_size = patch_size
        self.H, self.W = image_size // patch_size, image_size // patch_size
        self.num_patches = self.H * self.W
      
        # the intermediate representation space whether this is a compressed or expanded representation of your features
        self.hidden_dim = model_dim or input_dim
        
        # input is [B, N, D] where D is the input dimension from the encoder and is projecting to the hidden dimension
        self.proj = nn.Linear(input_dim, self.hidden_dim, bias=False)

        # learnable mask token with shape [1, 1, hidden_dim] which gets expanded to [B, N, hidden_dim]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        # this adds the patches that were masked out to the input

        # positional encoding for each patch
        position = get_2d_sincos_pos_embed(self.num_patches, self.H, cls_token=False)
        self.pos_embed = nn.Parameter(torch.from_numpy(position).float().unsqueeze(0), requires_grad=False)

        # transfomrer encoder layers
        # processes all the patches (visible and masked tokens) together
        # each patch attends to all the other patches
        # helps understand relationships between visible patches, masked tokens, and their spatial relationships
        layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                           nhead=n_heads, 
                                           dim_feedforward=self.hidden_dim * 4,
                                           batch_first=True, 
                                           norm_first=True)
        # transformer encoder layers * depth
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        # layer norm to normalize the output of the transformer encoder
        self.norm = nn.LayerNorm(self.hidden_dim)

        # linear projection to the pixel space
        self.to_patch = nn.Linear(self.hidden_dim, 3 * patch_size * patch_size, dtype=torch.float32)

        # set on runtime based on the encoder's device
        self.device = None
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # In PixelReconstructor:
        # 1. Takes the context encoder (visble) with predicted patches (masked)
        # 2. All of these patches are now learnable to reverse engineer to the original image


    def _unpatchify(self, patches: torch.Tensor, B: int) -> torch.Tensor:
        """Turns patches back into an image (assuming RGB info got folded into channel dim)"""
        P = self.patch_size
        H = W = self.H_lat
        img = patches.view(B,H,W,3,P,P)
        img = einops.rearrange(img, 'b h w c p1 p2 -> b c (h p1) (w p2)', p1=P, p2=P)
        return img
    
    def _repatchify(self, img: torch.Tensor) -> torch.Tensor:
        """Turns an image back into patches, folding RGB into channel dim"""
        (B, C, H_img, W_img), P = img.shape, self.patch_size
        assert H_img % P == 0 and W_img % P == 0, "image dims not divisible by patch"
        H, W = H_img // P, W_img // P
        patches = einops.rearrange(img, 'b c (h p1) (w p2) -> b h w c p1 p2', p1=P, p2=P)
        patches = patches.reshape(B,H*W,C*P*P)
        return patches

    def forward(self,
                latent_patches: torch.Tensor,
                mask_ctxt: Optional[torch.Tensor] = None,
                mask_tgt: Optional[torch.Tensor] = None) -> torch.Tensor:

        if mask_ctxt is not None or mask_tgt is not None: # both or neither must be provided
            assert mask_ctxt is not None and mask_tgt is not None

        B, N, _ = latent_patches.shape

        if mask_ctxt is None:
            x_vis   = latent_patches
            pos_vis = self.pos_embed_dec
        else:
            x_vis   = apply_mask(latent_patches, mask_ctxt)
            pos_vis = apply_mask(self.pos_embed_dec.repeat(B,1,1), mask_ctxt)

        x = self.in_proj(x_vis) + pos_vis

        if mask_tgt is not None: # create tokens we wanna infill and add pos_embeds, then concat with input
            mask_tok = self.mask_token.repeat(B, mask_tgt.size(1), 1)
            mask_tok += apply_mask(self.pos_embed_dec.repeat(B,1,1), mask_tgt)
            x = torch.cat([x, mask_tok], dim=1)    # order: visible | masked
            x = self.norm(self.core(x))
            patches = self.to_patch(x[:, -mask_tgt.size(1):]) # (B, N_mask, 3*P*P)
            dst = latent_patches.new_zeros(B, N, patches.size(-1)).to(patches.dtype)
            patches = overlay_patches(dst, patches, mask_tgt)
        else:
            x = self.norm(self.core(x))
            patches = self.to_patch(x)

        return self._unpatchify(patches, B)