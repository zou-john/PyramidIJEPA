import torch
from typing import Union, Literal
import torch.nn.functional as F

def mask_idx_to_bool(mask_idxs: torch.Tensor, num_patches: int) -> torch.Tensor:
    """
    Given a batch of indices of visible patches and the total number of patches, it creates a dense boolean mask.
    True = masked / hidden
    False = visible / kept
    """
    B, N_sel = mask_idxs.shape
    dense = torch.ones(B, num_patches, dtype=torch.bool,
                       device=mask_idxs.device)
    batch_idx = torch.arange(B, device=mask_idxs.device)[:, None]
    dense[batch_idx.expand(-1, N_sel), mask_idxs] = False
    return ~dense

def downsample_mask_max(mask: torch.Tensor,
                        H_f: int, W_f: int, 
                        H_c: int, W_c: int) -> torch.Tensor:
    """
    Max-pool boolean mask from a fine resolution (H_f, W_f) to a coarser resolution (H_c, W_c).
    """
    B = mask.size(0)
    # -- Compute the pooling kernel size
    k_h, k_w = H_f // H_c, W_f // W_c
    pooled = F.max_pool2d(mask.view(B, 1, H_f, W_f).float(),
                          kernel_size=(k_h, k_w),
                          stride=(k_h, k_w))

    coarse = pooled.flatten(2).squeeze(1).bool() # [B, H_c x W_c]
    return coarse

def downsample_mask_avg(mask_fine: torch.Tensor,
                        H_f: int, W_f: int, 
                        H_c: int, W_c: int, 
                        threshold: float = 0.5) -> torch.Tensor:
    """
    Avg-pool boolean mask to a coarser resolution. Convolutions with >threshold masked children are masked.
    """
    B = mask_fine.size(0)
    # -- Compute the pooling kernel size    
    k_h, k_w = H_f // H_c, W_f // W_c 
    # -- Avg-pool gives fraction of masked children in each coarse patch
    avg = F.avg_pool2d(mask_fine.view(B, 1, H_f, W_f).float(),
                       kernel_size=(k_h, k_w),
                       stride=(k_h, k_w)) 
    # -- Convolutions with >threshold masked children are masked.
    coarse = (avg >= threshold).flatten(2).squeeze(1).bool()  # [B, H_c x W_c]
    return coarse

def build_mask_pyramid(mask_fine: torch.Tensor,
                       H_f: int, W_f: int, num_scales: int,
                       mask_downsampling: Literal['or', 'avg'] = 'or',
                       ) -> list[torch.Tensor]:
    """
    Generate matching list of masks (coarse -> fine)
    """
    masks = []
    # -- Full downsampling from coarse to fine
    for s in range(num_scales-1, -1, -1): 
        factor = 2 ** s # downsampling factor becuase of log2 num_scales
        H_c, W_c = H_f // factor, W_f // factor
        if mask_downsampling == 'or':
            mask = downsample_mask_max(mask_fine, H_f, W_f, H_c, W_c)
        elif mask_downsampling == 'avg':
            mask = downsample_mask_avg(mask_fine, H_f, W_f, H_c, W_c)
        masks.append(mask)

    # -- Return the list of masks (coarse -> fine) 
    # Scale 1 mask (shape torch.Size([1, 1])):
    #  tensor([[ True]])

    # Scale 2 mask (shape torch.Size([1, 4])):
    #  tensor([[ True, False, False,  True]])

    # Scale 3 mask (shape torch.Size([1, 16])):
    #  tensor([[ True, False, False, False, 
    #            False,  True, False, False, 
    #            False, False, False, False, 
    #            False, False, False,  True]])
    return masks 
