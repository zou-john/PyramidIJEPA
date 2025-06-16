import torch
import math
from typing import Optional, Literal

from src.masking import RandomMask, MultiBlock, mask_idx_to_bool, build_mask_pyramid

def collate_supervised(batch: list[dict[str, torch.Tensor]]):
    """
    Collate a batch of data for supervised training.
    """
    # a full batch is a list of dictionaries with the following keys: 'image', 'label'
    images, labels = zip(*batch)
    # [1, C, H, W], [1, C, H, W] ... B -> [B, C, H, W]
    stacked_images = torch.stack(images, dim=0).squeeze(1)
    # [1], [1] ... B -> [B]
    stacked_labels = torch.stack(labels, dim=0)

    return stacked_images, stacked_labels

class MaskCollator:
    def __init__(
        self,
        mask_generator: RandomMask | MultiBlock = RandomMask,
        downsampling_variant: Literal[
            'fine_only', 'coarse_only', 'full_downsampling', 'unmasked', 'full_resampling'
        ] = 'full_downsampling',
        num_scales: int = 0,
        num_patches: int = (256 // 16) ** 2
    ):
        self.mask_generator = mask_generator
        self.downsampling_variant = downsampling_variant
        self.num_patches = num_patches
        self.lat_h = self.lat_w = int(math.sqrt(num_patches))
        self.num_scales = 1 + min(num_scales, math.floor(math.log2(min(self.lat_h, self.lat_w))))

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
        images, labels = zip(*batch)
        B = len(batch)

        # Generate masks
        mask_data = {
            "mask_context_keep": [],
            "mask_target_keep": [],
            "mask_pyramid": [],
        }

        mask_enc, mask_pred = self.mask_generator()
        masks_enc = torch.unique(torch.cat(mask_enc, dim=0)).repeat(B, 1)
        masks_pred = torch.unique(torch.cat(mask_pred, dim=0)).repeat(B, 1)

        mask_data["mask_context_keep"] = masks_enc
        mask_data["mask_target_keep"] = masks_pred

        if self.num_scales == 1:
            return {
                "image": torch.stack(images, dim=0),
                "label": torch.tensor(labels).unsqueeze(-1),
                **mask_data,
            }

        mask_discard_bool = mask_idx_to_bool(masks_pred, self.num_patches)
        mask_pyr = build_mask_pyramid(
            mask_discard_bool, self.lat_h, self.lat_w, self.num_scales, 'avg'
        )[:-1]

        if self.downsampling_variant == 'full_downsampling':
            mask_data["mask_pyramid"] = mask_pyr

        elif self.downsampling_variant == 'fine_only':
            for mask in mask_pyr:
                mask.fill_(False)
            mask_data["mask_pyramid"] = mask_pyr

        elif self.downsampling_variant == 'coarse_only':
            mask_data["mask_context_keep"] = torch.empty([B, 0])
            mask_data["mask_target_keep"] = torch.empty([B, 0])
            mask_data["mask_pyramid"] = mask_pyr

        elif self.downsampling_variant == 'full_resampling':
            resampled_masks = [
                mask_idx_to_bool(torch.cat(self.mask_generator()[1]).repeat(B, 1), self.num_patches)
                for _ in range(self.num_scales)
            ]
            mask_pyr = [
                build_mask_pyramid(mask, self.lat_h, self.lat_w, self.num_scales, 'avg')[i]
                for i, mask in enumerate(resampled_masks)
            ][:-1]
            mask_data["mask_pyramid"] = mask_pyr

        else:
            raise ValueError(f"Invalid downsampling variant: {self.downsampling_variant}")

        return {
            "image": torch.stack(images, dim=0),
            "label": torch.tensor(labels).unsqueeze(-1),
            **mask_data,
        }


