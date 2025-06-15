# NOTE Adapted from StableSSL
import math
import torch
from multiprocessing import Value

class RandomMask:
    """Apply tube masking to spatiotemporal video data by masking aligned spatial patches across time.

    This class implements tube masking as used in V-JEPA and similar architectures. It can handle:
    1. Raw video tensors [T, C, H, W]
    2. Pre-patchified tensors where H,W represent a grid of patches

    For example, given:
    - Raw video: [16, 3, 224, 224]
    - Patchified video: [16, 768, 14, 14] (using 16x16 patches)
    The masking pattern is consistent across the temporal dimension, creating "tubes".

    Parameters
    ----------
    ratio : float
        Ratio of patches to mask out (between 0 and 1)
    patch_size : Union[tuple[int, int], int]
        Size of patches for masking. For pre-patchified input, use (1,1)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors containing:
        1. Kept patches with shape [T, N_kept, C]
        2. Masked patches with shape [T, N_masked, C]
        where N_kept + N_masked = H*W/patch_size^2
    """

    def __init__(
        self,
        ratio: float,
        patch_size: Union[tuple[int, int], int],
        input_size: int,
    ):
        super(RandomMask, self).__init__()
        self.ratio = ratio
        if isinstance(patch_size, int):
            self.patch_size = (patch_size,) * 2
        else:
            self.patch_size = patch_size
        self.height = input_size
        self.width = input_size

    def sample_spatial_mask(
        self, ratio: float, num_spatial_patches: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spatial masking pattern to be applied across temporal dimension.

        Parameters
        ----------
        ratio : float
            Ratio of patches to mask (between 0 and 1)
        num_spatial_patches : int
            Total number of spatial patches (H*W/patch_size^2)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Indices of patches to keep and discard
        """
        num_keep_spatial = int(num_spatial_patches * (1.0 - ratio))
        mask = torch.cat(
            [
                torch.zeros(num_spatial_patches - num_keep_spatial),
                torch.ones(num_keep_spatial),
            ]
        )
        # NOTE Equivalent to np.random.shuffle(mask)
        mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
        mask_discard = torch.argwhere(mask == 0).squeeze()
        mask_keep = torch.nonzero(mask).squeeze()
        return mask_keep, mask_discard

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply tube masking to input video.

        Parameters
        ----------
        video_tchw : torch.Tensor
            Input video tensor in [T, C, H, W] format
            Can be either raw video or pre-patchified
            If input tensor is an image-like [C, H, W],
            it will be casted to a video-like by the addition
            of an extra temporal dimension into [1, C, H, W]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Kept patches and masked patches
            Both have shape [T, N, C] where N varies based on ratio
        """
        num_patches_spatial: int = (self.height // self.patch_size[0]) * (self.width // self.patch_size[1])
        mask_keep, mask_discard = self.sample_spatial_mask(
            self.ratio, num_patches_spatial
        )

        return [mask_keep], [mask_discard]