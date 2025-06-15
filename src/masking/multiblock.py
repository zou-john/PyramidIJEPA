# NOTE Adapted from IJEPA repository
import math
import torch
from multiprocessing import Value


class MultiBlock(object):
    def __init__(
        self,
        input_size: int,
        patch_size=4,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MultiBlock, self).__init__()
        # (H , W) square if it is only a int
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2 
        # size of each patch
        self.patch_size = patch_size 
        # determines what percentage of image is unveiled (visible) to the encoder
        self.enc_mask_scale = enc_mask_scale
        # determines what percentage of image should be predicted based on what the encoder provides
        self.pred_mask_scale = pred_mask_scale
        # determines the shape of unveiled regions (e.g., thin, wide)
        self.aspect_ratio = aspect_ratio    
        # number of encoder masks to generate 
        self.nenc = nenc
        # number of predictor mask to generate
        self.npred = npred
        # minimum number of patches that must be kept unveiled in any valid mask
        self.min_keep = min_keep  
        # whether the encoder and predictor unveiled regions can overlap
        self.allow_overlap = allow_overlap 
        # collator is shared across worker processes
        self._itr_counter = Value('i', -1)  
        # define new size of image given patchsize
        self.height, self.width = input_size
        self.height //= self.patch_size
        self.width //= self.patch_size

    def step(self):
        """
        This function is used to generate a deterministic seed for the mask generator.
        This also ensures a process-safe global counter for the mask generator.
        """
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        """
        This function is used to sample a random block size for the encoder and predictor masks.
        """
        _rand = torch.rand(1, generator=generator).item()
        # -- Samples a random scale between min and max scale, e.g., 0.2 to 0.8
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        # -- Compute the maximum number of patches that can be kept unveiled
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        # -- Ensure the block size is within the image dimensions
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """
        This function is used to sample a block mask for the encoder and predictor masks.
        """
        h, w = b_size # should be returned from _sample_block_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator: "Valid mask not found, {len(mask)} <= {self.min_keep} decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.),
        )

        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        masks_p, masks_C = [], []
        for _ in range(self.npred):
            mask, mask_C = self._sample_block_mask(p_size)
            masks_p.append(mask)
            masks_C.append(mask_C)
            min_keep_pred = min(min_keep_pred, len(mask))

        acceptable_regions = masks_C
        try:
            if self.allow_overlap:
                acceptable_regions = None
        except Exception as e:
            print(f'Encountered exception in mask-generator {e}')

        masks_e = []
        for _ in range(self.nenc):
            mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
            masks_e.append(mask)
            min_keep_enc = min(min_keep_enc, len(mask))
            
        # -- Return the masks for the encoder and predictor
        # --- `masks_e` and `masks_p` are lists of 1D tensors, each containing indices of visible patches for the encoder and predictor, respectively.
        # --- The number of tensors in each list is determined by `nenc` and `npred`.
        # --- The indices correspond to flattened patch positions in the image grid.
        # --- These masks are used to control which parts of the image are visible to the encoder and which parts the predictor must predict, enabling the self-supervised learning objective of IJEPA.
        return masks_e, masks_p

