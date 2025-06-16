import hydra
import torch
import wandb
from PIL import Image
import src.utils as ijepa_utils
from src.utils import apply_mask
from typing import Mapping, Optional
from stable_ssl.base import BaseTrainer
from src.image_datasets import denormalize_image
import src.pixel_reconstructor as pixel_reconstructor, src.vit_encoder as vit_encoder

class PixelReconstructionTrainer(BaseTrainer):

    required_modules = {
        "backbone": vit_encoder.VisionTransformer,
        "reconstructor": pixel_reconstructor.PixelReconstructor,
    }

    def __init__(self,
        data, module, hardware, optim: Mapping, logger,
        loss=torch.nn.MSELoss(),
        extra: Mapping = dict(),
        use_masking: bool = True,
        mask_generator: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        super().__init__(data, module, hardware, optim, logger, loss, **kwargs)
        self.extra = extra
        self.use_masking = use_masking
        if self.use_masking:
            assert mask_generator is not None

        self.mask_generator = mask_generator
        self.dataset = extra.get("dataset", 'cifar10')
        self.current_loss = float('inf')
        self.best_loss = float('inf')

        _tmp_data = hydra.utils.instantiate(self._data, _convert_="object")
        self.epochs: int = self._optim["epochs"]
        self.ipe = sum(len(loader) for k, loader in _tmp_data.items() if isinstance(loader, torch.utils.data.DataLoader) and k == 'train')        
        self._optim["scheduler"]["total_steps"] = self.ipe * self.epochs

    def before_fit(self):
        self.ipe: int = self.optim["max_steps"]
        self.epochs: int = self.optim["epochs"]
        self.wd: float = self.extra.get("wd", (0.01, 0.001))
        self.ipe_scale: float = self.extra.get("ipe_scale", 1.25)
        self.wd_scheduler = ijepa_utils.create_wd_scheduler(
            self.optim["optimizer"],
            self.ipe, self.epochs, self.ipe_scale,
            self.wd[0], self.wd[1])

        for p in self.module['backbone'].parameters():
            p.requires_grad = False

        self.patch_size: int = self.module['backbone'].patch_embed.patch_size
        return super().before_fit()

    def format_label_image_mask(self):
        return self.batch['label'], self.batch['image'], self.batch['mask_context_keep'], self.batch['mask_target_keep']

    def compute_loss(self):
        label, image, mask_ctxt, mask_tgt = self.format_label_image_mask()
        encoding = self.module['backbone'](image)
        reconstruction = self.module['reconstructor'](encoding,
                                                      mask_ctxt if self.use_masking else None,
                                                      mask_tgt  if self.use_masking else None) # shape (B, 3, H, W)

        # if masked, only compute loss on masked regions. note that the reconstructor gives the full image
        if self.use_masking:
            reconstruction  = apply_mask(self.module['reconstructor']._repatchify(reconstruction), mask_tgt)
            image           = apply_mask(self.module['reconstructor']._repatchify(image), mask_tgt)

        loss = self.loss(reconstruction, image)
        return {'reconstruction_loss': loss}

    def after_fit_epoch(self):
        if self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            self._save_checkpoint(f"best_loss.ckpt", model_only=False)

        return super().after_fit_epoch()
    
    def after_fit_step(self):
        self.wd_scheduler.step()
        return super().after_fit_step()

    def before_eval(self):
        super().before_eval()
        self.eval_log_per_label = set()

    def after_eval(self):
        super().after_eval()
        self._log(commit=True)
        self.eval_log_per_label.clear()

    def _to_wandb_image(self, image: torch.Tensor, label: Optional[str] = None) -> wandb.Image:
        torch_img = (denormalize_image(image, self.dataset) * 255.0) \
            .clamp(0,255).to(torch.uint8).permute(1,2,0) # chw->hwc
        pil_img = Image.fromarray(torch_img.detach().cpu().numpy())
        return wandb.Image(pil_img, caption=f"Label: {label}" if label is not None else None)

    def before_eval_step(self):
        # NOTE we override this to mask the image if we're using masking because
        # the `_eval_step` automatically takes self.batch[1]
        # and compares it with `predict()`'s output. So, if we are masking in predict,
        # then we need to mask the image here as well.
        super().before_eval_step()
        label, image, mask_ctxt, mask_tgt = self.batch['label'], self.batch['image'], self.batch['mask_context_keep'], self.batch['mask_target_keep']
        if self.use_masking:
            masked_image = apply_mask(self.module['reconstructor']._repatchify(image), mask_tgt)
        else: masked_image = image

        self.batch = image, masked_image, mask_ctxt, mask_tgt, label

    def _log(self, packet=None, commit=True):
        # bugfixing logic in stable_ssl for the non-wandb case (serializes json with tensors)
        if packet is not None:
            for k, v in packet.items():
                if isinstance(v, torch.Tensor) and v.shape == torch.Size([]):
                    packet[k] = v.item()
        return super()._log(packet, commit)

    def predict(self):
        B, *_ = self.batch[0].shape
        image, masked_image, mask_ctxt, mask_tgt, label = self.batch
        encoding       = self.module['backbone'](image)
        reconstruction = self.module['reconstructor'](encoding,
                                                      mask_ctxt if self.use_masking else None,
                                                      mask_tgt  if self.use_masking else None)
        
        for _label, _image, _reconstruction in zip(label, image, reconstruction):
            if _label in self.eval_log_per_label: continue # this ensures only one of each label is logged!
            if not self._logger['wandb']: continue
            self.eval_log_per_label.add(_label.item())
            self._log({
                f"images/reconstructed_image": self._to_wandb_image(_reconstruction, _label),
                f"images/original_image": self._to_wandb_image(_image, _label),
            }, commit=False)

        if self.use_masking: # apply mask to reconstruction for evaluation if necessary
            reconstruction = self.module['reconstructor']._repatchify(reconstruction)
            reconstruction = apply_mask(reconstruction, mask_tgt)

        return reconstruction
