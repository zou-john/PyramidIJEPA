import hydra
import torch
import torch.nn.functional as F
from typing import Literal, Optional
import einops

from stable_ssl.trainer import JointEmbeddingPredictiveTrainer

# NOTE: USED IN PRETRAINING
class IJEPATrainer(JointEmbeddingPredictiveTrainer):

    required_modules = {
        "context_encoder": torch.nn.Module,
        "target_encoder": torch.nn.Module,
        "predictor": torch.nn.Module,
    }

    # Parameters are directly defined from BaseTrainer
    def __init__(self,
                 data
                 module,
                 hardware,
                 optimizer: Mapping,
                 logger,
                 loss = PatchWiseL2Loss,
                 extra: Mapping = dict(),
                 mask_generator: Callable = RandomMask,
                 **kwargs
    ):
        super().__init__(data, module, hardware, optimizer, logger, loss,**kwargs)
        self.extra = extra
        self.mask_generator = mask_generator
    
        self.stop_after_epoch : int | None = kwargs.get('stop_after_epoch', None)
        self.stop_after_step = int(self.stop_after_epoch) if self.stop_after_epoch else None
         
        _tmp_data = hydra.utils.instanciate(self._data, _convert_="object")
        self.epochs: int = self._optim["epochs"] 
        self.ipe = sum(len(loader) for k, loader in _tmp_data.items() if isinstance(loader, torch.utils.data.DataLoader) and k == 'train')
        self._optim["scheduler"]["total_steps"] = self.epochs * self.ipe


    def before_fit(self):
        """
        Before training for the IJEPATrainer.
        """
        # sets up gradient hooks for monitoring to prevent vanishing or exploding gradients
        self._set_gradnorm_hooks("context_encoder")
        self._set_gradnorm_hooks("predictor")

        # sets up the exponential moving average scheduler
        self.ipe_scale: float = self.extra.get("ipe_scale", 1.25)
        # favors 90% of the old values and 10% of the new values
        # this shifts to 99.99% towards the end, this is to prevent model collapse
        self.ema : tuple[float, float] = self.extra.get("ema", (0.9, 0.999))
        self.momentum_scheduler = ijepa_utils.create_ema_scheduler(self.ipe, self.epochs, self.ipe_scale, self.ema[0], self.ema[1])
        
        # sets up the weight decay scheduler
        self.weight_decay : tuple[float, float] = self.extra.get("weight_decay", (0.1, 0.001))
        # regularization that penalizes large weights in the beggining and then gradually decreases
        self.weight_decay_scheduler = ijepa_utils.create_weight_decay_scheduler(self.optimizer["optimizer"], self.ipe, self.epochs, self.ipe_scale, self.weight_decay[0], self.weight_decay[1])

        # sets the target encoder to not require gradients
        for p in self.module["target_encoder"].parameters(): p.requires_grad = False
        # prevents the target encoder from being updated during training because this target encoder is the ground truth
        # calls the before_fit method from the BaseTrainer
        return super().before_fit()
    
    def format_context_target(self):
        return self.batch["image"], self.batch["mask_context_keep"], self.batch["mask_target_keep"]
    
    def forward_target(self, image: torch.Tensor, target_patches: torch.Tensor | None):
        """
        Forward pass for the target encoder.
        """
        # no gradients are computed for the target encoder
        with torch.no_grad():
            # encode the whole image without masking [B, N, D]
            overall_patches = self.module["target_encoder"](image)
            # layer norm normalizes the features of the patches to have a mean of 0 and a standard deviation of 1
            overall_patches = F.layer_norm(overall_patches, (overall_patches.size(-1),)) 
            # expose only the target patches which are the patches that the predictor will predict
            target_patches = ijepa_utils.apply_mask(overall_patches, target_patches) if target_patches is not None else overall_patches
        # B x N x D
        return overall_patches, target_patches
    
    def forward_context(self, image: torch.Tensor, mask_context_keep: list[torch.Tensor]):
        return self.module["context_encoder"](image, mask_context_keep)

    def forward_predictor(self, *args, **kwargs):
        return self.module["predictor"](*args, **kwargs)

    def _set_gradnorm_hooks(self, module_name: str):
        def create_grad_hook(param_name: str):
            def hook_fn(grad: torch.Tensor):
                if (grad is not None) and not (param_name.endswith('.bias') or len(grad.shape) == 1):
                    self._log_buffer.update({f"layer_stats/{module_name}/{param_name}/grad_norm": grad.norm().item()})
                return grad

            return hook_fn

        for name, param in self.module[module_name].named_parameters():
            if param.requires_grad: param.register_hook(create_grad_hook(name))
    
    def _log(self, packet=None, commit=True):
        # bugfixing logic in stable_ssl for the non-wandb case (serializes json with tensors)
        if packet is not None:
            for k, v in packet.items():
                if isinstance(v, torch.Tensor) and v.shape == torch.Size([]):
                    packet[k] = v.item()

        # aggregate grad norms then hand back to super
        grad_norms = {k: v for k, v in self._log_buffer.items() if k.startswith("layer_stats/") and k.endswith("/grad_norm")}
        if len(grad_norms) > 0:
            if len(context_grad_norms := [v for k, v in grad_norms.items() if '/context_encoder/' in k]) > 0:
                self._log_buffer["stats/context_encoder/grad_norm"] = sum(context_grad_norms) / len(context_grad_norms)
            if len(predictor_grad_norms := [v for k, v in grad_norms.items() if '/predictor/' in k]) > 0:
                self._log_buffer["stats/predictor/grad_norm"] = sum(predictor_grad_norms) / len(predictor_grad_norms)

        super()._log(packet=packet, commit=commit)

    def compute_loss(self):
        """
        Compute the loss for the IJEPATrainer.
        """
        image, mask_context_keep, mask_target_keep = self.format_context_target()

        # extract the context patches
        context_representation = self.forward_context(image, mask_context_keep)

        # extract the target patches
        whole_image_representation, target_representation = self.forward_target(image, mask_target_keep)

        self._latest_embedding = self._latest_representation = whole_image_representation

        # compute the loss: patch wise l2 loss
        loss = self.loss(context_representation, target_representation)

        # Update the _log_buffer with stats about the model
        if self.global_step % self.logger["log_every_step"] == 0:
            self._log({
                ###### context stats
                "stats/context/patch_norm": patch_norm(context_representation).detach().item(),
                "stats/context/repr_mean": context_representation.mean().detach().item(),
                "stats/context/repr_std": context_representation.std().detach().item(),
                ###### target stats
                "stats/target/patch_norm": patch_norm(target_representation).detach().item(),
                "stats/target/repr_mean": target_representation.mean().detach().item(),
                "stats/target/repr_std": target_representation.std().detach().item(),
                ###### predicted stats
                "stats/pred/patch_norm": patch_norm(context_representation).detach().item(),
                "stats/pred/repr_mean": context_representation.mean().detach().item(),
                "stats/pred/repr_std": context_representation.std().detach().item(),
            }, commit=False)

        return {"loss_ssl": loss}

    def after_fit_epoch(self):
        if self.stop_after_epoch and self.epoch >= self.stop_after_epoch:
            raise BreakAllEpochs()
        
        return super().after_fit_epoch()
    
    def after_fit_step(self):
        momentum = next(self.momentum_scheduler)
        self.weight_decay_scheduler.step()

        self._log({'train/ema' : momentum}, commit=False)
        with torch.no_grad():
            for param_q, param_k in zip(self.module["context_encoder"].parameters(), self.module["predictor"].parameters()):
                param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)
        # calls the after_fit_step method from the BaseTrainer
        return super().after_fit_step() 
