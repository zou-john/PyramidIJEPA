import hydra
import torch
import torch.nn.functional as F
from typing import Literal, Optional
import einops

from stable_ssl.trainer import SupervisedTrainer

empty_module = torch.nn.Module()

# NOTE: USED IN FINETUNING
class ClassificationTrainer(SupervisedTrainer):

    required_modules = {
        "backbone": torch.nn.Module,
        "predictor": torch.nn.Module,
    }

    def __init__(self,
                 pooling: Literal['mean', 'max', 'flatten', 'sum'],
                 downsampling: Optional[int] = None,
                 num_scales: Optional[int] = None,
                 downstream_model: Literal['mlp', 'linear'] = 'mlp',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the pooling method to use for the encoding
        self.pooling = pooling
        # the number of scales to use for the latent pyramid
        self.num_scales = num_scales
        # the downsampling factor to use for the latent pyramid
        self.downsampling = downsampling
        # the downstream model to use for the classification task
        self.downstream_model = downstream_model 

        # self._data and self._modules are the data and modules that are passed to the trainer and initialized in SupervisedTrainer
        _tmp_data = hydra.utils.instantiate(self._data, _convert_ = "object")
        _tmp_modules = hydra.utils.instantiate(self._modules, _convert_ = "object")

        # self._optim is the optimizer that is passed to the trainer and initialized in SupervisedTrainer
        self.epochs: int = self._optim['epochs']
        # iterations per epoch is the number of batches in the train loader
        self.ipe = sum(len(loader) for k, loader in _tmp_data.items() if isinstance(loader, torch.utils.data.DataLoader) and k == 'train')
        # the total number of steps is the number of epochs times the number of iterations per epoch
        self._optim["scheduler"]["total_steps"] = self.epochs * self.ipe

        sample_batch = next(iter(_tmp_data['train']))
        with torch.no_grad():
            pooled = self.process_encoding(_tmp_modules['backbone'], sample_batch[0])
        # the number of features is the number of channels in the pooled encoding
        self._module['in_features'] = pooled.shape[1]
    
    def process_encoding(self, encoding: torch.Tensor):
        """
        Process the encoding to be used for the classification task.
        """
        if self.downsampling is not None: # time to downsample before pooling
            encoding_pyramid = build_latent_pyramid(encoding, self.num_sclaes or 12)
            # downsampling factor
            encoding = encoding_pyramid[len(encoding_pyramid) - self.downsampling - 1]
            # encoding pyramid stores index of different latent represetnations from coarse to fine.
        if self.pooling in ['mean', 'max', 'sum']:
            # einops is a library for manipulating tensors with a simple syntax
            pooled = einops.reduce(encoding, 'b n d -> b d' , reduction=self.pooling) 
        elif self.pooling == 'flatten':
            pooled = einops.rearrange(encoding, 'b n d -> b (n d)')
        else: raise ValueError(f"Invalid pooling method: {self.pooling}")
        return pooled
    
    def forward(self, *args, **kwargs):
        """
        Forward pass for the classification task with additional steps from StableSSL.
        """
        encoder = self._modules['backbone'](*args, **kwargs)
        pooled = self.process_encoding(encoder)

        # the global steps is initialized in the SupervisedTrainer
        if self.global_step % self.logger['log_every_n_steps'] == 0:
            # initial logger for the first step
            self._log({
                ###### pooled stats
                "stats/pooled/feat_mean": pooled.mean().item(),
                "stats/pooled/feat_std": pooled.std().item(),
                "stats/pooled/feat_min": pooled.min().item(),
                "stats/pooled/feat_max": pooled.max().item(),
                ###### encoding stats
                "stats/encoding/feat_mean": encoder.mean().item(),
                "stats/encoding/feat_std": encoder.std().item(),
                "stats/encoding/feat_min": encoder.min().item(),
                "stats/encoding/feat_max": encoder.max().item(),
            }, commit=False)
        
        return self.modules['predictor'](pooled) # called from the VisionTransformerPredictor
    
    def predict(self):
        """
        Predict the class of the input image.
        """
        # if pooling is mean, max, or sum, the input is B x D
        # if pooling is flatten, the input is B x (N x D)
        return self.forward(self.batch[0])

    def compute_loss(self):
        """
        Compute the loss for the classification task using the cross entropy loss for classification.
        """
        loss = F.cross_entropy(self.predict(), self.batch[1])
        return loss
    