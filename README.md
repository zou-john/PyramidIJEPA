# PyramidIJEPA

**Pyramid I-JEPA (PI-JEPA)** is an advanced self-supervised learning framework that enhances representation learning efficiency through innovative cross-scale prediction. Inspired by hierarchical processing in human vision, PI-JEPA incorporates multi-scale analysis to accelerate learning while improving feature robustness across resolutions.

## Key Features

- **Multi-scale Representation Learning**  
  Constructs a latent pyramid through progressive downsampling, capturing coarse-to-fine visual information
- **Bidirectional Scale Prediction**  
  Simultaneously predicts representations across multiple resolutions for richer context
- **5× Faster Convergence**  
  Achieves comparable performance to I-JEPA in just 1/5th the training time (10 vs 50 epochs)
- **Scale-Robust Features**  
  Maintains 92% accuracy when evaluated on downsampled representations (vs 80% for I-JEPA)
- **Flexible Masking Strategies**  
  Supports multiple scale-space approaches:
  - Full downsampling
  - Full resampling
  - Coarse-only masking
  - Fine-only masking

## Acknowledgements
This project is a reimplementation of the final project from the graduate course CSCI2952N: Advanced Topics in Deep Learning at Brown University. I would like to thank my teammates Nikunj Harlalka, Jiayi Shen, and especially Sami Bou Ghanem, whose guidance and contributions to the project's ideation were instrumental.
