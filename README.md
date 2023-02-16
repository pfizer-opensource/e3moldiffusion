# E(3) Equivariant Diffusion for Molecules

Research repository exploring the capabalities of (continuous) denoising diffusion probabilistic models applied on molecular data.

## Installation
Best installed using conda.
```bash
conda env create -f environment.yml
```
For training on the HPC, a gclib error can occur. This is most likely due to `pytorch-spline-conv` package.
Just uninstall and install pytorch geometric again

```bash
conda activate e3moldiffusion
conda uninstall pytorch-spline-conv
conda install pyg -c pyg
```

## Experiments
Python scripts for experiments on the GEOM database are located in `geom/`.
Currently only experiments on GEOM-Drugs database are performed.

## Useful resources
- [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)
 - [Score-Based Generative Modeling through Stochastic Differential Equations (2021)](https://arxiv.org/abs/2011.13456)
 - [GeoDiff: a Geometric Diffusion Model for Molecular Conformation Generation (2022)](https://arxiv.org/abs/2203.02923)
 - [Equivariant Diffusion for Molecule Generation in 3D (2022)](https://arxiv.org/abs/2203.17003)

## To Do's:
1) Only 3D coordinates learning on GEOM-Drugs using variance-preserving SDE.
2) Implement Evaluation functions to track during training instead of only the denoising loss.
3) Try out energy-preserving score-model so the final (energy-based) model can be also be used to compare configurations.

