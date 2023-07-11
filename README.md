# E(3) Equivariant Diffusion for Molecules

Research repository exploring the capabalities of (continuous and discrete) denoising diffusion probabilistic models applied on molecular data.

## Installation
Best installed using mamba.
```bash
mamba env create -f environment.yml
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

