# InfoNCE Geometry — Numeric Validations

This repository contains the numerical validations used the paper:
 *The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence*. 

## Directory structure
- `grad_consistency/` — large-batch gradient consistency vs number of negatives.
- `unimodal_gibbs/` — unimodal Gibbs equilibrium comparison on the sphere + low-temperature concentration proxy.
- `modality_gap/` — multimodal “marginal gap” vs misalignment scale under symmetric CLIP loss.

## Setup
Tested with Python 3.10+.

Install dependencies:
```bash
pip install numpy matplotlib torch
```

## Reproducing figures

Each experiment is self-contained. Change into the corresponding subdirectory and run the script. PDFs will be written to the same folder by default.