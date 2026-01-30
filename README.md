# InfoNCE Geometry

This repository contains the numerical validations used the paper:
 *The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence* (https://arxiv.org/abs/2601.19597). 

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

## Animations

### Unimodal: hypersphere distribution evolution
![Image](https://github.com/user-attachments/assets/693efa50-477d-4232-b6b6-ac56db311041)

*Unimodal hypersphere distribution animation.*


## Multimodal: joint-angle coupling evolution
![Image](https://github.com/user-attachments/assets/af8ed31c-1a04-4209-9b46-0ebeb561406a)

*Animated joint-angle coupling: The diagonal concentration indicates improved pairwise coupling, while persistent off-diagonal mass reflects mismatch-induced structure that does not disappear with longer training.*


## Citation

```bibtex
@misc{cai2026geometric,
  title        = {The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence},
  author       = {Cai, Yichao and Zhang, Zhen and Liu, Yuhang and Shi, Javen Qinfeng},
  year         = {2026},
  eprint       = {2601.19597},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  doi          = {10.48550/arXiv.2601.19597},
  url          = {https://arxiv.org/abs/2601.19597}
}
```