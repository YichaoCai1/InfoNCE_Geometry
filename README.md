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

## Unimodal: hypersphere distribution evolution
<p align="center"> <a href="unimodal_gibbs/animation/uni_train_s2_potential_overlay_evolution.mp4"> <img src="unimodal_gibbs/animation/uni_train_s2_potential_and_overlays.png" width="820" alt="Unimodal hypersphere distribution animation (click to play)"> </a> <br/> <em>Unimodal hypersphere distribution animation (click to play).</em> </p>


## Multimodal: joint-angle coupling evolution
<p align="center"> <img src="modality_gap/mm_joint_cosine_anim_grid_2x4_individual_cbars.gif" width="820" alt="Animated joint-angle coupling (1 frame per 10 steps)"> <br/> <em> Animated joint-angle coupling (1 frame per 10 steps). Diagonal concentration indicates improved pairwise coupling, while persistent off-diagonal mass reflects mismatch-induced structure that does not disappear with longer training. </em> </p>