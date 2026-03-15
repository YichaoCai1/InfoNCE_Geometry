# InfoNCE Geometry

This repository contains the numerical and COCO-based experiments used in the paper
*The Geometric Mechanics of Contrastive Learning: Alignment Potentials, Entropic Dispersion, and Modality Gap*
([arXiv:2601.19597](https://arxiv.org/abs/2601.19597)).

Each experiment can be run directly from its folder, and most outputs are written next to the script or into an explicitly configured output directory.

## What Is In This Repo

There are two main parts:

- `numerical_val/`: synthetic and low-dimensional validations of the paper's theory.
- `coco_experiments/`: CLIP-based experiments on MS COCO, including training, evaluation, and result summarization scripts.

## Repository Layout

```text
InfoNCE_Geometry/
├── README.md
├── numerical_val/
│   ├── grad_consistency/
│   ├── unimodal_gibbs/
│   └── modality_gap/
└── coco_experiments/
    ├── build_coco_samecat_index.py
    ├── coco_samecat_corrupt_dataset.py
    ├── coco_pretrained_gap.py
    └── train_exp2_samecat.py
```

## Requirements

- Python 3.10+
- A recent PyTorch install
- `numpy`
- `matplotlib`
- `pillow`
- `tqdm`
- `open_clip_torch`
- `pycocotools`

If you want to save MP4 animations, install `ffmpeg`. GIF export works through Pillow.

## Installation

Create an environment, install PyTorch for your platform, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch separately for your system if needed.
pip install numpy matplotlib pillow tqdm open_clip_torch pycocotools
```

## Quick Start

### 1. Synthetic / numerical validations

Each script is self-contained and writes figures to its current directory unless noted otherwise.

Gradient consistency vs number of negatives:

```bash
cd numerical_val/grad_consistency
python large_batch_consistency.py
```

Outputs:

- `grad_alignment_comparison.pdf`
- `grad_relerr_comparison.pdf`

Unimodal Gibbs comparison on the sphere:

```bash
cd numerical_val/unimodal_gibbs
python unimodal_gibbs.py
```

Outputs include:

- `uni_train_s2_potential_and_overlays.pdf`
- `uni_train_s2_overlay_row5.pdf`
- `uni_train_s2_overlay_row4.pdf`
- `uni_train_concentration_vs_tau_s2.pdf`

To also save an animation from the same script:

```bash
cd numerical_val/unimodal_gibbs
python unimodal_gibbs.py --make-animation
```

Multimodal modality-gap toy experiment:

```bash
cd numerical_val/modality_gap
python multimodal_modality_gap.py
```

Outputs include:

- `mm_gapcurve_cosine.pdf`
- `mm_polar_*.pdf`
- `mm_joint_*.pdf`
- `mm_delta_*.pdf`

For the static gap curve, the script uses the raw mean across seeds with a standard-error band:

- all seeds are included in the curve summary
- the plotted line is the mean gap across seeds
- the shaded band is plus/minus one standard error of the mean (SEM)

With the default `20` seeds, the curve is summarized over all `20` runs automatically.

The per-sigma static visualizations (`mm_polar_*`, `mm_joint_*`, `mm_delta_*`) use one fixed representative seed, controlled by `--animation-seed` (default `0`), together with the same shared evaluation angles used by the animation path.

```bash
cd numerical_val/modality_gap
python multimodal_modality_gap.py
```

To also save the joint-angle animation:

```bash
cd numerical_val/modality_gap
python multimodal_modality_gap.py --make-animation
```

### 2. COCO experiments

The COCO scripts expect the MS COCO 2017 layout below:

```text
<COCO_ROOT>/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    ├── captions_val2017.json
    └── instances_train2017.json
```

The evaluator uses `val2017` and `captions_val2017.json`. The corruption-training pipeline uses `train2017`, `captions_train2017.json`, and `instances_train2017.json`.

## Experiment 1: Evaluate Pretrained CLIP Models on COCO

The main evaluation script is:

```bash
cd coco_experiments
python coco_pretrained_gap.py \
  --coco_root /path/to/coco2017 \
  --model_name RN50 \
  --pretrained openai \
  --amp \
  --batch_images 128 \
  --bootstrap 20 \
  --metric_subsample 2000 \
  --median_subsample 2000 \
  --shuffle_control \
  --out_json exp1_rn50.json \
  --cache_pt exp1_rn50_cache.pt
```

What this script does:

- extracts image and caption embeddings with OpenCLIP
- computes retrieval metrics (`I->T` and `T->I` recall)
- computes marginal-gap statistics such as centroid gap, energy distance, and MMD
- optionally caches embeddings to avoid re-encoding

To summarize a folder of evaluation JSON files:

```bash
cd coco_experiments
python exp1_summarize_results.py \
  --result_dir exp1_results \
  --out_prefix exp1_summary/coco_pretrained
```

This writes:

- `exp1_summary/coco_pretrained_table.txt`
- paired PNG/PDF plots under `exp1_summary/`

## Experiment 2: Same-Category Caption Corruption Fine-Tuning

This experiment fine-tunes a subset of CLIP parameters while corrupting image-caption pairs. With corruption probability `p_corrupt`, an image is paired with a caption sampled from a different image in the same COCO category.

### Step 1: Build the training index

```bash
cd coco_experiments
python build_coco_samecat_index.py \
  --coco_root /path/to/coco2017 \
  --out coco_train_samecat_index.json
```

### Step 2: Train corrupted-pair runs

Example for RN50:

```bash
cd coco_experiments
python train_exp2_samecat.py \
  --coco_root /path/to/coco2017 \
  --index_json coco_train_samecat_index.json \
  --model_name RN50 \
  --pretrained openai \
  --p_corrupt 0.50 \
  --train_steps 5000 \
  --batch_size 256 \
  --lr 1e-5 \
  --wd 0.05 \
  --amp \
  --workers 8 \
  --trainable heads \
  --out_dir exp2_samecat_runs \
  --seed 0
```

For the full sweep used in this repo, see:

- `coco_experiments/exp2_train_RN50.sh`
- `coco_experiments/exp2_train_ViT-B-16.sh`

### Step 3: Evaluate trained checkpoints

Use the same evaluation script as Experiment 1, but pass a checkpoint:

```bash
cd coco_experiments
python coco_pretrained_gap.py \
  --coco_root /path/to/coco2017 \
  --model_name RN50 \
  --pretrained openai \
  --checkpoint exp2_samecat_runs/RN50_openai_samecat_p0.50_heads_seed0/ckpt_step5000.pt \
  --amp \
  --batch_images 128 \
  --bootstrap 20 \
  --metric_subsample 2000 \
  --median_subsample 2000 \
  --shuffle_control \
  --out_json exp2_RN50_p0.50_eval.json \
  --cache_pt exp2_RN50_p0.50_cache.pt
```

Batch validation commands for the full sweep are in:

- `coco_experiments/exp2_validation_rn50.sh`
- `coco_experiments/exp2_validation_vitb16.sh`

### Step 4: Summarize Experiment 2 results

```bash
cd coco_experiments
python exp2_summarize_results.py \
  --result_dir exp2_results \
  --out_prefix exp2_summary/exp2
```

This writes:

- `exp2_summary/exp2_table.txt`
- paired PNG/PDF plots under `exp2_summary/`

## Checked-In Artifacts

This repository already includes saved outputs from the reported runs:

- `coco_experiments/exp1_results/`: pretrained-model evaluation JSONs and caches
- `coco_experiments/exp1_summary/`: Experiment 1 tables and plots
- `coco_experiments/exp2_samecat_runs/`: fine-tuned checkpoints
- `coco_experiments/exp2_results/`: Experiment 2 evaluation JSONs and caches
- `coco_experiments/exp2_summary/`: Experiment 2 tables and plots

These artifacts let you inspect the reported results without rerunning the full COCO pipeline.

## Notes

- `coco_pretrained_gap.py` can reuse cached embeddings via `--use_cache`, but it will check that the cache matches the requested checkpoint.
- `--shuffle_control` is useful as a sanity check: retrieval should collapse when pairings are shuffled, while marginal-gap statistics should change much less.
- OpenCLIP model weights may be downloaded the first time you use a model/pretrained combination.
