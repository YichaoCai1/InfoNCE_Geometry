import os
import json
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from pycocotools.coco import COCO


# -------------------------
# Repro + utils
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def check_coco(coco_root: str) -> Tuple[str, str]:
    img_dir = os.path.join(coco_root, "val2017")
    ann_path = os.path.join(coco_root, "annotations", "captions_val2017.json")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing COCO dir: {img_dir}")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing COCO ann: {ann_path}")
    return img_dir, ann_path

def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float((a @ b) / ((a.norm() + eps) * (b.norm() + eps)))


# -------------------------
# Distance helpers on unit sphere
# -------------------------
def dist_from_cos(cos_sim: torch.Tensor) -> torch.Tensor:
    # if ||x||=||y||=1: ||x-y|| = sqrt(2 - 2<x,y>)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0))


# -------------------------
# Gap metrics (pair-weighted by default)
# -------------------------
@torch.inference_mode()
def centroid_stats(z_img: torch.Tensor, z_txt: torch.Tensor) -> Dict[str, float]:
    mu_i = z_img.mean(0)
    mu_t = z_txt.mean(0)
    return {
        "centroid_gap": float(torch.norm(mu_i - mu_t)),
        "mu_img_norm": float(mu_i.norm()),
        "mu_txt_norm": float(mu_t.norm()),
        "mu_cosine": cosine(mu_i, mu_t),
    }

@torch.inference_mode()
def energy_distance(z_img: torch.Tensor, z_txt: torch.Tensor, n: int, seed: int) -> Dict[str, float]:
    """
    Returns energy distance plus its components to aid interpretation:
      d_xy = E||X-Y||, d_xx = E||X-X'||, d_yy = E||Y-Y'||
      energy = 2 d_xy - d_xx - d_yy
    """
    set_seed(seed)
    n = min(n, len(z_img), len(z_txt))
    Xi = z_img[torch.randperm(len(z_img))[:n]]
    Yt = z_txt[torch.randperm(len(z_txt))[:n]]

    d_xy = dist_from_cos(Xi @ Yt.T).mean()
    d_xx = dist_from_cos(Xi @ Xi.T).mean()
    d_yy = dist_from_cos(Yt @ Yt.T).mean()
    energy = 2.0 * d_xy - d_xx - d_yy

    return {
        "energy": float(energy),
        "d_xy": float(d_xy),
        "d_xx": float(d_xx),
        "d_yy": float(d_yy),
    }   

@torch.inference_mode()
def median_heuristic_sigmas(z_img: torch.Tensor, z_txt: torch.Tensor, n_med: int, seed: int) -> Dict[str, float]:
    """
    Choose RBF bandwidths from the median of cross distances.
    Returns med distance + sigma grid (0.5*med, 1*med, 2*med).
    """
    set_seed(seed)
    n_med = min(n_med, len(z_img), len(z_txt))
    Xi = z_img[torch.randperm(len(z_img))[:n_med]]
    Yt = z_txt[torch.randperm(len(z_txt))[:n_med]]

    # cross distances
    D = dist_from_cos(Xi @ Yt.T).flatten()
    med = float(D.median())
    # avoid zero / degenerate
    med = max(med, 1e-6)
    return {
        "median_cross_dist": med,
        "sigma_0.5x": 0.5 * med,
        "sigma_1.0x": 1.0 * med,
        "sigma_2.0x": 2.0 * med,
    }

@torch.inference_mode()
def mmd_rbf(z_img: torch.Tensor, z_txt: torch.Tensor, sigmas: List[float], n: int, seed: int) -> Dict[str, float]:
    """
    MMD^2 with RBF kernel on unit sphere:
      dist^2 = 2 - 2cos
      k = exp(-dist^2 / (2 sigma^2))
    """
    set_seed(seed)
    n = min(n, len(z_img), len(z_txt))
    Xi = z_img[torch.randperm(len(z_img))[:n]]
    Yt = z_txt[torch.randperm(len(z_txt))[:n]]

    Cxx = Xi @ Xi.T
    Cyy = Yt @ Yt.T
    Cxy = Xi @ Yt.T

    Dxx = 2.0 - 2.0 * Cxx
    Dyy = 2.0 - 2.0 * Cyy
    Dxy = 2.0 - 2.0 * Cxy

    out = {}
    for s in sigmas:
        s = float(s)
        Kxx = torch.exp(-Dxx / (2.0 * s * s)).mean()
        Kyy = torch.exp(-Dyy / (2.0 * s * s)).mean()
        Kxy = torch.exp(-Dxy / (2.0 * s * s)).mean()
        out[f"mmd2_sigma={s:.4f}"] = float(Kxx + Kyy - 2.0 * Kxy)
    return out


# -------------------------
# Retrieval (correct for 5 captions / image) – chunked
# -------------------------
@torch.inference_mode()
def recall_i2t(z_img, z_txt, cap_to_img, Ks=(1, 5, 10), device="cuda", chunk=256) -> Dict[str, float]:
    z_img = z_img.to(device)
    z_txt = z_txt.to(device)
    cap_to_img = cap_to_img.to(device)
    N = z_img.shape[0]
    maxK = max(Ks)
    correct = {K: 0 for K in Ks}

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        sims = z_img[i0:i1] @ z_txt.T
        topk = torch.topk(sims, k=maxK, dim=1).indices
        hit_img = cap_to_img[topk]
        gt = torch.arange(i0, i1, device=device).unsqueeze(1)
        for K in Ks:
            correct[K] += (hit_img[:, :K] == gt).any(dim=1).sum().item()

    return {f"i2t_R@{K}": correct[K] / N for K in Ks}

@torch.inference_mode()
def recall_t2i(z_img, z_txt, cap_to_img, Ks=(1, 5, 10), device="cuda", chunk=512) -> Dict[str, float]:
    z_img = z_img.to(device)
    z_txt = z_txt.to(device)
    cap_to_img = cap_to_img.to(device)
    M = z_txt.shape[0]
    maxK = max(Ks)
    correct = {K: 0 for K in Ks}

    for j0 in range(0, M, chunk):
        j1 = min(j0 + chunk, M)
        sims = z_txt[j0:j1] @ z_img.T
        topk = torch.topk(sims, k=maxK, dim=1).indices
        gt = cap_to_img[j0:j1].unsqueeze(1)
        for K in Ks:
            correct[K] += (topk[:, :K] == gt).any(dim=1).sum().item()

    return {f"t2i_R@{K}": correct[K] / M for K in Ks}


# -------------------------
# COCO embedding extraction (cacheable)
# -------------------------
@torch.inference_mode()
def extract_embeddings(
    coco_root: str,
    model_name: str,
    pretrained: str,
    device: str,
    num_images: int,
    batch_images: int,
    captions_per_image: int,
    amp: bool,
    checkpoint: str = "",
    strict_load: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    img_dir, ann_path = check_coco(coco_root)
    coco = COCO(ann_path)

    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
        if (not strict_load) and (len(missing) > 0 or len(unexpected) > 0):
            print("[warn] load_state_dict used strict=False")
            if len(missing) > 0:
                print(f"[warn] missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing)>10 else ''}")
            if len(unexpected) > 0:
                print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")

        print(f"Loaded checkpoint: {checkpoint}")
        

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    img_ids = sorted(coco.getImgIds())[:num_images]

    z_img_list = []
    z_txt_list = []
    cap_to_img = []

    for i0 in tqdm(range(0, len(img_ids), batch_images), desc="Encoding COCO val"):
        ids_batch = img_ids[i0:i0 + batch_images]

        imgs = []
        caps = []
        local_map = []

        for local_i, coco_img_id in enumerate(ids_batch):
            meta = coco.loadImgs([coco_img_id])[0]
            path = os.path.join(img_dir, meta["file_name"])
            with Image.open(path) as im:
                im = im.convert("RGB")
                imgs.append(preprocess_val(im))

            ann_ids = coco.getAnnIds(imgIds=[coco_img_id])
            anns = coco.loadAnns(ann_ids)
            cap_list = [a["caption"] for a in anns][:captions_per_image]
            for c in cap_list:
                caps.append(c)
                local_map.append(i0 + local_i)

        imgs = torch.stack(imgs, 0).to(device, non_blocking=True)
        txt = tokenizer(caps).to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):
            img_feat = model.encode_image(imgs)
            txt_feat = model.encode_text(txt)

        img_feat = l2_normalize(img_feat.float()).cpu()
        txt_feat = l2_normalize(txt_feat.float()).cpu()

        z_img_list.append(img_feat)
        z_txt_list.append(txt_feat)
        cap_to_img.extend(local_map)

    z_img = torch.cat(z_img_list, 0)
    z_txt = torch.cat(z_txt_list, 0)
    cap_to_img = torch.tensor(cap_to_img, dtype=torch.long)

    # sanity checks
    assert z_img.shape[0] == len(img_ids), "Image count mismatch."
    assert z_txt.shape[0] == cap_to_img.shape[0], "Caption mapping mismatch."
    assert int(cap_to_img.min()) >= 0 and int(cap_to_img.max()) < len(img_ids), "cap_to_img out of range."
    assert abs(float(z_img.norm(dim=1).mean()) - 1.0) < 1e-2, "Image embeddings not unit-normalized?"
    assert abs(float(z_txt.norm(dim=1).mean()) - 1.0) < 1e-2, "Text embeddings not unit-normalized?"

    return z_img, z_txt, cap_to_img, img_ids


# -------------------------
# Main eval (with bootstrap + shuffle control)
# -------------------------
def mean_std(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="ViT-B-32")   # e.g., ViT-L-14, RN50
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_images", type=int, default=5000)
    ap.add_argument("--batch_images", type=int, default=64)
    ap.add_argument("--captions_per_image", type=int, default=5)

    ap.add_argument("--cache_pt", type=str, default="")
    ap.add_argument("--use_cache", action="store_true")

    ap.add_argument("--metric_subsample", type=int, default=1500)
    ap.add_argument("--median_subsample", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--shuffle_control", action="store_true")
    ap.add_argument("--out_json", type=str, default="exp1_metrics_v2.json")
    
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional fine-tuned checkpoint path. Expected format: torch.save({'model': state_dict}) "
            "or a raw state_dict.",
    )
    ap.add_argument(
        "--strict_load",
        action="store_true",
        help="Use strict=True for load_state_dict. Default is strict=False with warnings.",
    )
    
    args = ap.parse_args()

    set_seed(args.seed)

    # Load or extract
    if args.use_cache and args.cache_pt and os.path.isfile(args.cache_pt):
        ckpt = torch.load(args.cache_pt, map_location="cpu")
        z_img = ckpt["z_img"]
        z_txt = ckpt["z_txt"]
        cap_to_img = ckpt["cap_to_img"]
        img_ids = ckpt.get("img_ids", list(range(z_img.shape[0])))
        encode_seconds = ckpt.get("encode_seconds", None)
        
        if args.checkpoint:
            cached_ckpt = ckpt.get("checkpoint", "")
            if os.path.abspath(cached_ckpt) != os.path.abspath(args.checkpoint):
                raise ValueError(
                    f"Cache checkpoint mismatch.\n"
                    f"  cache: {cached_ckpt}\n"
                    f"  args : {args.checkpoint}\n"
                    f"Use a different --cache_pt or disable --use_cache."
                )
    else:
        t0 = time.time()
        z_img, z_txt, cap_to_img, img_ids = extract_embeddings(
            coco_root=args.coco_root,
            model_name=args.model_name,
            pretrained=args.pretrained,
            device=args.device,
            num_images=args.num_images,
            batch_images=args.batch_images,
            captions_per_image=args.captions_per_image,
            amp=args.amp,
            checkpoint=args.checkpoint,
            strict_load=args.strict_load,
        )
        encode_seconds = time.time() - t0

        if args.cache_pt:
            torch.save(
                {
                    "z_img": z_img,
                    "z_txt": z_txt,
                    "cap_to_img": cap_to_img,
                    "img_ids": img_ids,
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "encode_seconds": encode_seconds,
                    "checkpoint": args.checkpoint,
                    "strict_load": bool(args.strict_load),
                },
                args.cache_pt
            )

    # Pair-weighted image marginal (caption-weighted)
    z_img_rep = z_img[cap_to_img]

    # Choose sigma grid by median heuristic (data-driven)
    sigma_info = median_heuristic_sigmas(z_img_rep, z_txt, n_med=args.median_subsample, seed=args.seed)
    sigmas = [sigma_info["sigma_0.5x"], sigma_info["sigma_1.0x"], sigma_info["sigma_2.0x"]]

    # Deterministic summary (centroid stats + retrieval)
    metrics = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "num_images": int(z_img.shape[0]),
        "num_captions": int(z_txt.shape[0]),
        "encode_seconds": float(encode_seconds) if encode_seconds is not None else None,
        "metric_subsample": int(args.metric_subsample),
        "median_subsample": int(args.median_subsample),
        "bootstrap": int(args.bootstrap),
        "sigmas": [float(s) for s in sigmas],
        **sigma_info,
        **centroid_stats(z_img, z_txt),
    }

    metrics.update(recall_i2t(z_img, z_txt, cap_to_img, device=args.device))
    metrics.update(recall_t2i(z_img, z_txt, cap_to_img, device=args.device))

    # Bootstrap energy + MMD for uncertainty
    rows = []
    for b in range(args.bootstrap):
        seed_b = args.seed + 1000 + b
        e = energy_distance(z_img_rep, z_txt, n=args.metric_subsample, seed=seed_b)
        m = mmd_rbf(z_img_rep, z_txt, sigmas=sigmas, n=args.metric_subsample, seed=seed_b)
        row = {}
        row.update(e)
        row.update(m)
        rows.append(row)

    # Aggregate
    keys = list(rows[0].keys()) if rows else []
    metrics.update(mean_std(rows, keys))

    # Shuffle control (break “ground truth pairing” only)
    if args.shuffle_control:
        set_seed(args.seed + 9999)
        perm = torch.randperm(len(cap_to_img))
        cap_to_img_shuf = cap_to_img[perm]

        # retrieval should collapse; gap metrics should be (nearly) unchanged
        metrics.update({f"shuf_{k}": v for k, v in recall_i2t(z_img, z_txt, cap_to_img_shuf, device=args.device).items()})
        metrics.update({f"shuf_{k}": v for k, v in recall_t2i(z_img, z_txt, cap_to_img_shuf, device=args.device).items()})

        z_img_rep_shuf = z_img[cap_to_img_shuf]
        rows_shuf = []
        for b in range(args.bootstrap):
            seed_b = args.seed + 20000 + b
            e = energy_distance(z_img_rep_shuf, z_txt, n=args.metric_subsample, seed=seed_b)
            m = mmd_rbf(z_img_rep_shuf, z_txt, sigmas=sigmas, n=args.metric_subsample, seed=seed_b)
            row = {}
            row.update(e); row.update(m)
            rows_shuf.append(row)
        metrics.update({f"shuf_{k}": v for k, v in mean_std(rows_shuf, list(rows_shuf[0].keys())).items()})

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== Exp1 revised metrics ====")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved: {args.out_json}")
    if args.cache_pt:
        print(f"Cache: {args.cache_pt}")


if __name__ == "__main__":
    main()
    
    """
    Experiment 1 usage:
    
    python coco_pretrained_gap.py \
        --coco_root ../Dataset/coco2017 \
        --model_name RN50 \
        --pretrained openai \
        --amp \
        --batch_images 128 \
        --bootstrap 20 \
        --metric_subsample 2000 \
        --median_subsample 2000 \
        --shuffle_control \
        --out_json exp1_rn50_v2.json \
        --cache_pt exp1_rn50_cache.pt
    
    
    Experiment 2 usage:
    
    python coco_pretrained_gap.py \
        --coco_root ../../Dataset/coco2017 \
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
    """