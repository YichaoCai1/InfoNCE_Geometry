import os, json, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip
from coco_samecat_corrupt_dataset import CocoSameCatCorruptPairs

def symmetric_infonce(img_f, txt_f, logit_scale):
    logits = logit_scale * (img_f @ txt_f.t())
    labels = torch.arange(img_f.size(0), device=img_f.device)
    li = F.cross_entropy(logits, labels)
    lt = F.cross_entropy(logits.t(), labels)
    return 0.5 * (li + lt)

def set_trainable(model, mode="heads"):
    """
    mode:
      - "heads": only projection heads + logit_scale
      - "heads_last": heads + last block/layer of both towers (more sensitive, slower)
    """
    for p in model.parameters():
        p.requires_grad = False

    # always train logit_scale if present
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    # projection heads differ slightly across backbones
    for n, p in model.named_parameters():
        if ("text_projection" in n) or ("visual.proj" in n) or ("visual_projection" in n):
            p.requires_grad = True

    if mode == "heads_last":
        # Vision: ResNet uses visual.layer4; ViT uses visual.transformer.resblocks
        for n, p in model.named_parameters():
            if "visual.layer4" in n:
                p.requires_grad = True
            # ViT-like
            if "visual.transformer.resblocks" in n:
                # unfreeze only last block
                if n.split("visual.transformer.resblocks.")[1].startswith(str(len(model.visual.transformer.resblocks)-1)):
                    p.requires_grad = True

        # Text: transformer.resblocks
        for n, p in model.named_parameters():
            if "transformer.resblocks" in n:
                if n.split("transformer.resblocks.")[1].startswith(str(len(model.transformer.resblocks)-1)):
                    p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--model_name", default="RN50")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--p_corrupt", type=float, required=True)
    ap.add_argument("--train_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trainable", choices=["heads","heads_last"], default="heads")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out_dir", default="exp2_samecat_runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)

    set_trainable(model, mode=args.trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable params selected. Check set_trainable().")

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ds = CocoSameCatCorruptPairs(
        coco_root=args.coco_root,
        index_json=args.index_json,
        preprocess=preprocess_train,
        tokenize=tokenizer,
        p_corrupt=args.p_corrupt,
        seed=args.seed,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True
    )

    run_name = f"{args.model_name}_{args.pretrained}_samecat_p{args.p_corrupt:.2f}_{args.trainable}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model.train()
    t0 = time.time()
    step = 0

    it = iter(dl)
    while step < args.train_steps:
        try:
            images, texts = next(it)
        except StopIteration:
            it = iter(dl)
            images, texts = next(it)

        images = images.to(device, non_blocking=True)
        texts  = texts.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            img_f = model.encode_image(images)
            txt_f = model.encode_text(texts)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp().clamp(1e-3, 100.0)
            loss = symmetric_infonce(img_f, txt_f, logit_scale)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        step += 1
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"[{run_name}] step={step}/{args.train_steps} loss={loss.item():.4f} time={dt:.1f}s")

        if step % args.save_every == 0 or step == args.train_steps:
            ckpt_fp = os.path.join(out_dir, f"ckpt_step{step}.pt")
            torch.save({"model": model.state_dict()}, ckpt_fp)
            print(f"Saved: {ckpt_fp}")

if __name__ == "__main__":
    main()