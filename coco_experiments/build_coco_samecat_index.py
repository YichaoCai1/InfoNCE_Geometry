import os, json, argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True)
    ap.add_argument("--out", default="coco_train_samecat_index.json")
    args = ap.parse_args()

    ann_dir = os.path.join(args.coco_root, "annotations")
    cap_fp  = os.path.join(ann_dir, "captions_train2017.json")
    inst_fp = os.path.join(ann_dir, "instances_train2017.json")

    with open(cap_fp, "r") as f:
        cap = json.load(f)
    with open(inst_fp, "r") as f:
        inst = json.load(f)

    # image_id -> list[captions]
    caps_by_img = defaultdict(list)
    for a in cap["annotations"]:
        caps_by_img[a["image_id"]].append(a["caption"])

    # image_id -> set[cat_id]
    cats_by_img = defaultdict(set)
    for a in inst["annotations"]:
        cats_by_img[a["image_id"]].add(a["category_id"])

    # cat_id -> list[image_id]
    imgs_by_cat = defaultdict(list)
    for img_id, cats in cats_by_img.items():
        if img_id not in caps_by_img:
            continue
        for c in cats:
            imgs_by_cat[c].append(img_id)

    # keep images that have at least 1 caption and 1 category
    img_ids = sorted(list(set(caps_by_img.keys()) & set(cats_by_img.keys())))

    out = {
        "img_ids": img_ids,
        "caps_by_img": {str(k): v for k, v in caps_by_img.items()},
        "cats_by_img": {str(k): sorted(list(v)) for k, v in cats_by_img.items()},
        "imgs_by_cat": {str(k): v for k, v in imgs_by_cat.items()},
    }

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")
    print(f"num_images_indexed={len(img_ids)}")
    print(f"num_categories_indexed={len(imgs_by_cat)}")

if __name__ == "__main__":
    main()
    
    """
    python build_coco_samecat_index.py --coco_root ../../Dataset/coco2017 --out coco_train_samecat_index.json
    """