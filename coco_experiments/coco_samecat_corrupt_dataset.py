import os, json, random
from typing import Dict, List, Tuple
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

class CocoSameCatCorruptPairs(Dataset):
    """
    Returns (image_tensor, text_tokens).

    With prob p_corrupt:
      pick a random category among the image's categories,
      pick a *different* image from that category,
      then pick a random caption of that other image.
    With prob 1-p_corrupt:
      pick a random caption of the current image.

    Fallbacks:
      - if missing categories or category pool too small -> random other image
    """
    def __init__(
        self,
        coco_root: str,
        index_json: str,
        preprocess,     # open_clip train preprocess
        tokenize,       # open_clip tokenizer
        p_corrupt: float,
        seed: int = 0,
    ):
        self.coco_root = coco_root
        self.train_img_dir = os.path.join(coco_root, "train2017")
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.p = float(p_corrupt)
        self.seed = int(seed)

        with open(index_json, "r") as f:
            idx = json.load(f)

        self.img_ids: List[int] = [int(x) for x in idx["img_ids"]]
        self.caps_by_img: Dict[int, List[str]] = {int(k): v for k, v in idx["caps_by_img"].items()}
        self.cats_by_img: Dict[int, List[int]] = {int(k): v for k, v in idx["cats_by_img"].items()}
        self.imgs_by_cat: Dict[int, List[int]] = {int(k): v for k, v in idx["imgs_by_cat"].items()}

        # COCO uses 12-digit zero-padded file names
        self._path = lambda img_id: os.path.join(self.train_img_dir, f"{img_id:012d}.jpg")

    def __len__(self):
        return len(self.img_ids)

    def _sample_caption(self, rng: random.Random, img_id: int) -> str:
        caps = self.caps_by_img[img_id]
        return caps[rng.randrange(len(caps))]

    def _sample_random_other_image(self, rng: random.Random, img_id: int) -> int:
        # sample until different
        while True:
            j = self.img_ids[rng.randrange(len(self.img_ids))]
            if j != img_id:
                return j

    def _sample_same_category_other_image(self, rng: random.Random, img_id: int) -> int:
        cats = self.cats_by_img.get(img_id, [])
        if not cats:
            return self._sample_random_other_image(rng, img_id)

        # choose one category possessed by this image
        c = cats[rng.randrange(len(cats))]
        pool = self.imgs_by_cat.get(c, [])
        if len(pool) <= 1:
            return self._sample_random_other_image(rng, img_id)

        # try a few times to pick a different image
        for _ in range(20):
            j = pool[rng.randrange(len(pool))]
            if j != img_id and j in self.caps_by_img:
                return j

        return self._sample_random_other_image(rng, img_id)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.img_ids[idx]

        # per-sample RNG that is stable across dataloader workers
        # (idx + seed) makes it reproducible; the extra random draw comes from dataloader shuffling
        rng = random.Random(self.seed + idx * 1000003)

        # choose caption source
        if rng.random() < self.p:
            src_img = self._sample_same_category_other_image(rng, img_id)
        else:
            src_img = img_id

        caption = self._sample_caption(rng, src_img)

        im = Image.open(self._path(img_id)).convert("RGB")
        im_t = self.preprocess(im)
        txt_t = self.tokenize([caption])[0]  # [seq_len]

        return im_t, txt_t