python coco_pretrained_gap.py \
    --coco_root ../../Dataset/coco2017 \
    --model_name RN50 \
    --pretrained openai \
    --checkpoint exp2_samecat_runs/RN50_openai_samecat_p0.00_heads_seed0/ckpt_step5000.pt \
    --amp \
    --batch_images 128 \
    --bootstrap 20 \
    --metric_subsample 2000 \
    --median_subsample 2000 \
    --shuffle_control \
    --out_json exp2_RN50_p0.00_eval.json \
    --cache_pt exp2_RN50_p0.00_cache.pt

python coco_pretrained_gap.py \
    --coco_root ../../Dataset/coco2017 \
    --model_name RN50 \
    --pretrained openai \
    --checkpoint exp2_samecat_runs/RN50_openai_samecat_p0.25_heads_seed0/ckpt_step5000.pt \
    --amp \
    --batch_images 128 \
    --bootstrap 20 \
    --metric_subsample 2000 \
    --median_subsample 2000 \
    --shuffle_control \
    --out_json exp2_RN50_p0.25_eval.json \
    --cache_pt exp2_RN50_p0.25_cache.pt

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

python coco_pretrained_gap.py \
    --coco_root ../../Dataset/coco2017 \
    --model_name RN50 \
    --pretrained openai \
    --checkpoint exp2_samecat_runs/RN50_openai_samecat_p0.75_heads_seed0/ckpt_step5000.pt \
    --amp \
    --batch_images 128 \
    --bootstrap 20 \
    --metric_subsample 2000 \
    --median_subsample 2000 \
    --shuffle_control \
    --out_json exp2_RN50_p0.75_eval.json \
    --cache_pt exp2_RN50_p0.75_cache.pt

python coco_pretrained_gap.py \
    --coco_root ../../Dataset/coco2017 \
    --model_name RN50 \
    --pretrained openai \
    --checkpoint exp2_samecat_runs/RN50_openai_samecat_p1.00_heads_seed0/ckpt_step5000.pt \
    --amp \
    --batch_images 128 \
    --bootstrap 20 \
    --metric_subsample 2000 \
    --median_subsample 2000 \
    --shuffle_control \
    --out_json exp2_RN50_p1.00_eval.json \
    --cache_pt exp2_RN50_p1.00_cache.pt