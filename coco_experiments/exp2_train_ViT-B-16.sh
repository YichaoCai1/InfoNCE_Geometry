export COCO=../../Dataset/coco2017

for p in 0.00 0.25 0.50 0.75 1.00; do
  python train_exp2_samecat.py \
    --coco_root $COCO \
    --index_json coco_train_samecat_index.json \
    --model_name ViT-B-16 --pretrained openai \
    --p_corrupt $p \
    --train_steps 5000 \
    --batch_size 256 \
    --lr 1e-5 --wd 0.05 \
    --amp --workers 8 \
    --trainable heads \
    --out_dir exp2_samecat_runs \
    --seed 0
done