cd /home/marilton-lab/Leonardo/Classification-of-Plant-Species-YOLO26
source ./venv/Scripts/activate

for sz in 512; do
python3 ./scripts/pipeline/cv_confirm.py \
  --model yolo26m-cls.pt --epoch 30 --k 3 --seed 42 \
  --imgsz $sz --batch 16 --optimizer AdamW --lr0 0.001 \
  --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 \
  --scale 0.5 --fliplr 0.5 \
  --auto_augment randaugment --erasing 0.4 --weight-decay 0.05 \
  --train-path ./cls/dataset_Outros/train \
  --val-path ./cls/dataset_Outros/val \
  --fold-output ./cls/dataset_kfold3 --skip-fold-gen \
  --models-path ./Models/Base \
  --save-models-path ./Models/treino_5/resolucao/cv_06_res$sz \
  --result-path ./results/treino_5/resolucao/cv_06_res$sz \
  --csv-out ./results/treino_5/resolucao/cv_06_res$sz/summary.csv
done