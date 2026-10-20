cd /home/marilton-lab/Leonardo/Classification-of-Plant-Species-YOLO26
source ./venv/Scripts/activate

#python3 ./scripts/project.py model yolo26m-cls.pt \
 #   --train \
 #   --bench \
 #   --epoch 40 \
 #   --dataset-name AugOffline \
 #   --train-path ./cls/dataset_Outros_clahe_3/train  \
 #   --val-path ./cls/dataset_Outros_clahe_3/val \
 #   --test-path ./cls/dataset_Outros_clahe_3/val \
 #   --save-models-path ./Models/treino_4/baseline/clahe_3 \
 #   --result-path ./results/treino_4/baseline/clahe_3\


# as flags "  --imgsz 224 --batch 16 --optimizer AdamW --lr0 0.001 \"
#é apenas para registro, pois é o default
#python3 ./scripts/project.py model yolo26m-cls.pt \
#  --train --bench --epoch 30 \
#  --dataset-name baseline \
#  --train-path ./cls/dataset_Outros/train \
#  --val-path ./cls/dataset_Outros/val \
#  --test-path ./cls/dataset_Outros/val \
#  --imgsz 224 --batch 16 --optimizer AdamW --lr0 0.001 \
#  --save-models-path ./Models/treino_5/baseline/02_base \
#  --result-path ./results/treino_5/baseline/02_base

  #Com aug para evitar queda rapida do loss
  python3 ./scripts/project.py model yolo26m-cls.pt \
  --train --bench --epoch 30 \
  --dataset-name baseline_aug \
  --train-path ./cls/dataset_Outros/train \
  --val-path ./cls/dataset_Outros/val \
  --test-path ./cls/dataset_Outros/val \
  --imgsz 224 --batch 16 --optimizer AdamW --lr0 0.001 \
  --aug-hsv_h 0.015 --aug-hsv_s 0.7 --aug-hsv_v 0.4 \
  --aug-scale 0.5 --aug-fliplr 0.5 \
  --aug-auto_augment randaugment --aug-erasing 0.4 --weight-decay 0.05\
  --save-models-path ./Models/treino_5/baseline/04_aug \
  --result-path ./results/treino_5/baseline/04_aug