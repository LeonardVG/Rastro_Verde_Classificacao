cd /home/marilton-lab/Leonardo/Classification-of-Plant-Species-YOLO26
source ./venv/Scripts/activate


for res in 448 512 640; do
  echo "Rodando resolucao $res..."
  python ./scripts/project.py model yolo26m-cls.pt \
    --train \
    --bench \
    --epoch 40 \
    --batch 8 \
    --dataset-name "AugOffline_$res" \
    --train-path ./cls/dataset_Outros_clahe_2/train \
    --val-path ./cls/dataset_Outros_clahe_2/val \
    --test-path ./cls/dataset_Outros_clahe_2/val \
    --save-models-path "./Models/treino_4/baseline/clahe_2_$res" \
    --result-path "./results/treino_4/baseline/clahe_2_$res" \
    --imgsz $res
done