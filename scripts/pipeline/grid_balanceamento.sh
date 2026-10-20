cd /home/marilton-lab/Leonardo/Classification-of-Plant-Species-YOLO26
source ./venv/Scripts/activate


for gamma in 0.0 1.0 2.0; do
  python ./scripts/project.py model yolo26m-cls.pt \
    --train \
    --bench \
    --epoch 40 \
    --dataset-name com_class_weighted_$gamma \
    --train-path ./cls/dataset_Outros/train \
    --val-path ./cls/dataset_Outros/val \
    --test-path ./cls/dataset_Outros/val \
    --save-models-path ./Models/treino_4/Balanceamento_semFlag \
    --result-path ./results/treino_4/balanceamento_semFlag \
    --gamma $gamma     \
    --class-weighted   \
    --imgsz 448        \
    --batch 8

  python ./scripts/project.py model yolo26m-cls.pt \
    --train \
    --bench \
    --epoch 40  \
    --dataset-name sem_class_Weighted_$gamma \
    --train-path ./cls/dataset_Outros/train \
    --val-path ./cls/dataset_Outros/val \
    --test-path ./cls/dataset_Outros/val \
    --save-models-path ./Models/treino_4/Balanceamento_semFlag \
    --result-path ./results/treino_4/balanceamento_semFlag \
    --gamma $gamma     \
    --imgsz 448        \
    --batch 8
done
