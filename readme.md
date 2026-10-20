# Classificação de Espécies de Plantas com YOLO

Pipeline de treinamento e avaliação de um modelo de classificação de imagens
(YOLO, cabeça `-cls`) para identificação de espécies de plantas nativas em
fotos de campo (planta inteira, ambiente natural — não folha/fruto isolado),
com foco em uso para reflorestamento.

## Estrutura do repositório

```
scripts/
├── project.py              # CLI principal: python scripts/project.py <DATASET|MODEL> ...
├── dataset/
│   ├── core.py              # gera splits train/val/test a partir do dataset bruto (um comando do CLI)
│   ├── cls.py                # lógica de split estratificado por classe (usado por core.py e kfold.py)
│   ├── kfold.py               # gera k-folds estratificados a partir de train+val (test nunca entra)
│   ├── clean.py               # limpeza do dataset bruto (corrompidas, borradas, duplicadas via phash)
│   ├── clahe.py               # aplica CLAHE (equalização de contraste) como pré-processamento fixo
│   ├── aug.py                 # augmentation offline em disco, focado nas classes minoritárias
│   └── loss.py                # ClassBalancedFocalLoss (class-weighted + focal loss combináveis)
├── model/
│   ├── core.py                 # lógica de CLI do comando MODEL (treino/benchmark)
│   ├── train.py                  # treino (usa Ultralytics YOLO, suporta trainer com focal loss)
│   ├── bench.py                  # avaliação em um split (calcula macro-F1 a partir da matriz de confusão)
│   └── predict_with_rejection.py  # inferência com rejeição por limiar de confiança ("outros"/open-set)
└── pipeline/
    ├── baseline.sh                # treino/benchmark baseline
    ├── busca_resolucao.sh          # grid search de resolução (448/512/640)
    ├── grid_balanceamento.sh        # grid search de gamma (focal loss) x class-weighted
    ├── optuna_search.py              # busca de augmentation/hiperparâmetros com Optuna
    ├── ablation.py                    # ablação individual de cada augmentation vs. baseline sem augmentation
    ├── cv_confirm.py                   # confirma a config vencedora via k-fold cross-validation
    └── final_train.sh                   # treino final com a configuração escolhida
```

## Instalação

```bash
python3 -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
```

## Uso básico

```bash
# 1. Gerar splits a partir do dataset bruto (uma pasta por classe)
python scripts/project.py dataset \
    --dataset-path ./dataset_plantas_raw \
    --split-out ./cls/dataset \
    --split 15 15   # 15% val, 15% test, resto (70%) vai pro train

# 2. Treinar um modelo
python scripts/project.py model yolo26m-cls.pt \
    --train --bench \
    --epoch 25 \
    --train-path ./cls/dataset/train \
    --val-path ./cls/dataset/val \
    --test-path ./cls/dataset/test \
    --models-path ./Models/Base \
    --save-models-path ./Models/Trained \
    --result-path ./results

# 3. Ver todas as opções disponíveis
python scripts/project.py model --help
python scripts/project.py dataset --help
```

Cada script em `pipeline/` (exceto os `.sh`) também aceita `--help` e tem um
exemplo de uso completo no docstring no topo do arquivo.

## Antes de rodar (ajustes necessários)

Alguns arquivos ainda têm caminhos absolutos da minha máquina — **edite antes
de rodar no seu ambiente**:

- `scripts/dataset/clean.py` — as variáveis `DIRETORIO_IMAGENS` no topo do
  arquivo (esse script ainda não tem `--dataset-path` via CLI).
- `scripts/pipeline/baseline.sh`, `busca_resolucao.sh`, `final_train.sh`,
  `grid_balanceamento.sh` — a linha `cd /home/andressa/...` no início de cada
  um, e alguns caminhos de `--train-path`/`--val-path`.

Fora isso, todos os outros scripts recebem os caminhos via argumentos de
linha de comando (`--train-path`, `--result-path`, etc.), então funcionam em
qualquer máquina desde que os argumentos apontem pros lugares certos.

## Requisitos de hardware

Desenvolvido e testado em uma GPU com ~3.7GB de VRAM utilizável (notebook,
RTX 2050) — por isso vários scripts usam `--batch 8` fixo em vez do
auto-batch do Ultralytics (que variava demais entre resoluções). Ajuste
`--batch` conforme a VRAM disponível na sua máquina.