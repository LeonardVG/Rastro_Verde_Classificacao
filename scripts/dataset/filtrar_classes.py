
"""
Cria um novo dataset contendo apenas as classes selecionadas de um dataset original.

Exemplo:

python3 ./scripts/dataset/filtrar_classes.py \
    --dataset-root ./cls/dataset_inteiro \
    --output-root ./cls/dataset_Outros \
    --classes Eugenia_brasiliensis Pleroma_mutabile Libidibia_ferrea Eugenia_pyriformis Cariniana_legalis Campomanesia_phaea 

    
Comportamento:
- Preserva o dataset original.
- Copia somente as classes especificadas.
- Preserva train/, val/ e test/.
- Preserva exatamente os nomes das classes.
- Copia somente .jpg, .jpeg e .png.
- Se a pasta de saída já existir, substitui somente as classes solicitadas (D1).
- Se alguma classe solicitada não existir em algum split, aborta antes de copiar.
"""

import argparse
import shutil
from pathlib import Path


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um dataset contendo apenas as classes selecionadas."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Diretório raiz do dataset original.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Diretório onde o novo dataset será criado.",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Classes que devem ser copiadas.",
    )

    return parser.parse_args()


def validar_dataset(dataset_root: Path, classes: list[str]) -> None:
    """Valida a estrutura e verifica se todas as classes existem em todos os splits."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset original não encontrado: {dataset_root}"
        )

    faltantes = []

    for split in SPLITS:
        split_dir = dataset_root / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split não encontrado: {split_dir}"
            )

        for classe in classes:
            class_dir = split_dir / classe

            if not class_dir.is_dir():
                faltantes.append(f"{split}/{classe}")

    if faltantes:
        mensagem = "\n".join(f"  - {item}" for item in faltantes)
        raise FileNotFoundError(
            "As seguintes classes não foram encontradas:\n" + mensagem
        )


def copiar_classe(
    dataset_root: Path,
    output_root: Path,
    split: str,
    classe: str,
) -> int:
    """Substitui a classe no destino e copia somente imagens permitidas."""
    origem = dataset_root / split / classe
    destino = output_root / split / classe

    # D1: substitui somente a classe solicitada.
    if destino.exists():
        shutil.rmtree(destino)

    destino.mkdir(parents=True, exist_ok=True)

    quantidade = 0

    for arquivo in origem.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_IMAGEM:
            shutil.copy2(arquivo, destino / arquivo.name)
            quantidade += 1

    return quantidade


def main() -> None:
    args = get_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    # Remove classes duplicadas mantendo a ordem informada.
    classes = list(dict.fromkeys(args.classes))

    # Valida tudo antes de modificar a saída.
    validar_dataset(dataset_root, classes)

    print("=" * 60)
    print("FILTRAGEM DE DATASET")
    print("=" * 60)
    print(f"Origem : {dataset_root}")
    print(f"Destino: {output_root}")
    print(f"Classes: {', '.join(classes)}")
    print()

    total = 0

    for split in SPLITS:
        print(f"[{split}]")

        for classe in classes:
            quantidade = copiar_classe(
                dataset_root,
                output_root,
                split,
                classe,
            )

            total += quantidade

            print(f"  {classe}: {quantidade} imagens copiadas")

        print()

    print(f"Total de imagens copiadas: {total}")
    print(f"Dataset criado/atualizado em: {output_root}")


if __name__ == "__main__":
    main()
