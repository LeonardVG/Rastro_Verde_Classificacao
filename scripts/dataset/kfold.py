"""
Gera k folds estratificados a partir de train+val já existentes.
O test nunca entra aqui.

Correção de vazamento: o train/ contém cópias geradas pelo aug.py
(arquivos "<original>_augNNN.ext"). Antes, o StratifiedKFold sorteava
cada arquivo individualmente, então uma foto podia ir para o train do
fold e as cópias dela (ou o contrário) para o val do fold. O val ficava
"fácil" e o CV saía alto e com desvio padrão artificialmente baixo.

Agora:
- os folds são sorteados só entre as imagens ORIGINAIS;
- cada cópia _aug acompanha a sua original no fold-train;
- o fold-val contém só originais (as cópias das originais que caíram no
  val são descartadas), igual ao val/test reais.

A interface gen(train_path, val_path, output, k, seed) não mudou,
então o cv_confirm.py continua funcionando sem alteração.
"""
import os
import re
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

try:
    from .cls import CLS, Entry
except ImportError:
    from cls import CLS, Entry

# mesmo padrão de nome do aug.py: f"{src.stem}_aug{i:03d}{src.suffix}"
AUG_RE = re.compile(r"_aug\d+$")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _is_aug(path: str) -> bool:
    return AUG_RE.search(_stem(path)) is not None


def _group_key(path: str, label: str) -> str:
    """Chave da imagem original: classe + nome sem o sufixo _augNNN."""
    return f"{label}/{AUG_RE.sub('', _stem(path))}"


def _populate(root: str, split: str, items: List[Tuple[str, str]]) -> None:
    split_root = os.path.join(root, split)
    for path, label in items:
        droot = os.path.join(split_root, label)
        os.makedirs(droot, exist_ok=True)
        dst = os.path.join(droot, os.path.basename(path))
        if not os.path.exists(dst):
            shutil.copyfile(path, dst)


def gen(train_path: str, val_path: str, output: str, k: int, seed: int = 42) -> List[str]:
    entries: List[Entry] = CLS.get_entries(train_path) + CLS.get_entries(val_path)

    originals = [e for e in entries if not _is_aug(e.path)]
    augs = [e for e in entries if _is_aug(e.path)]
    keys = [_group_key(e.path, e.label) for e in originals]

    # originais com o mesmo nome dentro da mesma classe (ex.: um em train/ e outro em val/)
    dups = [key for key, c in Counter(keys).items() if c > 1]
    if dups:
        print(f"[kfold] AVISO: {len(dups)} nome(s) de original repetido(s) na mesma classe, "
              f"ex.: {dups[:3]} -- as cópias _aug desses nomes vão para todos eles")

    aug_by_key: Dict[str, List[Entry]] = defaultdict(list)
    for e in augs:
        aug_by_key[_group_key(e.path, e.label)].append(e)

    orphans = set(aug_by_key) - set(keys)
    if orphans:
        n_orph = sum(len(aug_by_key[o]) for o in orphans)
        print(f"[kfold] AVISO: {n_orph} imagem(ns) _aug sem a original no pool "
              f"(ex.: {sorted(orphans)[:3]}) -- descartadas")

    print(f"[kfold] pool (train+val): {len(originals)} originais + {len(augs)} _aug")
    for label, n in sorted(Counter(e.label for e in originals).items()):
        n_aug = sum(len(v) for kk, v in aug_by_key.items() if kk.startswith(f"{label}/"))
        print(f"    {label:30s} {n:4d} originais  {n_aug:4d} _aug")
    print(f"[kfold] gerando {k} folds estratificados (sorteio só entre originais)")

    labels = [e.label for e in originals]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    shutil.rmtree(output, ignore_errors=True)
    os.makedirs(output, exist_ok=True)

    fold_roots = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(originals)), labels)):
        fold_root = os.path.join(output, f"fold_{fold_idx}")

        train_items: List[Tuple[str, str]] = []
        for i in train_idx:
            e = originals[i]
            train_items.append((e.path, e.label))
            train_items.extend((a.path, a.label) for a in aug_by_key.get(keys[i], []))

        val_items = [(originals[i].path, originals[i].label) for i in val_idx]
        n_aug_dropped = sum(len(aug_by_key.get(keys[i], [])) for i in val_idx)

        # checagem de sanidade: nenhuma original do val pode ter "parente" no train
        train_keys = {keys[i] for i in train_idx}
        leaked = train_keys & {keys[i] for i in val_idx}
        if leaked:
            print(f"[kfold] AVISO fold_{fold_idx}: {len(leaked)} grupo(s) em train e val ao mesmo tempo "
                  f"(nomes repetidos), ex.: {sorted(leaked)[:3]}")

        _populate(fold_root, "train", train_items)
        _populate(fold_root, "val", val_items)

        n_train_aug = len(train_items) - len(train_idx)
        print(f"[kfold] fold_{fold_idx}: train = {len(train_idx)} originais + {n_train_aug} _aug | "
              f"val = {len(val_items)} originais | {n_aug_dropped} _aug descartadas")
        fold_roots.append(fold_root)

    return fold_roots