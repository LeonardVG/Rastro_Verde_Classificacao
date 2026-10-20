from typing import *
from dataclasses import dataclass
import os
import random
import math
import collections
import shutil


@dataclass
class Entry:
    path: str
    label: str


class CLS:
    @staticmethod
    def get_entries(dataset: str, ) -> List[Entry]:
        entries = []

        for root, _, files in os.walk(dataset):
            for file in files:
                label = root.rsplit(os.path.sep, maxsplit=1)[1]
                entries.append(Entry(os.path.join(root, file), label))

        return entries

    @staticmethod
    def populate(per: float, total: int, entries: Dict[str, list], por: Dict[str, float]) -> Tuple[Dict[str, List[Entry]], Dict[str, int], int]:
        res = {}
        ftotal = math.ceil(per / 100 * total)
        remaining = ftotal

        amounts = {}
        keys = list(entries.keys())
        for k in keys[:-1]:
            amounts[k] = math.ceil(por[k] * ftotal)
            remaining -= amounts[k]

        amounts[keys[-1]] = remaining

        for k, v in entries.items():
            if amounts[k] > len(v):
                amounts[k] = len(v)
            choice = random.sample(v, amounts[k])
            for c in choice:
                entries[k].remove(c)
            res[k] = [Entry(i, k) for i in choice]

        return res, amounts, ftotal

    @staticmethod
    def print_split(name: str, amount: Dict[str, int], ens: Dict[str, List[Entry]], total: int):
        list = [item for sublist in ens.values() for item in sublist]
        print(f"[{name}]")
        print(f"{len(list)}/{total} ({len(list)/total})")
        for k, v in amount.items():
            print(f"\t{v}/{len(list)} ({v/len(list)})")
        print()

    @staticmethod
    def gen(dataset: str, output: str, val: float, test: float):
        entries = CLS.get_entries(dataset)
        total = len(entries)

        count = DefaultDict(int)
        entries_kv = DefaultDict(list)
        for entry in entries:
            count[entry.label] += 1
            entries_kv[entry.label].append(entry.path)

        por = {}
        print("Por:")
        for k, v in count.items():
            random.shuffle(entries_kv[k])
            por[k] = v / len(entries)
            print(f"\t{k} = {v}/{len(entries)} ({por[k]})")

        rest = len(entries)
        val_list, val_amount, val_total = CLS.populate(
            val, len(entries), entries_kv, por)
        rest -= val_total
        test_list, test_amount, test_total = CLS.populate(
            test, len(entries), entries_kv, por)
        rest -= test_total
        train_list, train_amount, train_total = CLS.populate(
            100, rest, entries_kv, por)

        CLS.print_split("val", val_amount, val_list, total)
        CLS.print_split("test", test_amount, test_list, total)
        CLS.print_split("train", train_amount, train_list, total)

        shutil.rmtree(output, ignore_errors=True)
        os.makedirs(output, exist_ok=True)
        CLS.populate_out(output, "val", val_list, val_total)
        CLS.populate_out(output, "test", test_list, test_total)
        CLS.populate_out(output, "train", train_list, train_total)

    def populate_out(out: str, ds: str, ens: Dict[str, List[Entry]], total: int):
        root = os.path.join(out, ds)
        i = 0

        os.makedirs(root, exist_ok=True)
        for k in ens.keys():
            droot = os.path.join(root, k)
            os.makedirs(droot, exist_ok=True)
            for v in ens[k]:
                i += 1
                print(f"{i}/{total}")
                pa = os.path.join(droot, os.path.basename(v.path))
                if not os.path.exists(pa):
                    shutil.copyfile(v.path, pa)
