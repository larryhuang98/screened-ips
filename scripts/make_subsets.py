#!/usr/bin/env python
"""Nested training subsets for learning curves (the same permutation for every target).

usage: make_subsets.py TARGET_DIR OUT_ROOT [--sizes 80,160,320,640] [--seed 20261029]
writes OUT_ROOT/n<size>/{train,valid,test}.extxyz (valid and test copied unchanged)
"""
import argparse
import os
import shutil

import numpy as np
from ase.io import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--sizes", default="80,160,320,640")
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args()
    tr = read(f"{a.src}/train.extxyz", ":")
    perm = np.random.default_rng(a.seed).permutation(len(tr))
    for n in map(int, a.sizes.split(",")):
        d = os.path.join(a.out, f"n{n}")
        os.makedirs(d, exist_ok=True)
        write(f"{d}/train.extxyz", [tr[i] for i in sorted(perm[:n])])
        for s in ("valid", "test"):
            shutil.copy(f"{a.src}/{s}.extxyz", f"{d}/{s}.extxyz")
        print(d)


if __name__ == "__main__":
    main()
