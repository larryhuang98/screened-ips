#!/usr/bin/env python
"""Reference splits for SuperSalt salts and mixtures.

Source: SuperSalt data set (Zenodo doi:10.5281/zenodo.15734798, SuperSalt.zip). Extract Training.extxyz and
Validation.extxyz from the archive into RAW_DIR.

Frames are selected by composition (cations given; Cl always present). Split: 640 train and 150 valid frames
drawn with numpy default_rng(SEED) from the published training split, test = all frames of that composition in
the published validation split. Seeds used in the paper: 20261030 (CsCl, BaCl2, SrCl2), 20261103 (mixtures).

usage: prepare_supersalt.py RAW_DIR OUT_DIR --cations Cs [--seed 20261030]
       prepare_supersalt.py RAW_DIR OUT_DIR --cations Li,Na,K,Rb,Cs,Mg,Ca,Sr,Ba,Zn,Zr --seed 20261103
       prepare_supersalt.py RAW_DIR OUT_DIR --cations Mg,Na --seed 20261103
"""
import argparse
import os

import numpy as np
from ase.io import read

from screened_ips.io import write_dataset


def energy_forces(a):
    e = a.info["energy"] if "energy" in a.info else a.get_potential_energy()
    f = a.arrays["force"] if "force" in a.arrays else a.get_forces()
    return float(e), np.asarray(f, float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw")
    p.add_argument("out")
    p.add_argument("--cations", required=True)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--ntrain", type=int, default=640)
    p.add_argument("--nvalid", type=int, default=150)
    a = p.parse_args()
    key = set(a.cations.split(","))
    sel = lambda fn: [x for x in read(os.path.join(a.raw, fn), ":") if set(x.get_chemical_symbols()) - {"Cl"} == key]
    tr, te = sel("Training.extxyz"), sel("Validation.extxyz")
    perm = np.random.default_rng(a.seed).permutation(len(tr))
    S = {"train": [tr[i] for i in perm[:a.ntrain]], "valid": [tr[i] for i in perm[a.ntrain:a.ntrain + a.nvalid]],
         "test": te}
    for s, fr in S.items():
        for x in fr:
            h = 1.0 / np.linalg.norm(np.linalg.inv(x.cell.array), axis=0)
            if h.min() <= 10.0:
                raise SystemExit("cell height below 2 R_c")
    write_dataset(a.out, {s: [(x, *energy_forces(x)) for x in fr] for s, fr in S.items()})
    print({s: len(v) for s, v in S.items()})


if __name__ == "__main__":
    main()
