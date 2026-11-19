#!/usr/bin/env python
"""Build the NaCl reference splits used in the paper from the published PBE-D3 data.

Source: S. Attarian, C. Shen, D. Morgan, I. Szlufarska, "Best practices for fitting machine learning interatomic
potentials for molten salts: a case study using NaCl-MgCl2", data at figshare doi:10.6084/m9.figshare.26535952.v1.
Download Pot_N.pckl.gzip and Testing.pckl.gzip into RAW_DIR.

The selection (640 train / 201 valid rows of Pot_N, drawn with numpy default_rng(20261001), and the 62 NaCl-only
frames of Testing) is stored in splits/nacl_split.json together with geometry hashes, which are checked.
Energies are shifted by the training mean energy per atom (stored in the split file).

usage: prepare_nacl.py RAW_DIR OUT_DIR
writes OUT_DIR/{train,valid,test}.extxyz with REF_energy (eV) and REF_forces (eV/A).
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from ase import Atoms

from screened_ips.io import write_labelled

HERE = os.path.dirname(os.path.abspath(__file__))


def geometry_sha256(atoms, decimals=8):
    """Ordered fingerprint of species, cell and wrapped fractional coordinates (rounded to 8 decimals)."""
    cell = np.asarray(atoms.cell.array, float)
    frac = np.asarray(atoms.positions, float) @ np.linalg.inv(cell)
    frac %= 1.0
    frac = np.round(frac, decimals)
    frac %= 1.0
    rcell = np.round(cell, decimals)
    frac[frac == 0] = 0.0
    rcell[rcell == 0] = 0.0
    ident = {"numbers": np.asarray(atoms.numbers).tolist(), "cell": rcell.tolist(),
             "pbc": [True, True, True], "fractional": frac.tolist()}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir")
    p.add_argument("out_dir")
    p.add_argument("--split-file", default=os.path.join(HERE, "..", "splits", "nacl_split.json"))
    p.add_argument("--no-hash-check", action="store_true", help="skip the geometry hash check")
    a = p.parse_args()
    sel = json.load(open(a.split_file))
    shift = sel["energy_shift_eV_per_atom"]
    tables = {}
    for name in sorted({f[1] for f in sel["frames"]}):
        path = os.path.join(a.raw_dir, name)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        if md5 != sel["md5"][name]:
            raise SystemExit(f"MD5 mismatch for {name}: {md5}")
        tables[name] = pd.read_pickle(path, compression="gzip")
    rows = {"train": [], "valid": [], "test": []}
    mismatches = 0
    for split, name, row, ghash in sel["frames"]:
        r = tables[name].iloc[row]
        at = r.ase_atoms
        atoms = Atoms(numbers=at.numbers, positions=at.positions, cell=at.cell, pbc=True)
        if not a.no_hash_check and geometry_sha256(atoms) != ghash:
            mismatches += 1
        rows[split].append((atoms, float(r.energy), np.asarray(r.forces, float)))
    for split, rr in rows.items():
        write_labelled(os.path.join(a.out_dir, f"{split}.extxyz"), rr, shift)
        print(split, len(rr))
    if mismatches:
        raise SystemExit(f"{mismatches} geometry hashes differ from the split file")


if __name__ == "__main__":
    main()
