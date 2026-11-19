#!/usr/bin/env python
"""Reference splits for liquid water (revPBE0-D3, 64 molecules).

Source: B. Cheng et al., PNAS 116, 1110 (2019); training-set/dataset_1593.xyz of
https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water (SHA-256 5e2ec3b4...dfdbc).
Units are converted from Bohr/Hartree to Angstrom/eV. Contiguous split in source order:
frames 0-1194 train, 1195-1393 valid, 1394-1592 test.

usage: prepare_water.py dataset_1593.xyz OUT_DIR
Then: make_targets.py OUT_DIR WATER_TARGETS --charges tip3p --lam 0,0.25,0.5,0.75,1.0,1.25,delta
"""
import argparse
import hashlib

import numpy as np
from ase import units
from ase.io import read

from screened_ips.io import write_dataset

SHA256 = "5e2ec3b4b1334e012daa6a2e2dbd138108c41c7b6f692ff89b8747b81cadfdbc"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    a = p.parse_args()
    if hashlib.sha256(open(a.src, "rb").read()).hexdigest() != SHA256:
        print("warning: source checksum differs from the file used in the paper")
    frames = read(a.src, ":")
    rows = []
    for x in frames:
        e = float(x.info["TotEnergy"]) * units.Hartree
        fh = x.arrays["force"] if "force" in x.arrays else x.get_forces()
        f = np.array(fh, float) * units.Hartree / units.Bohr
        x.calc = None
        x.set_cell(x.cell.array * units.Bohr, scale_atoms=False)
        x.positions = x.positions * units.Bohr
        x.pbc = True
        rows.append((x, e, f))
    write_dataset(a.out, {"train": rows[:1195], "valid": rows[1195:1394], "test": rows[1394:1593]})


if __name__ == "__main__":
    main()
