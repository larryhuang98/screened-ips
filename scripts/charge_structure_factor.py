#!/usr/bin/env python
"""Charge-charge structure factor S_ZZ(k) from NVT trajectories (formal charges).

All files PREFIX_r*.traj.extxyz are pooled for each prefix.
usage: charge_structure_factor.py OUT.json PREFIX [PREFIX ...] [--kmax 2.0]
"""
import argparse
import glob
import json

import numpy as np
from ase.io import read

from screened_ips.spectra import charge_structure_factor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("prefixes", nargs="+")
    p.add_argument("--kmax", type=float, default=2.0)
    a = p.parse_args()
    out = {}
    for pre in a.prefixes:
        frames = [f for fn in sorted(glob.glob(f"{pre}_r*.traj.extxyz")) for f in read(fn, ":")]
        ks, samples = charge_structure_factor(frames, a.kmax)
        out[pre] = dict(frames=len(frames), k=ks, S=[float(np.mean(s)) for s in samples],
                        sem=[float(np.std(s) / np.sqrt(len(s))) for s in samples], n=[len(s) for s in samples])
        print(pre, len(frames), [(k, round(S, 4)) for k, S in zip(ks[:6], out[pre]["S"][:6])], flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
