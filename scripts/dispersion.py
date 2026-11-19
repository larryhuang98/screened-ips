#!/usr/bin/env python
"""D2 dispersion test (NaCl).

targets : add D2 to a target, either as the full periodic sum (to 30 A) or IPS6-corrected within 5 A.
          usage: dispersion.py targets SRC_DIR OUT_FULL OUT_IPS [--nproc 8]
analyze : size of the dispersion tail beyond 5 A and of the residual after the IPS6 correction.
          usage: dispersion.py analyze DATA_DIR OUT.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from ase import Atoms

from screened_ips.dispersion import d2
from screened_ips.io import SPLITS, labels, load_split, write_dataset


def _clean(a):
    b = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=True)
    b.wrap()
    return b


def _work(a):
    b = _clean(a)
    return d2(b, 30.0), d2(b, ips_rc=5.0)


def targets(a):
    data = {s: load_split(a.src, s) for s in SPLITS}
    with Pool(a.nproc) as p:
        R = {s: p.map(_work, data[s], chunksize=4) for s in SPLITS}
    for out, k in ((a.out_full, 0), (a.out_ips, 1)):
        rows = {s: [(x, labels(x)[0] + r[k][0], labels(x)[1] + r[k][1]) for x, r in zip(data[s], R[s])] for s in SPLITS}
        write_dataset(out, rows)
        print("wrote", out)


def analyze(a):
    fr = load_split(a.data, "test")
    keys = ["full", "cut", "ips"]
    F = {k: [] for k in keys}
    E = {k: [] for k in keys}
    n = []
    for x in fr:
        b = _clean(x)
        for k, (e, f) in zip(keys, (d2(b, 30.0), d2(b, 5.0), d2(b, ips_rc=5.0))):
            F[k].append(f)
            E[k].append(e / len(b))
        n.append(len(b))
    rms = lambda z: float(np.sqrt(np.mean(np.concatenate(z) ** 2)) * 1e3)
    res = dict(F_D2_full_meV_A=rms(F["full"]),
               F_tail_beyond_Rc_meV_A=rms([f - c for f, c in zip(F["full"], F["cut"])]),
               F_after_IPS6_meV_A=rms([f - c for f, c in zip(F["full"], F["ips"])]),
               E_tail_per_atom_sd_meV=float(np.std(np.array(E["full"]) - np.array(E["cut"])) * 1e3),
               E_after_IPS6_per_atom_sd_meV=float(np.std(np.array(E["full"]) - np.array(E["ips"])) * 1e3))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("src"); t.add_argument("out_full"); t.add_argument("out_ips")
    t.add_argument("--nproc", type=int, default=os.cpu_count())
    n = sub.add_parser("analyze")
    n.add_argument("data"); n.add_argument("out")
    a = p.parse_args()
    targets(a) if a.cmd == "targets" else analyze(a)


if __name__ == "__main__":
    main()
