#!/usr/bin/env python
"""Teacher experiment: label sampled configurations with a teacher whose exterior is a known screened correction.

Teacher energy E_T = E_theta(R) - lam * Delta(R) with a screened IPS model (lam = 0.5205), so that its long-range
part is the screened Ewald exterior by construction. Student targets: L0 = E_T, L<lam> = E_T + lam * Delta and
Ldelta = Delta. Sample configurations first, e.g. with run_md.py nvt using a plain MACE model with ZBL.

usage: make_teacher_targets.py TEACHER_MODEL OUT_ROOT --train-trajs a.extxyz,b.extxyz --test-traj c.extxyz
                               [--lam 0.5205] [--stride 2] [--ntrain 300] [--nvalid 50] [--ntest 100]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from ase.io import read

from screened_ips.coulomb import charges_for, delta
from screened_ips.io import write_dataset


def _dl(a):
    return delta(a, charges_for(a))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("teacher")
    p.add_argument("out")
    p.add_argument("--train-trajs", required=True)
    p.add_argument("--test-traj", required=True)
    p.add_argument("--lam", type=float, default=0.5205)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--ntrain", type=int, default=300)
    p.add_argument("--nvalid", type=int, default=50)
    p.add_argument("--ntest", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--nproc", type=int, default=os.cpu_count())
    a = p.parse_args()
    from mace.calculators import MACECalculator
    trv = [f for t in a.train_trajs.split(",") for f in read(t, ":")[::a.stride]]
    te = read(a.test_traj, ":")[::a.test_stride][:a.ntest]
    idx = np.random.default_rng(a.seed).permutation(len(trv))
    S = {"train": [trv[i] for i in idx[:a.ntrain]], "valid": [trv[i] for i in idx[a.ntrain:a.ntrain + a.nvalid]],
         "test": te}
    calc = MACECalculator(model_paths=a.teacher, device="cpu", default_dtype="float64")
    lab = {}
    for s, frames in S.items():
        with Pool(a.nproc) as pool:
            D = pool.map(_dl, frames, chunksize=2)
        rows = []
        for x, (de, df) in zip(frames, D):
            y = x.copy()
            y.calc = calc
            rows.append((x, y.get_potential_energy() - a.lam * de, y.get_forces() - a.lam * df, de, df))
        lab[s] = rows
        print(s, len(rows), flush=True)
    for tag, fn in [("L0", lambda r: (r[1], r[2])), (f"L{a.lam:g}", lambda r: (r[1] + a.lam * r[3], r[2] + a.lam * r[4])),
                    ("Ldelta", lambda r: (r[3], r[4]))]:
        write_dataset(os.path.join(a.out, tag), {s: [(r[0], *fn(r)) for r in rows] for s, rows in lab.items()})


if __name__ == "__main__":
    main()
