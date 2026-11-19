#!/usr/bin/env python
"""Write IPS training targets for a reference data set.

  target   (default)  T_lambda = E_ref + lambda * Delta,  Delta = E_kind(q; R_c) - E_Ewald(q)
  baseline            E_ref - lambda * E_IPS(q; R_c)      (IPS used as an analytical baseline; add back at prediction)
  local               E_ref + lambda * E_IPS(q; R_c)      (local control: only the in-cutoff IPS energy)
  'delta' as a strength writes Delta alone (the target of the Delta-only models used for the lambda estimate).

Energies are re-centred by the training mean energy per atom. Charges: formal ionic charges (default) or TIP3P.

usage: make_targets.py SRC_DIR OUT_ROOT --lam 0,0.3,0.5205,delta [--kind ips|sf|dsf|ewald] [--rc 5]
                       [--form target|baseline|local] [--charges formal|tip3p] [--permuted] [--nproc 8]
writes OUT_ROOT/L<lam>/{train,valid,test}.extxyz
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from ase import Atoms

from screened_ips.coulomb import TIP3P_CHARGES, charges_for, delta, ips_coulomb
from screened_ips.io import SPLITS, labels, load_split, write_dataset


def correction(atoms, kind, rc, table, form, perm_seed=None):
    q = charges_for(atoms, table)
    if perm_seed is not None:  # species-permuted control: same kernel, charges shuffled among atoms
        rng = np.random.default_rng(perm_seed)
        q = rng.permutation(q)
    if form == "target":
        return delta(atoms, q, rc, kind)
    b = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=True)
    b.wrap()
    return ips_coulomb(b, q, rc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--lam", required=True)
    p.add_argument("--kind", default="ips", choices=["ips", "sf", "dsf", "ewald"])
    p.add_argument("--rc", type=float, default=5.0)
    p.add_argument("--form", default="target", choices=["target", "baseline", "local"])
    p.add_argument("--charges", default="formal", choices=["formal", "tip3p"])
    p.add_argument("--permuted", action="store_true",
                   help="permute charges among atoms (control); Delta is rescaled to the physical force RMS on train")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--nproc", type=int, default=os.cpu_count())
    a = p.parse_args()
    table = TIP3P_CHARGES if a.charges == "tip3p" else None
    data = {s: load_split(a.src, s) for s in SPLITS}
    corr = {}
    with Pool(a.nproc) as pool:
        for s in SPLITS:
            seeds = [None] * len(data[s])
            if a.permuted:
                seeds = list(np.random.default_rng(a.seed + SPLITS.index(s)).integers(0, 2**31, len(data[s])))
            fn = partial(_job, kind=a.kind, rc=a.rc, table=table, form=a.form)
            corr[s] = pool.starmap(fn, zip(data[s], seeds), chunksize=4)
    if a.permuted:
        with Pool(a.nproc) as pool:
            phys = pool.map(partial(_job_phys, kind=a.kind, rc=a.rc, table=table), data["train"], chunksize=4)
        scale = np.sqrt(np.mean(np.concatenate([f for _, f in phys]) ** 2)
                        / np.mean(np.concatenate([f for _, f in corr["train"]]) ** 2))
        corr = {s: [(scale * e, scale * f) for e, f in c] for s, c in corr.items()}
        print("permuted-charge scale", scale)
    for s in SPLITS:
        f = np.concatenate([c[1] for c in corr[s]])
        print(s, len(data[s]), "correction force RMS (eV/A)", float(np.sqrt(np.mean(f ** 2))))
    sign = {"target": 1.0, "baseline": -1.0, "local": 1.0}[a.form]
    for lam in a.lam.split(","):
        rows = {}
        for s in SPLITS:
            rr = []
            for atoms, (de, df) in zip(data[s], corr[s]):
                e, f = labels(atoms)
                if lam == "delta":
                    rr.append((atoms, de, df))
                else:
                    l = sign * float(lam)
                    rr.append((atoms, e + l * de, f + l * df))
            rows[s] = rr
        out = os.path.join(a.out, f"L{lam}")
        write_dataset(out, rows)
        print("wrote", out)


def _job(atoms, seed, kind, rc, table, form):
    return correction(atoms, kind, rc, table, form, seed)


def _job_phys(atoms, kind, rc, table):
    return correction(atoms, kind, rc, table, "target")


if __name__ == "__main__":
    main()
