#!/usr/bin/env python
"""Committee force spread along a trajectory driven by the first of several models trained on the same target.

With --lam 0 the driving model is the IPS Hamiltonian (E_theta alone); with --lam > 0 the screened IPS model
(E_theta - lam*Delta). Every 10 steps the standard deviation of the forces predicted by all models is recorded.

usage: committee.py OUT --models m1.model,m2.model,m3.model --init init100.xyz [--lam 0] [--seed 1] [--steps 2500]
"""
import argparse
import json

import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from screened_ips.md import ScreenedIPSCalculator


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--models", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=2500)
    a = p.parse_args()
    from mace.calculators import MACECalculator
    M = [MACECalculator(model_paths=m, device="cpu", default_dtype="float32") for m in a.models.split(",")]
    atoms = read(a.init)
    atoms.calc = ScreenedIPSCalculator(M[0], a.lam)
    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=1200, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, 2.0 * units.fs, temperature_K=1200, friction=0.01 / units.fs, rng=rng)
    rows = []
    for i in range(a.steps):
        if i % 10 == 0:
            F = []
            for m in M:
                x = atoms.copy()
                x.calc = m
                F.append(x.get_forces())
            spread = float(np.sqrt(np.mean(np.array(F).var(axis=0, ddof=1))))
            d = atoms.get_all_distances(mic=True)
            np.fill_diagonal(d, 9)
            same = np.equal.outer(atoms.numbers, atoms.numbers)
            rows.append(dict(step=i, epot=float(atoms.get_potential_energy()), spread=spread, dmin=float(d.min()),
                             dmin_like=float(d[same].min()), dmin_unlike=float(d[~same].min())))
            if not np.isfinite(rows[-1]["epot"]) or d.min() < 1.0:
                break
        dyn.run(1)
    json.dump(rows, open(a.out + ".json", "w"))
    print(a.out, len(rows), rows[-1])


if __name__ == "__main__":
    main()
