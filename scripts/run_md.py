#!/usr/bin/env python
"""Molecular dynamics of molten NaCl with a (screened) IPS model.

nvt: Langevin, 1200 K, 2 fs, friction 0.01 / fs, Maxwell-Boltzmann start; frames every 25 steps after NEQ.
nve: velocity Verlet, 1 fs, started from a frame (positions and momenta) of an NVT trajectory; positions and
     momenta written every 5 steps (for charge-current spectra), energies every 50 steps.
The model is deployed as E_theta - lam * Delta (lam = 0: model unchanged).

usage:
  run_md.py nvt MODEL OUT --lam 0.5205 --init init100.xyz [--repeat 2] [--seed 1] [--neq 1250] [--nprod 11250]
  run_md.py nve MODEL OUT --lam 0.5205 --start NVT.traj.extxyz@300 [--nprod 8000]
writes OUT.traj.extxyz, OUT.log, OUT.json
"""
import argparse
import json
import time

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from screened_ips.md import ScreenedIPSCalculator


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["nvt", "nve"])
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--init", default=None)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--start", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1200.0)
    p.add_argument("--neq", type=int, default=1250)
    p.add_argument("--nprod", type=int, default=11250)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    from mace.calculators import MACECalculator
    calc = ScreenedIPSCalculator(MACECalculator(model_paths=a.model, device=a.device, default_dtype="float32"), a.lam)
    t0, ok = time.time(), True
    if a.mode == "nvt":
        atoms = read(a.init).repeat((a.repeat,) * 3)
        atoms.calc = calc
        rng = np.random.default_rng(a.seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=rng)
        Stationary(atoms)
        dyn = Langevin(atoms, 2.0 * units.fs, temperature_K=a.temperature, friction=0.01 / units.fs, rng=rng)
        for i in range(a.neq + a.nprod):
            dyn.run(1)
            if i % 50 == 0:
                d = atoms.get_all_distances(mic=True)
                np.fill_diagonal(d, 9)
                ep = atoms.get_potential_energy()
                with open(a.out + ".log", "a") as fh:
                    fh.write(f"{i} {ep:.4f} {atoms.get_temperature():.1f} {d.min():.3f} {time.time() - t0:.0f}\n")
                if d.min() < 1.0 or not np.isfinite(ep):  # failure criterion of the paper
                    ok = False
                    break
            if i >= a.neq and (i - a.neq) % 25 == 0:
                write(a.out + ".traj.extxyz", atoms, append=True)
        steps = i + 1
    else:
        fn, idx = (a.start.split("@") + ["-1"])[:2]
        a0 = read(fn, int(idx))
        atoms = a0.copy()
        atoms.set_momenta(a0.get_momenta())
        atoms.calc = calc
        dyn = VelocityVerlet(atoms, 1.0 * units.fs)
        for i in range(a.nprod):
            dyn.run(1)
            if i % 5 == 0:
                write(a.out + ".traj.extxyz", atoms, append=True)
            if i % 50 == 0:
                ep, ek = atoms.get_potential_energy(), atoms.get_kinetic_energy()
                with open(a.out + ".log", "a") as fh:
                    fh.write(f"{i} {ep:.5f} {ek:.5f} {ep + ek:.5f} {atoms.get_temperature():.1f} {time.time() - t0:.0f}\n")
        steps = a.nprod
    json.dump(dict(ok=ok, steps=steps, lam=a.lam, wall_s=time.time() - t0, timing=calc.timing),
              open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
