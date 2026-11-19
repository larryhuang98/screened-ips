#!/usr/bin/env python
"""Timing of a model force call and of the exterior correction for 100- and 800-atom NaCl cells (CPU).

model : median wall time of ten MACE (or MACE-LES) force calls in single precision.
delta : PyTorch evaluation of Delta and its forces in double precision: IPS and Ewald real-space pair terms over
        the R_c neighbour list plus the reciprocal sum of a Gaussian-split Ewald summation (sigma = 1 A), using the
        Ewald module of the LES library with fixed formal charges; forces by automatic differentiation. Accuracy is
        checked against the converged NumPy implementation (screened_ips.coulomb.delta). The grid spacing dl sets
        the reciprocal cutoff 2 pi / dl.

usage: benchmark.py model MODEL CONFIG.extxyz OUT.json
       benchmark.py delta CONFIG.extxyz OUT.json [--dl 1.0,1.5,2.0]
CONFIG.extxyz: any file whose first 100-atom frame is used (replicated 2x2x2 for 800 atoms).
"""
import argparse
import json
import math
import os
import time

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

from screened_ips.coulomb import COULOMB_EV_ANGSTROM as KE
from screened_ips.coulomb import charges_for, delta, ips_polynomial


def cells(config):
    fr = [a for a in read(config, ":") if len(a) == 100][0]
    a1 = fr.copy()
    a8 = fr.repeat((2, 2, 2))
    for a in (a1, a8):
        a.wrap()
    return [(100, a1), (800, a8)]


def bench_model(a):
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=a.model, device="cpu", default_dtype="float32")
    res = {"model": os.path.basename(a.model), "threads": os.environ.get("OMP_NUM_THREADS")}
    for n, at in cells(a.config):
        at.calc = calc
        for _ in range(3):
            at.rattle(1e-3); at.get_forces()
        t = []
        for _ in range(10):
            at.rattle(1e-3)
            t0 = time.perf_counter(); at.get_forces(); t.append(time.perf_counter() - t0)
        res[f"model_{n}"] = float(np.median(t))
        print(n, res[f"model_{n}"], flush=True)
    return res


def bench_delta(a, rc=5.0, sigma=1.0):
    import torch
    from les.module.ewald import Ewald
    torch.set_default_dtype(torch.float64)

    def delta_torch(at, q, ew, nl):
        i, j, S = nl
        r = torch.tensor(at.positions, requires_grad=True)
        cell = torch.tensor(np.array(at.cell))
        qt = torch.tensor(q)
        d = r[j] - r[i] + torch.tensor(S, dtype=torch.float64) @ cell
        rr = torch.linalg.norm(d, dim=1)
        x = rr / rc
        poly = (-35 + 35 * x ** 2 - 21 * x ** 4 + 5 * x ** 6) / 16
        # IPS pair 1/r + P/R_c minus Ewald real-space pair erfc(r/(sqrt2 sigma))/r
        pair = 0.5 * KE * torch.sum(qt[i] * qt[j] * (torch.erf(rr / (math.sqrt(2) * sigma)) / rr + poly / rc))
        e = pair + KE * ips_polynomial(0.0) / (2 * rc) * torch.sum(qt ** 2) - ew(qt, r, cell.unsqueeze(0))[0].sum()
        return float(e), (-torch.autograd.grad(e, r)[0]).numpy()

    res = {"threads": os.environ.get("OMP_NUM_THREADS")}
    ref = {}
    for dl in map(float, a.dl.split(",")):
        ew = Ewald(dl=dl, sigma=sigma, remove_self_interaction=True)
        for n, at in cells(a.config):
            q = charges_for(at)
            nl = neighbor_list("ijS", at, rc)
            e, f = delta_torch(at, q, ew, nl)
            if n not in ref:
                ref[n] = delta(at, q, rc)
            e0, f0 = ref[n]
            t, tn = [], []
            for _ in range(5):
                t0 = time.perf_counter(); nl = neighbor_list("ijS", at, rc); t1 = time.perf_counter()
                delta_torch(at, q, ew, nl); t.append(time.perf_counter() - t1); tn.append(t1 - t0)
            res[f"dl{dl}_n{n}"] = dict(time=float(np.median(t)), time_neighbour_list=float(np.median(tn)),
                                       dE_per_atom=abs(e - e0) / n, dF_rms=float(np.sqrt(np.mean((f - f0) ** 2))))
            print(dl, n, res[f"dl{dl}_n{n}"], flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model"); m.add_argument("model"); m.add_argument("config"); m.add_argument("out")
    d = sub.add_parser("delta"); d.add_argument("config"); d.add_argument("out"); d.add_argument("--dl", default="1.0,1.5,2.0")
    a = p.parse_args()
    res = bench_model(a) if a.cmd == "model" else bench_delta(a)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
