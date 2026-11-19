#!/usr/bin/env python
"""Held-out errors of trained models on their own targets.

Force RMSE (meV/A) over all components and energy RMSE (meV/atom) after removing the mean per-atom offset.
For a model trained on T_lambda the own-target error equals the error of the deployed model E_theta - lambda*Delta
against the reference, because Delta is added back exactly.

usage: evaluate.py OUT.json RUN_DIR [RUN_DIR ...] [--splits valid,test] [--per-config OUT.npz]
RUN_DIR contains data/{split}.extxyz and models/m.model (MACE) or nq.yaml + models/m.model (NequIP).
--per-config writes per-configuration squared force errors for paired bootstrap confidence intervals.
"""
import argparse
import json
import os

import numpy as np
from ase.io import read


def calculator(run):
    if os.path.exists(f"{run}/nq.yaml"):
        from nequip.ase import NequIPCalculator
        return NequIPCalculator.from_deployed_model(f"{run}/models/m.model", device="cpu",
                                                    species_to_type_name={"Na": "Na", "Cl": "Cl"})
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=f"{run}/models/m.model", device="cpu", default_dtype="float32")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--splits", default="valid,test")
    p.add_argument("--per-config", default=None)
    a = p.parse_args()
    rms = lambda z: float(np.sqrt(np.mean(z ** 2)))
    out, pc = {}, {}
    for r in a.runs:
        c = calculator(r)
        out[r] = {}
        for s in a.splits.split(","):
            fr = read(f"{r}/data/{s}.extxyz", ":")
            E, F, Er, Fr, n, sse = [], [], [], [], [], []
            for at in fr:
                x = at.copy()
                x.calc = c
                E.append(x.get_potential_energy())
                F.append(x.get_forces())
                Er.append(at.info["REF_energy"])
                Fr.append(at.arrays["REF_forces"])
                n.append(len(at))
                sse.append(float(((F[-1] - Fr[-1]) ** 2).sum()))
            dE = (np.array(E) - np.array(Er)) / np.array(n)
            dE -= dE.mean()
            out[r][s] = dict(F=1e3 * rms(np.concatenate(F) - np.concatenate(Fr)), E=1e3 * rms(dE))
            if s == "test":
                k = r.replace("/", "__")
                pc[k + "__sse"], pc[k + "__nF"] = np.array(sse), 3 * np.array(n)
        print(r, out[r], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    if a.per_config:
        np.savez(a.per_config, **pc)


if __name__ == "__main__":
    main()
