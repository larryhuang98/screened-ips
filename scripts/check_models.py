#!/usr/bin/env python
"""Evaluate the shipped NaCl models, deployed as E_theta - lambda * Delta, against reference labels.

usage: check_models.py FRAMES.extxyz [--models all|name,name] [--threads 4]
FRAMES: the NaCl test set written by prepare_nacl.py (data/nacl/ref/test.extxyz, 62 frames), or a single
labelled configuration such as examples/init100.xyz. Reports the force RMSE (meV/A) over all components and the
energy RMSE (meV/atom) after removing the mean per-atom offset (needs more than one frame).
"""
import argparse
import os

import numpy as np
from ase.io import read

from screened_ips.md import ScreenedIPSCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {  # name: (file, lambda, R_c)
    "plain_r5": ("nacl_plain_mace_r5.model", 0.0, 5.0),
    "screened_ips_0.52_r5": ("nacl_screened_ips_lam0.52_r5.model", 0.5205, 5.0),
    "screened_ips_0.44_r5": ("nacl_screened_ips_lam0.44_r5.model", 0.44, 5.0),
    "plain_r5_zbl": ("nacl_plain_mace_r5_zbl.model", 0.0, 5.0),
    "screened_ips_0.52_r5_zbl": ("nacl_screened_ips_lam0.52_r5_zbl.model", 0.5205, 5.0),
    "plain_r6": ("nacl_plain_mace_r6_5k.model", 0.0, 6.0),
    "screened_ips_0.52_r6": ("nacl_screened_ips_lam0.52_r6_5k.model", 0.5205, 6.0),
    "plain_r7": ("nacl_plain_mace_r7_5k.model", 0.0, 7.0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames")
    p.add_argument("--models", default="all")
    p.add_argument("--model-dir", default=os.path.join(HERE, "..", "models"))
    a = p.parse_args()
    from mace.calculators import MACECalculator
    frames = read(a.frames, ":")
    names = list(MODELS) if a.models == "all" else a.models.split(",")
    print(f"{len(frames)} frame(s)")
    print(f"{'model':28s} {'lambda':>7s} {'F RMSE (meV/A)':>15s} {'E RMSE (meV/atom)':>18s}")
    for name in names:
        fn, lam, rc = MODELS[name]
        base = MACECalculator(model_paths=os.path.join(a.model_dir, fn), device="cpu", default_dtype="float32")
        calc = ScreenedIPSCalculator(base, lam, rc)
        dF, dE = [], []
        for at in frames:
            x = at.copy()
            x.calc = calc
            dF.append(x.get_forces() - at.arrays["REF_forces"])
            dE.append((x.get_potential_energy() - at.info["REF_energy"]) / len(at))
        dE = np.array(dE) - np.mean(dE)
        f_rmse = 1e3 * np.sqrt(np.mean(np.concatenate(dF) ** 2))
        e_rmse = f"{1e3 * np.sqrt(np.mean(dE ** 2)):18.2f}" if len(frames) > 1 else f"{'-':>18s}"
        print(f"{name:28s} {lam:7.4f} {f_rmse:15.2f} {e_rmse}", flush=True)


if __name__ == "__main__":
    main()
