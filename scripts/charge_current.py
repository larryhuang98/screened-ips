#!/usr/bin/env python
"""Longitudinal and transverse charge-current spectra of NVE trajectories (positions + momenta every 5 fs).

usage: charge_current.py OUT.json TRAJ [TRAJ ...] [--dt 0.005]
"""
import argparse
import json

import numpy as np
from ase.io import iread

from screened_ips.coulomb import charges_for
from screened_ips.spectra import charge_current_spectra


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("trajs", nargs="+")
    p.add_argument("--dt", type=float, default=5e-3, help="time between frames (ps)")
    a = p.parse_args()
    res = {}
    for tr in a.trajs:
        R, V = [], []
        for at in iread(tr):
            R.append(at.get_positions())
            V.append(at.get_momenta() / at.get_masses()[:, None])
            box, q = at.cell.array.diagonal(), charges_for(at)
        sp = charge_current_spectra(np.array(R), np.array(V), q, box, a.dt)
        res[tr] = {str(n): {"k": v["k"], "f": v["f"].tolist(), "SL": v["SL"].tolist(), "ST": v["ST"].tolist(),
                            "fpeakL": float(v["f"][1:][np.argmax(v["SL"][1:])]),
                            "fpeakT": float(v["f"][1:][np.argmax(v["ST"][1:])])} for n, v in sp.items()}
        print(tr, {n: (round(v["k"], 3), round(v["fpeakL"], 2)) for n, v in res[tr].items()}, flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
