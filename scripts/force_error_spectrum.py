#!/usr/bin/env python
"""k-space spectrum of held-out force errors of trained models on their own targets.

For models whose target differs from the reference (lambda > 0), the error of the IPS Hamiltonian against the
reference ('ham') is also computed when --reference is given (the lambda = 0 data directory).
usage: force_error_spectrum.py OUT.json RUN [RUN ...] [--reference REF_DATA_DIR]
"""
import argparse
import json

import numpy as np
from ase.io import read

from screened_ips.spectra import force_error_spectrum


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--reference", default=None)
    a = p.parse_args()
    from mace.calculators import MACECalculator
    ref = read(f"{a.reference}/test.extxyz", ":") if a.reference else None
    out = {}
    for run in a.runs:
        c = MACECalculator(model_paths=f"{run}/models/m.model", device="cpu", default_dtype="float32")
        fr = read(f"{run}/data/test.extxyz", ":")
        P = []
        for at in fr:
            x = at.copy()
            x.calc = c
            P.append(x.get_forces())
        out[run] = {"own": force_error_spectrum(fr, [p - at.arrays["REF_forces"] for p, at in zip(P, fr)])}
        if ref is not None and not np.allclose(fr[0].arrays["REF_forces"], ref[0].arrays["REF_forces"]):
            out[run]["ham"] = force_error_spectrum(fr, [p - b.arrays["REF_forces"] for p, b in zip(P, ref)])
        print(run, "done", flush=True)
    if ref is not None:
        out["reference_forces"] = force_error_spectrum(ref, [b.arrays["REF_forces"] for b in ref])
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
