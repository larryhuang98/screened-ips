#!/usr/bin/env python
"""Three-training estimate of the correction strength.

Train one model on E_ref (or on T_lambda) and two or more independently seeded models on Delta alone
(make_targets.py --lam delta). Their held-out force residuals give
    single-Delta estimate  lambda' = -<r, N>/<N, N>
    instrumental variable  lambda' = -<r, N_a>/<N_a, N_b>   (all ordered pairs a != b)
For a model trained on T_lambda the estimate is lambda' - lambda.

usage: estimate_lambda.py OUT.json DELTA_RUN1,DELTA_RUN2[,...] -- RUN [RUN ...] [--splits valid,test]
"""
import argparse
import json
import sys

import numpy as np

from screened_ips.projection import force_residual, iv_all_pairs, single_delta_estimate


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("deltas")
    p.add_argument("--splits", default="valid,test")
    a = p.parse_args(argv[:i])
    runs = argv[i + 1:]
    deltas = a.deltas.split(",")
    out = {}
    for s in a.splits.split(","):
        Ns = [force_residual(d, s) for d in deltas]
        o = {"N_rms_meV_A": [1e3 * float(np.sqrt(np.mean(n ** 2))) for n in Ns]}
        if len(Ns) > 1:
            o["N_cross_rms_meV_A"] = 1e3 * float(np.sqrt(abs(np.mean(Ns[0] * Ns[1]))))
        for r in runs:
            R = force_residual(r, s)
            lam, expl = single_delta_estimate(R, Ns[0])
            d = dict(single_delta=lam, explained_fraction=expl, F_meV_A=1e3 * float(np.sqrt(np.mean(R ** 2))))
            if len(Ns) > 1:
                iv = iv_all_pairs(R, Ns)
                d.update(iv=iv, iv_mean=float(np.mean(iv)))
            o[r] = d
        out[s] = o
        print(s, json.dumps(o), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
