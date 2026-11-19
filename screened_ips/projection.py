"""Three-training estimate of the correction strength (Methods, Eq. 2).

r  : held-out force residual (reference - prediction) of a model trained on E_ref (or on T_lambda)
N_a: held-out residual of a model trained on Delta alone (seed a): the part of the Delta forces a local model
     cannot learn.
Single-Delta estimate  lambda' = -<r, N>/<N, N> is biased towards zero by the fitting noise in N;
instrumental-variable  lambda' = -<r, N_1>/<N_1, N_2> uses two independently seeded Delta models.
For a model trained on T_lambda, the same estimate returns lambda' - lambda.
"""
import numpy as np
from ase.io import read


def force_residual(run_dir, split="test", model_name="m.model", device="cpu"):
    """Reference minus predicted forces, concatenated over the split, for a MACE run directory."""
    from mace.calculators import MACECalculator
    frames = read(f"{run_dir}/data/{split}.extxyz", ":")
    calc = MACECalculator(model_paths=f"{run_dir}/models/{model_name}", device=device, default_dtype="float32")
    pred = []
    for a in frames:
        x = a.copy()
        x.calc = calc
        pred.append(x.get_forces())
    return np.concatenate([a.arrays["REF_forces"] for a in frames]) - np.concatenate(pred)


def single_delta_estimate(r, N):
    lam = -float((r * N).sum() / (N * N).sum())
    explained = float(1 - ((r + lam * N) ** 2).sum() / (r ** 2).sum())
    return lam, explained


def iv_estimate(r, N1, N2):
    return -float((r * N1).sum() / (N1 * N2).sum())


def iv_all_pairs(r, Ns):
    """IV estimates over all ordered pairs of Delta models."""
    return [iv_estimate(r, Ns[i], Ns[j]) for i in range(len(Ns)) for j in range(len(Ns)) if i != j]
