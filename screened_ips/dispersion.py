"""Grimme D2 dispersion (PBE, s6 = 0.75) and its IPS-type r^-6 correction within a cutoff.

IPS6 kernel: K(x) = x^-6 + P6(x) (in units of R_c^-6), x = r/R_c, P6 = 20 - 75 x^2 + 84 x^4 - 30 x^6, so that
K(1) = K'(1) = K''(1) = 0 and int_0^1 P6 x^2 dx = int_1^inf x^-4 dx = 1/3 (the energy of a uniform exterior is
folded into the sphere). Parameters: S. Grimme, J. Comput. Chem. 27, 1787 (2006).
"""
import numpy as np
from ase.neighborlist import neighbor_list

EVA6 = 10.364                      # J nm^6 mol^-1 -> eV A^6
C6 = {11: 5.71 * EVA6, 17: 5.07 * EVA6}
R0 = {11: 1.144, 17: 1.639}
S6, DAMP = 0.75, 20.0
P6 = np.array([20.0, -75.0, 84.0, -30.0])


def d2(atoms, rc: float = 30.0, ips_rc: float | None = None):
    """D2 energy and forces: full periodic sum to rc (default), or IPS6-corrected within ips_rc."""
    cut = rc if ips_rc is None else ips_rc
    i, j, d, D = neighbor_list("ijdD", atoms, cut, self_interaction=False)
    z = atoms.numbers
    c6 = np.sqrt(np.array([C6[k] for k in z[i]]) * np.array([C6[k] for k in z[j]]))
    r0 = np.array([R0[k] for k in z[i]]) + np.array([R0[k] for k in z[j]])
    ex = np.exp(-DAMP * (d / r0 - 1))
    fd = 1 / (1 + ex)
    dfd = fd ** 2 * ex * DAMP / r0
    k = fd / d ** 6
    dk = dfd / d ** 6 - 6 * fd / d ** 7
    if ips_rc is not None:
        x = d / ips_rc
        k = k + (P6[0] + P6[1] * x ** 2 + P6[2] * x ** 4 + P6[3] * x ** 6) / ips_rc ** 6
        dk = dk + (2 * P6[1] * x + 4 * P6[2] * x ** 3 + 6 * P6[3] * x ** 5) / ips_rc ** 7
    e = -0.5 * S6 * np.sum(c6 * k)
    g = -S6 * c6 * dk
    f = np.zeros_like(atoms.positions)
    np.add.at(f, i, (g / d)[:, None] * D)
    return float(e), f
