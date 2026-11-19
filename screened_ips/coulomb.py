"""Coulomb kernels for screened IPS targets.

Units: positions and cells in Angstrom, energies in eV (total), forces in eV/Angstrom, charges in e.

The IPS correction is
    Delta = E_IPS(q; R_c) - E_Ewald(q),
where E_IPS is the polynomial isotropic-periodic-sum Coulomb energy within the cutoff R_c and E_Ewald
is the full periodic (tin-foil) Ewald energy of the same fixed charges. A model trained on
T_lambda = E_ref + lambda * Delta is deployed as E_theta - lambda * Delta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.special import erfc

COULOMB_EV_ANGSTROM = 14.3996454784255  # e^2 / (4 pi eps0) in eV Angstrom

# Formal charges used in the paper (e).
FORMAL_CHARGES = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
                  "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Zr": 4, "Cl": -1}
TIP3P_CHARGES = {"O": -0.834, "H": 0.417}


def charges_for(atoms: Atoms, table: dict | None = None) -> np.ndarray:
    """Fixed charges from a symbol table (formal ionic charges by default)."""
    table = FORMAL_CHARGES if table is None else table
    q = np.array([table[s] for s in atoms.get_chemical_symbols()], dtype=float)
    if abs(q.sum()) > 1e-9:
        raise ValueError("configuration is not neutral")
    return q


# --------------------------------------------------------------------------- Ewald

@dataclass(frozen=True)
class EwaldSettings:
    alpha_per_A: float = 0.65
    tail_tolerance: float = 1.0e-12


def ewald(atoms: Atoms, charges: np.ndarray, settings: EwaldSettings = EwaldSettings()):
    """Direct real + reciprocal lattice Ewald sum with tin-foil boundary conditions.

    Real-space and reciprocal cutoffs follow alpha*r_max = k_max/(2 alpha) = sqrt(-log(tail_tolerance)).
    Returns (energy, forces). Converged to about 1e-7 eV for the cells used in the paper.
    """
    cell, positions = atoms.cell.array, atoms.positions
    volume = atoms.get_volume()
    alpha = settings.alpha_per_A
    tail = np.sqrt(-np.log(settings.tail_tolerance))
    real_cutoff, k_cutoff = tail / alpha, 2 * alpha * tail
    K = COULOMB_EV_ANGSTROM

    left, right, distances, vectors = neighbor_list("ijdD", atoms, real_cutoff, self_interaction=False)
    if np.any(distances < 1e-8):
        raise ValueError("overlapping periodic particles")
    pre = K * charges[left] * charges[right]
    screened = erfc(alpha * distances) / distances
    energy = 0.5 * float(np.sum(pre * screened))
    deriv = -pre * (screened / distances
                    + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * distances) ** 2) / distances)
    forces = np.zeros_like(positions)
    np.add.at(forces, left, (deriv / distances)[:, None] * vectors)

    limits = np.ceil(k_cutoff * np.linalg.norm(cell, axis=1) / (2 * np.pi)).astype(int)
    ints = np.stack(np.meshgrid(*(np.arange(-n, n + 1) for n in limits), indexing="ij"), axis=-1).reshape(-1, 3)
    kvec = ints @ (2 * np.pi * np.linalg.inv(cell).T)
    k2 = np.sum(kvec ** 2, axis=1)
    keep = (k2 > 0) & (k2 < k_cutoff ** 2)
    kvec, k2 = kvec[keep], k2[keep]
    factor = 2 * np.pi * K / volume
    for s in range(0, len(k2), 1024):
        k, kk = kvec[s:s + 1024], k2[s:s + 1024]
        phase = positions @ k.T
        c, sn = np.cos(phase), np.sin(phase)
        re, im = charges @ c, charges @ sn
        w = factor * np.exp(-kk / (4 * alpha ** 2)) / kk
        energy += float(np.sum(w * (re ** 2 + im ** 2)))
        forces += 2 * charges[:, None] * ((sn * re - c * im) * w) @ k
    energy += -K * alpha / np.sqrt(np.pi) * float(charges @ charges)
    return energy, forces


# --------------------------------------------------------------------------- in-cutoff kernels

def ips_polynomial(x):
    """IPS correction polynomial P(x), x = r/R_c; the kernel 1/r + P(r/R_c)/R_c and its slope vanish at R_c."""
    return (-35 + 35 * x ** 2 - 21 * x ** 4 + 5 * x ** 6) / 16


def ips_coulomb(atoms: Atoms, charges: np.ndarray, rc: float = 5.0):
    """IPS Coulomb energy and forces within rc (pairs over periodic images), including the IPS self term."""
    i, j, d, D = neighbor_list("ijdD", atoms, rc, self_interaction=False)
    x = d / rc
    kern = 1 / d + ips_polynomial(x) / rc
    dk = -1 / d ** 2 + (70 * x - 84 * x ** 3 + 30 * x ** 5) / (16 * rc ** 2)
    pre = COULOMB_EV_ANGSTROM * charges[i] * charges[j]
    e = 0.5 * np.sum(pre * kern) + 0.5 * COULOMB_EV_ANGSTROM * np.sum(charges ** 2) * ips_polynomial(0.0) / rc
    f = np.zeros_like(atoms.positions)
    np.add.at(f, i, (pre * dk / d)[:, None] * D)
    return float(e), f


def damped_shifted_force(atoms: Atoms, charges: np.ndarray, rc: float = 5.0, alpha: float = 0.2):
    """Damped shifted-force Coulomb (Fennell and Gezelter); alpha = 0 gives the shifted-force kernel."""
    i, j, d, D = neighbor_list("ijdD", atoms, rc, self_interaction=False)
    ec = erfc(alpha * rc) / rc
    fc = erfc(alpha * rc) / rc ** 2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rc) ** 2) / rc
    v = erfc(alpha * d) / d - ec + fc * (d - rc)
    dv = -(erfc(alpha * d) / d ** 2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * d) ** 2) / d) + fc
    pre = COULOMB_EV_ANGSTROM * charges[i] * charges[j]
    e = 0.5 * np.sum(pre * v) - COULOMB_EV_ANGSTROM * (ec / 2 + alpha / np.sqrt(np.pi)) * np.sum(charges ** 2)
    f = np.zeros_like(atoms.positions)
    np.add.at(f, i, (pre * dv / d)[:, None] * D)
    return float(e), f


# --------------------------------------------------------------------------- corrections

def _clean(atoms: Atoms) -> Atoms:
    a = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=True)
    a.wrap()
    return a


def delta(atoms: Atoms, charges: np.ndarray | None = None, rc: float = 5.0, kind: str = "ips"):
    """Exterior correction Delta = E_kind(q; rc) - E_Ewald(q) and its forces.

    kind: 'ips' (paper default), 'sf' (shifted force), 'dsf' (damped shifted force, alpha = 0.2 / A),
    'ewald' (plain Delta-learning: Delta = -E_Ewald).
    Intramolecular bare-Coulomb exclusions (e.g. TIP3P) cancel exactly between the two terms.
    """
    a = _clean(atoms)
    q = charges_for(a) if charges is None else np.asarray(charges, float)
    e_ew, f_ew = ewald(a, q)
    if kind == "ewald":
        return -e_ew, -f_ew
    if kind == "ips":
        e1, f1 = ips_coulomb(a, q, rc)
    elif kind in ("sf", "dsf"):
        e1, f1 = damped_shifted_force(a, q, rc, 0.2 if kind == "dsf" else 0.0)
    else:
        raise ValueError(kind)
    return e1 - e_ew, f1 - f_ew
