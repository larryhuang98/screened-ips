"""Reading and writing labelled configurations (extended XYZ with REF_energy / REF_forces)."""
import os

import numpy as np
from ase.io import read, write

SPLITS = ("train", "valid", "test")


def load_split(directory, split):
    return read(os.path.join(directory, f"{split}.extxyz"), ":")


def labels(atoms):
    return float(atoms.info["REF_energy"]), np.asarray(atoms.arrays["REF_forces"], float)


def write_labelled(path, rows, shift_per_atom=0.0, keep_info=()):
    """rows: iterable of (atoms, energy, forces). Energies are shifted by -N * shift_per_atom."""
    out = []
    for a, e, f in rows:
        x = a.copy()
        x.calc = None
        x.info = {k: a.info[k] for k in keep_info if k in a.info}
        x.info["REF_energy"] = float(e - shift_per_atom * len(a))
        x.arrays = {k: v for k, v in a.arrays.items() if k in ("numbers", "positions")}
        x.arrays["REF_forces"] = np.asarray(f, float)
        out.append(x)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write(path, out, format="extxyz")


def write_dataset(outdir, rows_by_split, center_on="train", keep_info=()):
    """Write train/valid/test; energies are centred by the mean energy per atom of the training split."""
    tr = rows_by_split[center_on]
    mu = float(np.mean([e / len(a) for a, e, _ in tr]))
    for s, rows in rows_by_split.items():
        write_labelled(os.path.join(outdir, f"{s}.extxyz"), rows, mu, keep_info)
    return mu
