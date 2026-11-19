"""ASE calculator that deploys a model trained on an IPS target: E = E_theta - lambda * Delta."""
import time

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .coulomb import charges_for, delta


class ScreenedIPSCalculator(Calculator):
    """Wraps a base calculator (e.g. MACECalculator) trained on T_lambda = E_ref + lambda * Delta.

    lam = 0 gives the base model unchanged (plain MACE, or the IPS Hamiltonian if the base model was trained
    on an IPS target); lam > 0 restores the periodic exterior, E_theta - lambda * Delta.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, base, lam=0.0, rc=5.0, charge_table=None, **kw):
        super().__init__(**kw)
        self.base, self.lam, self.rc, self.charge_table = base, float(lam), rc, charge_table
        self.timing = {"model": 0.0, "delta": 0.0}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        t = time.time()
        a = atoms.copy()
        a.calc = self.base
        e, f = a.get_potential_energy(), a.get_forces()
        self.timing["model"] += time.time() - t
        if self.lam:
            t = time.time()
            de, df = delta(atoms, charges_for(atoms, self.charge_table), self.rc)
            e, f = e - self.lam * de, f - self.lam * df
            self.timing["delta"] += time.time() - t
        self.results = {"energy": e, "forces": np.asarray(f)}
