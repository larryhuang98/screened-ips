"""Screened isotropic-periodic-sum (IPS) targets for local machine-learning interatomic potentials."""
from .coulomb import (COULOMB_EV_ANGSTROM, FORMAL_CHARGES, TIP3P_CHARGES, charges_for, delta, ewald,
                      ips_coulomb, damped_shifted_force)

__version__ = "1.0.0"
