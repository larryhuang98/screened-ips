"""Reference values for examples/init100.xyz (100-atom molten NaCl, formal charges, R_c = 5 A).

Run with:  pytest -q tests
They were obtained with the code that produced the paper and require only numpy, scipy and ase.
"""
import os

import numpy as np
import pytest
from ase.io import read

from screened_ips import charges_for, delta
from screened_ips.dispersion import d2

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def atoms():
    return read(os.path.join(HERE, "..", "examples", "init100.xyz"))


def test_ips_correction(atoms):
    e, f = delta(atoms, charges_for(atoms))
    assert e == pytest.approx(-3.071730267538726, abs=1e-6)
    assert np.sqrt(np.mean(f ** 2)) == pytest.approx(0.3410043321933886, abs=1e-6)
    assert f[3] == pytest.approx([-0.56680236, -0.18923846, 0.42549516], abs=1e-6)


def test_forces_are_energy_gradients(atoms):
    q = charges_for(atoms)
    _, f = delta(atoms, q)
    h = 1e-4
    for c in range(3):
        p, m = atoms.copy(), atoms.copy()
        p.positions[3, c] += h
        m.positions[3, c] -= h
        assert -(delta(p, q)[0] - delta(m, q)[0]) / (2 * h) == pytest.approx(f[3, c], abs=1e-6)


@pytest.mark.parametrize("kind,value", [("sf", 103.30399190608375), ("dsf", 57.88832849369112),
                                        ("ewald", 385.41987098030665)])
def test_other_kernels(atoms, kind, value):
    assert delta(atoms, charges_for(atoms), kind=kind)[0] == pytest.approx(value, abs=1e-6)


def test_dispersion(atoms):
    assert d2(atoms)[0] == pytest.approx(-12.8609325817119, abs=1e-6)
    assert d2(atoms, ips_rc=5.0)[0] == pytest.approx(-11.030892442176183, abs=1e-6)


def test_translation_invariance(atoms):
    q = charges_for(atoms)
    b = atoms.copy()
    b.translate([1.3, -0.7, 2.1])
    assert delta(b, q)[0] == pytest.approx(delta(atoms, q)[0], abs=1e-6)
