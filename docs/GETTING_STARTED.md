# Getting started

This note shows how to install the code, check that it works, and what numbers you should get.
Everything below runs on a laptop CPU unless stated otherwise.

## 1. What the code does

A local ML potential (here MACE, cutoff 5 A) cannot learn the part of the Coulomb interaction beyond its cutoff.
We remove that part from the training labels and add it back analytically:

- train on `T = E_ref + lambda * Delta`, with `Delta = E_IPS - E_Ewald` (fixed formal charges, Na +1, Cl -1);
- predict with `E_theta - lambda * Delta`.

`lambda` is the screened strength (0.5205 for NaCl). `screened_ips.delta()` computes `Delta` and its forces;
`screened_ips.md.ScreenedIPSCalculator` wraps a MACE model so that the exterior is added back automatically.

## 2. Install

```bash
git clone https://github.com/larryhuang98/screened-ips.git
cd screened-ips
conda env create -f environment.yml
conda activate screened-ips
```

If you only want the Coulomb code (no MACE): `pip install -e .`

## 3. Test 1: Coulomb code (1 minute, no MACE needed)

```bash
pip install pytest
pytest -q tests
```

Expected: `7 passed`. The tests compare against values from the code used for the paper, for the
100-atom configuration `examples/init100.xyz`:

| Quantity | Expected |
|---|---|
| `Delta` energy (IPS kernel, R_c = 5 A) | -3.071730 eV |
| RMS of `Delta` forces | 0.341004 eV/A |
| `Delta` with shifted-force / DSF / Ewald-only kernels | 103.3040 / 57.8883 / 385.4199 eV |
| D2 dispersion, full sum / IPS6-corrected | -12.8609 / -11.0309 eV |

The tests also check that the forces are the negative gradient of the energy (finite differences).

## 4. Test 2: trained models on one configuration (1 minute)

```bash
python scripts/check_models.py examples/init100.xyz
```

`init100.xyz` is a validation configuration with its DFT forces. Expected force RMSE (meV/A), within about 0.1:

| Model | lambda | Force RMSE |
|---|---|---|
| plain_r5 | 0 | 44.6 |
| screened_ips_0.52_r5 | 0.5205 | 18.5 |
| screened_ips_0.44_r5 | 0.44 | 16.0 |
| plain_r5_zbl | 0 | 44.3 |
| screened_ips_0.52_r5_zbl | 0.5205 | 18.9 |
| plain_r6 | 0 | 17.9 |
| screened_ips_0.52_r6 | 0.5205 | 15.2 |
| plain_r7 | 0 | 10.5 |

The point to see: at the same cutoff, the screened IPS model is much more accurate than plain MACE.
(The 6 and 7 A plain models look good on this small cell; see Section 7 for why that is misleading.)

## 5. Test 3: full NaCl test set (10 minutes)

Download `Pot_N.pckl.gzip` and `Testing.pckl.gzip` from figshare
(doi:10.6084/m9.figshare.26535952.v1) into a folder `raw/`, then

```bash
python scripts/prepare_nacl.py raw data/nacl/ref
python scripts/check_models.py data/nacl/ref/test.extxyz
```

`prepare_nacl.py` checks MD5 sums and geometry hashes and should print `train 640`, `valid 201`, `test 62`.
Expected test errors (seed 20260850; small differences in the last digit are fine):

| Model | Force RMSE (meV/A) | Energy RMSE (meV/atom) |
|---|---|---|
| plain_r5 | 17.27 | 2.52 |
| screened_ips_0.52_r5 | 10.22 | 2.71 |
| screened_ips_0.44_r5 | 9.36 | 2.73 |
| plain_r5_zbl | 17.23 | 2.48 |
| screened_ips_0.52_r5_zbl | 10.17 | 2.68 |
| plain_r6 (5,000 updates) | 9.15 | 1.12 |
| screened_ips_0.52_r6 (5,000 updates) | 8.47 | 1.24 |
| plain_r7 (5,000 updates) | 6.61 | 0.41 |

The force error drops by about 41% when the screened exterior is used (17.3 -> 10.2).

## 6. Test 4: short molecular dynamics (10-20 minutes)

Run 4 ps of 1200 K dynamics for 100 atoms with the screened model, exterior restored:

```bash
python scripts/run_md.py nvt models/nacl_screened_ips_lam0.52_r5_zbl.model md/test_restored --lam 0.5205 \
       --init examples/init100.xyz --neq 0 --nprod 2000
```

Expected: `md/test_restored.json` has `"ok": true`; in `md/test_restored.log` (columns: step, energy,
temperature, minimum distance, time) the temperature fluctuates around 1200 K (roughly 1000-1400 K for 100 atoms)
and the minimum interatomic distance stays above about 1.9 A.

Now run the same model **without** adding the exterior back (`--lam 0`, the "IPS Hamiltonian"):

```bash
python scripts/run_md.py nvt models/nacl_screened_ips_lam0.52_r5_zbl.model md/test_ham --lam 0 \
       --init examples/init100.xyz --neq 0 --nprod 2000
```

Expected: the run usually stops early with `"ok": false` (in the paper every such run collapsed, after 0.8-1.8 ps,
i.e. 400-900 steps; a given seed may last somewhat longer).
This is the most common mistake: **a model trained on an IPS target must always be used with `lam` > 0**.

## 7. Optional: the large-cell check

Small QM cells (13-15 A) fit almost inside the receptive field of a 6-7 A model, so small-cell errors make a larger
cutoff look better than it is. The real test is the long-wavelength charge response of an 800-atom cell
(`--repeat 2`, about 1 day on 16 CPU cores per model):

```bash
python scripts/run_md.py nvt MODEL md/BIG_r1 --lam LAM --init examples/init100.xyz --repeat 2 --seed 1
python scripts/charge_structure_factor.py szz.json md/BIG
```

Expected `S_ZZ` at k = 0.20 1/A: about 0.0017 for the screened model (the Stillinger-Lovett value), about 0.006
for plain MACE at 5 A, 0.0052 at 6 A and 0.0044 at 7 A.

## 8. Optional: train a model yourself

```bash
python scripts/make_targets.py data/nacl/ref data/nacl --lam 0,0.4
mkdir -p runs/L0 runs/L0.4
ln -s $PWD/data/nacl/L0 runs/L0/data; ln -s $PWD/data/nacl/L0.4 runs/L0.4/data
python scripts/train_mace.py runs/L0 20260850 5000
python scripts/train_mace.py runs/L0.4 20260850 5000
python scripts/evaluate.py eval.json runs/L0 runs/L0.4
```

Each run takes several hours on CPU (minutes on a GPU, `--device cuda`). Expected test force RMSE after
5,000 updates: about 20.4 meV/A for `L0` and 12.4 meV/A for `L0.4` (mean of three seeds; one seed varies by about
0.5). `evaluate.py` reports errors on each model's own target, which equal the errors of the deployed model.

## 9. Things to remember

- Always deploy an IPS-target model with the same `lambda` and cutoff `R_c` it was trained with
  (`screened_ips_0.52_r6` uses `rc = 6`).
- Charges are fixed formal charges; configurations must be neutral.
- Energies in the data files are shifted by a constant per atom; compare energies only after removing the mean
  offset, as the scripts do.
- Every script prints its usage with `-h`.
