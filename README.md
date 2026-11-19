# screened-ips

Code, figure data and trained models for

> **Screened isotropic periodic sum targets make long-range electrostatics learnable by local machine-learning potentials**

Local graph-neural-network potentials cannot represent the part of a periodic Coulomb interaction that lies beyond
their cutoff. The isotropic periodic sum (IPS) separates the interaction at the cutoff into an interior that a local
model can learn and a homogeneous exterior that can be computed analytically. A model is trained on the IPS target

```
T_lambda = E_ref + lambda * Delta,     Delta = E_IPS(q; R_c) - E_Ewald(q)
```

and deployed with the exterior restored, `E_theta - lambda * Delta`. For quantum-mechanical data the correction
strength `lambda` is set by electronic screening (about 0.5 for molten salts) and can be estimated from three
ordinary trainings.

## Installation

```bash
conda env create -f environment.yml      # versions used in the paper
conda activate screened-ips
# or: pip install -e .[mace,figures]
```

Optional: MACE-LES runs and the PyTorch timing of `Delta` use the LES library
(<https://github.com/ChengUCB/les>, commit `fadfb0ca3b89ed371e5de6b383f4425defabd4a8`) and the LES-enabled MACE fork
(<https://github.com/ChengUCB/mace>, commit `c655c940fb42ab96e745c4f3c48b45c91a64d336`). The NequIP test used
`nequip==0.6.2`.

## Quick start

```python
from ase.io import read
from mace.calculators import MACECalculator
from screened_ips import delta, charges_for
from screened_ips.md import ScreenedIPSCalculator

atoms = read("examples/init100.xyz")                 # 100-atom molten NaCl configuration
dE, dF = delta(atoms, charges_for(atoms), rc=5.0)    # IPS correction (eV, eV/A)

base = MACECalculator(model_paths="models/nacl_screened_ips_lam0.52_r5.model", device="cpu", default_dtype="float32")
atoms.calc = ScreenedIPSCalculator(base, lam=0.5205)  # MACE + screened IPS
print(atoms.get_potential_energy())
```

## Repository layout

| Path | Content |
|---|---|
| `screened_ips/coulomb.py` | Ewald sum, IPS Coulomb kernel, shifted-force and DSF kernels, `delta()` |
| `screened_ips/dispersion.py` | Grimme D2 and the IPS-type r^-6 correction |
| `screened_ips/projection.py` | Single-Delta and instrumental-variable estimates of the strength |
| `screened_ips/spectra.py` | `S_ZZ(k)`, charge-current spectra, k-resolved force errors |
| `screened_ips/md.py` | ASE calculator `E_theta - lambda * Delta` |
| `scripts/` | Command-line tools for every step (each script documents its usage with `-h`) |
| `splits/nacl_split.json` | NaCl train/valid/test selection with geometry hashes |
| `models/` | Trained NaCl models (see `models/README.md`) |
| `figures/` | Figure scripts and all data needed to redraw the figures (see `figures/README.md`) |
| `docs/protocol/` | Study protocol as frozen before the confirmatory runs, and with time-stamped addenda |
| `examples/` | Starting configuration and an example batch script |

## Reproducing the paper

Reference data are not redistributed; the preparation scripts download-check and split the published sets.

**1. Reference data**

```bash
# NaCl, PBE-D3 (Attarian et al.; figshare doi:10.6084/m9.figshare.26535952.v1): Pot_N.pckl.gzip, Testing.pckl.gzip
python scripts/prepare_nacl.py RAW/ data/nacl/ref
# SuperSalt (Zenodo doi:10.5281/zenodo.15734798): extract Training.extxyz and Validation.extxyz
python scripts/prepare_supersalt.py RAW/ data/cscl/ref --cations Cs --seed 20261030      # also Ba, Sr
python scripts/prepare_supersalt.py RAW/ data/ss11/ref --cations Li,Na,K,Rb,Cs,Mg,Ca,Sr,Ba,Zn,Zr --seed 20261103
python scripts/prepare_supersalt.py RAW/ data/mgna/ref --cations Mg,Na --seed 20261103
# water, revPBE0-D3 (Cheng et al. 2019): training-set/dataset_1593.xyz
python scripts/prepare_water.py dataset_1593.xyz data/water/ref
```

**2. Targets.** `T_lambda` for a list of strengths, plus `Delta` alone for the strength estimate:

```bash
python scripts/make_targets.py data/nacl/ref data/nacl --lam 0,0.2,0.3,0.4,0.44,0.5,0.5205,0.6,delta
python scripts/make_targets.py data/nacl/ref data/nacl_dsf   --lam 0.5205 --kind dsf      # also sf, ewald
python scripts/make_targets.py data/nacl/ref data/nacl_base  --lam 0.5205 --form baseline # baseline model
python scripts/make_targets.py data/nacl/ref data/nacl_local --lam 0.5205 --form local    # local control
python scripts/make_targets.py data/nacl/ref data/nacl_perm  --lam 0.4 --permuted         # permuted charges
python scripts/make_targets.py data/nacl/ref data/nacl_r6    --lam 0.5205 --rc 6          # cutoff control
python scripts/make_targets.py data/water/ref data/water --charges tip3p --lam 0,0.25,0.5,0.75,1.0,1.25,delta
python scripts/make_subsets.py data/nacl/L0.5205 data/nacl_lc/L0.5205                     # learning curves
python scripts/dispersion.py targets data/nacl/L0.4 data/nacl_d2_full data/nacl_d2_ips    # D2 test
```

**3. Training** (`RUN_DIR/data` must contain the target; seeds 20260850-52):

```bash
mkdir -p runs/L0.5205 && ln -s $PWD/data/nacl/L0.5205 runs/L0.5205/data
python scripts/train_mace.py runs/L0.5205 20260850 20000          # confirmatory budget (scans: 5000)
python scripts/train_mace.py runs/R6 20260850 5000 --rmax 6       # cutoff control (data: data/nacl/L0)
python scripts/train_mace.py runs/ZBL 20260850 20000 --zbl        # models used for dynamics
python scripts/train_mace.py runs/LES 20260850 20000 --les        # MACE + LES
python scripts/train_nequip.py runs/NQ 20260850 5000 cpu          # NequIP robustness test
```
Water models used batch size 16 (`--batch 16`) and 7,500 updates.

**4. Evaluation and strength estimate**

```bash
python scripts/evaluate.py eval.json runs/L0 runs/L0.5205
python scripts/estimate_lambda.py lam.json runs/Ldelta,runs/Ldelta_s20260851,runs/Ldelta_s20260852 -- runs/L0
```

**5. Dynamics and long-wavelength diagnostics**

```bash
python scripts/run_md.py nvt models/nacl_screened_ips_lam0.52_r5_zbl.model md/BR_r1 --lam 0.5205 \
       --init examples/init100.xyz --repeat 2 --seed 1          # 800 atoms, 1200 K
python scripts/run_md.py nve models/nacl_screened_ips_lam0.52_r5_zbl.model md/NR_r1 --lam 0.5205 \
       --start md/BR_r1.traj.extxyz@300 --nprod 8000
python scripts/charge_structure_factor.py szz.json md/BR md/BU
python scripts/charge_current.py ccur.json md/NR_r1.traj.extxyz
python scripts/force_error_spectrum.py errk.json runs/L0 runs/L0.5205 --reference data/nacl/L0
python scripts/committee.py comm --models a.model,b.model,c.model --init examples/init100.xyz --lam 0
python scripts/analyze_md.py md/BR md_local.json
python scripts/make_teacher_targets.py models/nacl_screened_ips_lam0.52_r5_zbl.model data/teacher100 \
       --train-trajs t1.traj.extxyz,t2.traj.extxyz --test-traj t3.traj.extxyz
```

**6. Timing**

```bash
python scripts/benchmark.py model models/nacl_plain_mace_r5.model data/nacl/L0/test.extxyz t_model.json
python scripts/benchmark.py delta data/nacl/L0/test.extxyz t_delta.json --dl 1.0,1.5,2.0
```

**7. Figures**: see `figures/README.md`.

Random permutations of the charge-permuted control and the order of configurations written by these scripts may
differ from the original runs; all other splits are fixed by the seeds and files given above.

## Units and conventions

Positions in Angstrom, energies in eV (total; per-split energies are centred by the training mean per atom),
forces in eV/Angstrom, charges in e. Ewald sums use tin-foil boundary conditions. Formal charges: alkali +1,
alkaline earth and Zn +2, Zr +4, Cl -1; TIP3P: O -0.834, H +0.417.

## License

MIT (see `LICENSE`). The reference data sets are subject to their own licences.
