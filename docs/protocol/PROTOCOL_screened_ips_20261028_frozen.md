# Screened-charge IPS label transformation — prospective protocol (frozen 2026-10-28 16:55 PDT)

Written BEFORE any full-budget / learning-curve / MD / SuperSalt result exists.
Exploratory results already seen (NOT confirmatory): 5k-update lambda scans
(NaCl seed 20260850 test; NaCl 3-seed validation curves at epoch<=60; water 3-seed test at lambda 0,0.5,1.0).

## Target
T_lam = E_QM + lam * (E_IPSp(q,Rc=5) - E_Ewald(q)), forces likewise; q = formal charges
(NaCl +-1; water TIP3P shape O -0.834, H +0.417). Intramolecular bare-Coulomb exclusions cancel exactly.

## A-priori lambda (no training errors used)
NaCl: lam_ECC = 1/eps_inf = 1/1.9213338 = 0.5205; eps_inf from the project CP2K PBE finite-field
calculation (2026-10-10, 128-atom molten NaCl, 1.5 g/cm3; QM_METHOD_WEEK_PROGRESS_2026_10_09.md).
Water: lam = 0.5 (best of {0,0.5,1.0} on VALIDATION in the 3-seed 5k scan); reported as validation-selected.
SuperSalt single salts: lam = 1/eps_inf, eps_inf from Clausius-Mossotti with fixed literature ionic
polarizabilities and frame number density; values recorded before training.

## Confirmatory 1: learning curves (NaCl, Attarian Pot_N PBE-D3)
n_train in {80,160,320,640}, nested subsets (rng 20261029), seeds 20260850-52, 20000 updates,
MACE R5 2x32 L2 corr3, identical hyperparameters, validation-selected EMA checkpoint, author test set (62).
Primary endpoint: test force-component RMSE (own target), lam_ECC vs lam=0, paired by seed.
P1: lam_ECC lower at n=640 in 3/3 seeds. P2: relative gain does not shrink as n grows
(nonlocal floor: e^2 = e_loc^2(n) + (lam-lam')^2 N^2). Secondary: centered energy RMSE, reported regardless.

## Confirmatory 2: scrambled-charge control
Species-permuted charges, RMS-matched dF. Prediction: QM-residual projection explains <5% of variance;
lam=0.4 scrambled rewrite worse than lam=0.

## Experiment 3: MD fidelity (NaCl, 1200 K NVT Langevin; 100 atoms and 2x2x2 = 800 atoms)
Models (n=640, seed 20260850): M0 (lam=0 local), ML (lam_ECC, local, IPS physics),
MA (lam_ECC model + analytic add-back = QM surrogate with Ewald). Observables: stability, RDFs,
coordination, Na/Cl self-diffusion. Questions: physical cost of the IPS target (ML vs MA) and of plain
truncation (M0 vs MA). No direction predicted.

## Experiment 4: generality on SuperSalt (PBE-D3, boxes up to 18 A)
Single-salt subsets chosen by frame count only (before training): lam=0 vs lam_CM, 3 seeds.
Prediction from the N-diagnostic: gain grows with N_rms / e_loc.

All negative or null outcomes are reported. Test sets are never used for selection.
