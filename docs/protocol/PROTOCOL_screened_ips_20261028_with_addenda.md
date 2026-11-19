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

## Addendum 1 (2026-10-28 17:05 PDT, before any SuperSalt training)
Subsets: top-3 single salts by SuperSalt Training frame count: CsCl (946), BaCl2 (931), SrCl2 (909).
Split per salt: 640 train / 150 valid drawn from SuperSalt Training (rng 20261030); test = all SuperSalt
Validation frames of that salt (75). Charges formal (Cs+1, Ba+2, Sr+2, Cl-1).
lam rule changed: validation-selected from {0,0.3,0.5,0.7} at 5000 updates (seed 20260850), then
lam=0 vs lam_sel at 20000 updates x 3 seeds on test. Clausius-Mossotti 1/eps_inf (Shannon & Fischer 2016
polarizabilities) reported only as a post-hoc physics cross-check. Box heights must exceed 10 A.

## Addendum 2 (2026-10-28 21:45 PDT) — MD exposed arbitrary short-range extrapolation
Exploratory NVT at 1200 K without a repulsive prior: the lam=0.5205 pure-local model (ML) collapsed in 3/3
replicas within 0.8-2.3 ps (Na-Cl approach to ~0.5-0.8 A); lam=0 (M0) and add-back (MA) were still running.
A rigid Na-Cl contact scan below the 1.98 A training minimum showed BOTH models are unphysical there
(M0: erratic 1e3-1e8 eV values; ML: ~60 eV barrier then a ~1 eV hole). Stability without a prior therefore
reflects arbitrary extrapolation and is not used as evidence for or against the method; these runs are reported.
Change: MD models are retrained with MACE --pair_repulsion (ZBL) for lam=0 and lam=0.5205, n=640,
seeds 20260850-52, 20000 updates; learnability of ZBL models is reported alongside. MD protocol otherwise unchanged.

## Addendum 3 (2026-10-29 03:05 PDT)
SuperSalt validation-selected lam (5k scan, seed 20260850): CsCl 0.5, BaCl2 0.3, SrCl2 0.3 (selected on validation only).
Projection of lam=0 residuals onto the unlearned part of dF gives lam' = 0.46-0.47 (CsCl), 0.29-0.31 (BaCl2, SrCl2),
explaining 45-69% of residual variance (recorded before any confirmatory run).
Confirmatory runs: GPU unavailable (all but one GPU in use by other projects), so SuperSalt confirmation is run on CPU
at 5000 updates (same budget as the NaCl confirmatory scan), lam=0 vs lam_sel, seeds 20260850-52.
MD (no ZBL, 3x50 ps, 100 atoms, 1200 K): M0 and MA stable 3/3; ML collapsed 3/3 (see Addendum 2).

## Addendum 4 (2026-10-29 17:00 PDT) — deployment mode decided by MD
Pure-local deployment of the transformed Hamiltonian H_IPS = QM + lam(IPS-Ewald) failed in every tested form at 1200 K
(100 atoms): learned-target model without ZBL 3/3 collapse, with ZBL 3/3, and (QM - lam*Ewald)-network + analytic lam*IPS 3/3
(collapse delayed to 4.6-7 ps). Pre-collapse frames show normal minimum distances (2.1-2.3 A) and a steady potential-energy
runaway; the final contacts are frequently like-ion (Cl-Cl, Na-Na). Interpretation (not proven): distribution shift — the IPS
Hamiltonian ensemble differs from the QM sampling ensemble (the classical study sampled both parents); on-policy QM labels
would be required. QM-consistent deployment (network - lam*(IPS-Ewald), i.e. analytic screened fixed-charge Ewald exterior)
is stable 3/3 with and without ZBL and matches lam=0 structure/diffusion within noise.
Decision: the paper's deployed model is the QM-consistent add-back model. Pure-local results are reported as a negative result.

## Addendum 5 (2026-10-30 13:40 PDT)
Full-budget NaCl (n640, 20k, 3 seeds, test F/E): MACE lam0 17.24/2.54; MACE+IPS lam_ECC=0.5205 10.16/2.64;
LES 8.91/3.54; LES on IPS target (0.5205) 9.58/2.56. Learning curves: relative IPS gain 38/42/44/41% at n=80/160/320/640 (P2 holds).
LES latent charges (raw QM): q_Na=+0.55, q_Cl=-0.56 (q^2=0.30-0.32); LES on IPS target: ~0 charges.
NEW (exploratory, decided AFTER seeing LES full-budget results): IPS at lam=0.38 (validation projection lam') and 0.33
(test-free lower projection value), n640, 20k, 3 seeds. Reported as post-hoc; lam_ECC remains the pre-registered primary.

## Addendum 6 — erratum (2026-10-31, after independent numerical audit)
(a) Addendum 5's header time "13:40 PDT" is wrong; the file was last modified at 13:24 PDT on 2026-10-30.
(b) Addendum 5 called lam=0.33 "test-free"; 0.33 is the TEST-set projection (validation gives 0.38). The lam=0.33 run is
    test-informed and is not used for any claim.
(c) The permuted lam=0.4 control run started 2026-10-28 07:25, before this protocol was frozen (finished afterwards).
(d) P2 (relative gain does not shrink with n) was not met strictly (44% at n=320 -> 41% at n=640 in every seed).
(e) Experiment-4 prediction (gain grows with N_rms/e_loc among SuperSalt salts) failed.
(f) 800-atom MD (Experiment 3) was not performed.
(g) LES charges quoted earlier (+0.55/-0.56) came from the first 20 test frames and two seeds; full test set, three seeds: q_Na +0.52, q_Cl -0.53.
