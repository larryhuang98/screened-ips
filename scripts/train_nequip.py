#!/usr/bin/env python
"""NequIP (v0.6.2) trainer matched to the MACE setup of the paper (not tuned).

usage: train_nequip.py RUN_DIR SEED UPDATES DEVICE   (data in RUN_DIR/data/{train,valid}.extxyz)
The deployed model is written to RUN_DIR/models/m.model.
"""
import sys, os, math, subprocess, yaml
from ase.io import read
d, seed, updates, dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ntr = len(read(f"{d}/data/train.extxyz", ":")); nva = len(read(f"{d}/data/valid.extxyz", ":"))
syms = sorted({s for a in read(f"{d}/data/train.extxyz", ":5") for s in a.get_chemical_symbols()})
bs = 32; epochs = math.ceil(updates / math.ceil(ntr / bs))
cfg = dict(root=f"{d}/results", run_name="run", seed=seed, dataset_seed=seed, append=True, default_dtype="float32",
  model_builders=["SimpleIrrepsConfig", "EnergyModel", "PerSpeciesRescale", "ForceOutput", "RescaleEnergyEtc"],
  r_max=5.0, num_layers=2, l_max=2, parity=True, num_features=32, nonlinearity_type="gate", num_basis=8,
  BesselBasis_trainable=True, PolynomialCutoff_p=6, invariant_layers=2, invariant_neurons=64, avg_num_neighbors="auto", use_sc=True,
  dataset="ase", dataset_file_name=f"{d}/data/train.extxyz", ase_args={"format": "extxyz"},
  key_mapping={"REF_energy": "total_energy", "REF_forces": "forces"}, include_keys=["REF_energy", "REF_forces"],
  validation_dataset="ase", validation_dataset_file_name=f"{d}/data/valid.extxyz",
  chemical_symbols=syms, n_train=ntr, n_val=nva, batch_size=bs, validation_batch_size=bs, max_epochs=epochs,
  learning_rate=0.005, train_val_split="sequential", shuffle=True, metrics_key="validation_loss",
  use_ema=True, ema_decay=0.99, ema_use_num_updates=True, report_init_validation=True,
  loss_coeffs={"forces": 50.0, "total_energy": [10.0, "PerAtomMSELoss"]},
  metrics_components=[["forces", "rmse"], ["total_energy", "rmse", {"PerAtom": True}]],
  optimizer_name="Adam", optimizer_amsgrad=False, lr_scheduler_name="ReduceLROnPlateau", lr_scheduler_patience=20, lr_scheduler_factor=0.7,
  early_stopping_lower_bounds={"LR": 1.0e-6}, per_species_rescale_shifts=0.0, per_species_rescale_scales="dataset_forces_rms",
  per_species_rescale_shifts_trainable=False, per_species_rescale_scales_trainable=False, device=dev)
os.makedirs(d, exist_ok=True); yaml.safe_dump(cfg, open(f"{d}/nq.yaml", "w"))
print("n_train", ntr, "epochs", epochs, flush=True)
r = subprocess.run([sys.executable, "-m", "nequip.scripts.train", f"{d}/nq.yaml"])
if r.returncode == 0:
    os.makedirs(f"{d}/models", exist_ok=True)
    subprocess.run([sys.executable, "-m", "nequip.scripts.deploy", "build", "--train-dir", f"{d}/results/run", f"{d}/models/m.model"], check=True)
sys.exit(r.returncode)
