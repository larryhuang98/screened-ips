#!/usr/bin/env python
"""Latent LES charges of trained MACE-LES models on NaCl test frames (per-species mean and spread).

usage: les_charges.py RUN [RUN ...]   (requires the LES-enabled MACE fork)
"""
import sys, numpy as np, torch
from ase.io import read
from mace.calculators import MACECalculator
for run in sys.argv[1:]:
    c = MACECalculator(model_paths=f"{run}/models/m.model", device="cpu", default_dtype="float64")
    model = c.models[0]
    fr = read(f"{run}/data/test.extxyz", ":")
    qs = {11: [], 17: []}
    for a in fr:
        batch = c._atoms_to_batch(a)
        out = model(batch.to_dict(), compute_force=False, training=False)
        q = out["latent_charges"].detach().numpy().ravel()
        for z in (11, 17): qs[z] += list(q[a.numbers == z])
    na, cl = np.array(qs[11]), np.array(qs[17])
    print(run, f"q_Na {na.mean():.3f}+-{na.std():.3f}  q_Cl {cl.mean():.3f}+-{cl.std():.3f}  q_Na^2 {np.mean(na)**2:.3f}  frame_sum {na.sum()/len(fr)+cl.sum()/len(fr):.3f}")
