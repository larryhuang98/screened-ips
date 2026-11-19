#!/usr/bin/env python
"""Local structure and diffusion from NVT trajectories: partial RDFs (Na-Cl, Na-Na, Cl-Cl), first-shell
Na-Cl coordination number, and self-diffusion coefficients of Na and Cl (MSD slope between 2 and 10 ps).

usage: analyze_md.py PREFIX OUT.json   (all PREFIX*.traj.extxyz with matching .json metadata; frames every 50 fs)
"""
import sys, glob, json, numpy as np
from ase.io import read
dt_frame = 25 * 2.0e-3  # ps between saved frames
def unwrap(frames):
    P = np.array([f.get_scaled_positions(wrap=False) for f in frames]); d = np.diff(P, axis=0); d -= np.round(d)
    S = np.concatenate([P[:1], P[:1] + np.cumsum(d, axis=0)]); L = frames[0].cell.array
    return S @ L
def rdf(frames, a, b, rmax=7.5, nb=150):
    edges = np.linspace(0, rmax, nb + 1); h = np.zeros(nb); V = frames[0].get_volume()
    sym = np.array(frames[0].get_chemical_symbols()); ia, ib = np.where(sym == a)[0], np.where(sym == b)[0]
    for f in frames[::4]:
        D = f.get_all_distances(mic=True)[np.ix_(ia, ib)]
        if a == b: D = D[np.triu_indices(len(ia), 1)]
        h += np.histogram(D.ravel(), edges)[0]
    nf = len(frames[::4]); npair = len(ia) * len(ib) if a != b else len(ia) * (len(ia) - 1) / 2
    shell = 4 / 3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    g = h / (nf * npair * shell / V); r = 0.5 * (edges[1:] + edges[:-1]); return r, g
out = {}
for tr in sorted(glob.glob(sys.argv[1] + "*.traj.extxyz")):
    tag = tr.split("/")[-1].replace(".traj.extxyz", "")
    meta = json.load(open(tr.replace(".traj.extxyz", ".json")))
    fr = read(tr, ":")
    if len(fr) < 50: out[tag] = dict(meta=meta, n=len(fr)); continue
    R = unwrap(fr); sym = np.array(fr[0].get_chemical_symbols()); t = np.arange(len(fr)) * dt_frame
    res = dict(meta=meta, n=len(fr))
    for s in ["Na", "Cl"]:
        idx = sym == s; lags = np.arange(1, len(fr) // 2)
        msd = np.array([np.mean(np.sum((R[l:, idx] - R[:-l, idx])**2, axis=2)) for l in lags])
        tl = lags * dt_frame; m = (tl >= 2) & (tl <= min(10, tl.max()))
        res[f"D_{s}_1e-9m2s"] = float(np.polyfit(tl[m], msd[m], 1)[0] / 6 * 1e-20 / 1e-12 / 1e-9)
    for a, b in [("Na", "Cl"), ("Na", "Na"), ("Cl", "Cl")]:
        r, g = rdf(fr, a, b); res[f"g_{a}{b}"] = g.tolist(); res["r"] = r.tolist()
        res[f"peak_{a}{b}"] = [float(r[np.argmax(g)]), float(g.max())]
    r, g = np.array(res["r"]), np.array(res["g_NaCl"]); rho = (sym == "Cl").sum() / fr[0].get_volume()
    i1 = np.argmax(g); imin = i1 + np.argmin(g[i1:i1 + 40]); dr = r[1] - r[0]
    res["CN_NaCl"] = float(4 * np.pi * rho * np.sum(g[:imin] * r[:imin]**2) * dr); res["rmin_NaCl"] = float(r[imin])
    out[tag] = res
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k.startswith(("D_", "CN", "peak", "n"))}, meta["ok"])
json.dump(out, open(sys.argv[2], "w"))
