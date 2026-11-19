"""Main data figures 2-6 and SI figures, unified style (pstyle). Usage: python figs_v3.py DATA_DIR FIGSRC_DIR OUT_DIR
Fig 2 classical (pair + TIP3P), Fig 3 DFT screening, Fig 4 scan/projection/confirmation, Fig 5 comparisons, Fig 6 MD;
SI: figS_density, figS_explained, figS_lc_energy, figS_tip3p."""
import json, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator, FuncFormatter
from pstyle import FULL, HALF, INK, MUTE, FAINT, C

D, S, O = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(O, exist_ok=True)
J = lambda p: json.load(open(os.path.join(D, p)))
seeds = ["20260850", "20260851", "20260852"]
def ms(v): v = np.array(v, float); return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
def save(fig, name): fig.savefig(f"{O}/{name}.pdf"); fig.savefig(f"{O}/{name}.png", dpi=300); plt.close(fig)
def letters(fig, axes, labs="abcdefgh", dx=0.0, dy=0.0, lift=0.13):
    """Bold letters left of each axes' y-label, 'lift' inches above the axes top (same height for axes in one row)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); H = fig.get_figheight(); xs_ = []
    for ax, t in zip(axes, labs):
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted()); yl = ax.yaxis.label.get_window_extent(r).transformed(fig.transFigure.inverted())
        tl = [l.get_window_extent(r).transformed(fig.transFigure.inverted()).x0 for l in ax.get_yticklabels() if l.get_text()]
        xs_.append(min([yl.x0] + tl) if ax.yaxis.label.get_text() else ax.get_position().x0)
    for i, (ax, t) in enumerate(zip(axes, labs)):
        col = [xs_[j] for j, a2 in enumerate(axes[:len(xs_)]) if abs(a2.get_position().x0 - ax.get_position().x0) < 0.03]
        row = [a2.get_position().y1 for a2 in axes if abs(a2.get_position().y1 - ax.get_position().y1) < 0.12]
        fig.text(min(col) - 0.006 + dx, max(row) + lift / H + dy, t, fontsize=8, weight="bold", color=INK, va="bottom", ha="left")
EB = dict(elinewidth=0.6, capthick=0.6)
UNT, RES, BAS, HAM, LES, LESI, FORM, REF = (C[k] for k in ["untransformed", "restored", "baseline", "hamiltonian", "les", "les_ips", "formal", "reference"])
SKY = "#56B4E9"; NAVY = "#003F6B"
tex = lambda k: {"ss11": "11 chlorides", "ssmgna": "NaCl–MgCl$_2$", "water": "Water"}.get(k, k.replace("Cl2", "Cl$_2$"))
# system key (scan and projection only)
SYS = {"NaCl": ("#000000", "o", "-"), "CsCl": ("#E69F00", "s", "-"), "BaCl2": ("#009E73", "^", "-"), "SrCl2": ("#CC79A7", "D", "-"),
       "ss11": ("#8B5A2B", "v", "--"), "ssmgna": ("#7A7A7A", "P", "--"), "water": (SKY, "o", ":")}

# ---------------- data ----------------
nf = J("lamscan/nacl_final.json"); wf = J("wscan/water_final.json")
scan = {"NaCl": {l: ms([nf["runs"][f"L{l}|{s}"]["test"]["F"] for s in seeds]) for l in ["0", "0.2", "0.3", "0.4", "0.5", "0.6"]},
        "water": {l: ms([wf["runs"][f"W{l}|{s}"]["test"]["F"] for s in seeds]) for l in ["0", "0.5", "1.0"]}}
scan["water"]["0.75"] = (wf["runs"]["W0.75|20260850"]["test"]["F"], 0.0)
expl = {"NaCl": ms([nf["proj"][f"Ldelta|{s}"]["test"]["explained"] for s in seeds]),
        "water": ms([wf["proj"][f"Wdelta|{s}"]["test"]["explained"] for s in seeds]),
        "perm": ms([nf["proj"][f"Ldelta_scr|{s}"]["test"]["explained"] for s in seeds])}
for salt in ["CsCl", "BaCl2", "SrCl2"]:
    d = J(f"ss/{salt}_scan.json")
    scan[salt] = {l: (d["runs"][f"L{l}_scan|20260850"]["test"]["F"], 0.0) for l in ["0", "0.3", "0.5", "0.7"]}
    expl[salt] = (d["proj"]["Ldelta_scan|20260850"]["test"]["explained"], 0.0)
mx = J("mix_scan.json")
for k in ["ss11", "ssmgna"]: scan[k] = {l: ms(v) for l, v in mx["test_F"][k].items()}
# scan minima (fit of Eq. 2, test) and three-training (IV) projection, test
vertex = {"NaCl": 0.39, "CsCl": 0.50, "BaCl2": 0.35, "SrCl2": 0.33, "ss11": 0.30, "ssmgna": 0.29}
piv = J("projIV.json")["test"]
iv = {"NaCl": ms([piv[f"r{i}"]["iv_mean"] for i in range(3)]), "CsCl": (0.508, 0.0), "BaCl2": (0.348, 0.0), "SrCl2": (0.337, 0.0),
      "ss11": ms(mx["proj"]["ss11"]["iv_test"]), "ssmgna": ms(mx["proj"]["ssmgna"]["iv_test"])}
fa = J("formA_eval.json"); baseF = ms([fa[f"formA/L0.5205_s{s}"]["test"]["F"] for s in seeds])
lc = J("lc_eval.json"); n6 = J("n640_eval.json"); lc.update({k: v for k, v in n6.items() if k.startswith("lc/")})
NS = [80, 160, 320, 640]
lc.update(J("lc44_eval.json"))
cF = {l: {n: ms([lc[f"lc/L{l}_n{n}_s{s}"]["test"]["F"] for s in seeds]) for n in NS} for l in ["0", "0.5205", "0.44"]}
cE = {l: {n: ms([lc[f"lc/L{l}_n{n}_s{s}"]["test"]["E"] for s in seeds]) for n in NS} for l in ["0", "0.5205", "0.44"]}
def grp(d, pat, ss=seeds):
    v = [d[pat.format(s)]["test"] for s in ss if pat.format(s) in d]; return ms([x["F"] for x in v]), ms([x["E"] for x in v]), len(v)
lf, alt, les5, ss3, rv, l2 = J("lesf_eval.json"), J("alt_eval.json"), J("les_eval.json"), J("ss3_eval.json"), J("rev_eval1.json"), J("les2_eval.json")
lesN = {}
for n in NS:
    allk = {**lf, **l2, **(J("lesf_n80_eval.json") if os.path.exists(f"{D}/lesf_n80_eval.json") else {}), **(J("lesf_lc_eval.json") if os.path.exists(f"{D}/lesf_lc_eval.json") else {})}
    g = grp(allk, f"lesf/L0_n{n}_s{{}}")
    if g[2] >= 2: lesN[n] = g

# ================= Fig 2: classical liquids =================
def scanfile(kind, seed): return f"{S}/experiments/runs/mace_nacl_pair_20260827/{kind}/R5/seed{seed}/pair_functional_evaluation/pair_scan.csv"
def rd(p):
    rows = [r for r in csv.DictReader(open(p)) if r["orientation"] == "x"]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ("orientation", "target_mode")}
raw = rd(scanfile("raw", 20260827)); ips = [rd(scanfile("ips", s)) for s in (20260827, 20260828)]
r = raw["distance_A"]; ipsE = np.mean([d["predicted_interaction_energy_eV"] for d in ips], 0); ipsF = np.mean([d["predicted_radial_force_eV_A"] for d in ips], 0)
fmtv = lambda v: f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2g}")
TD = json.load(open(f"{S}/paper/arxiv_nacl_pme_completion_20260901/figures/tip3p_direct_learning_data.json"))

fig = plt.figure(figsize=(FULL, 2.25))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.07, right=0.99, bottom=0.2, top=0.78, wspace=0.42)
ax = fig.add_subplot(gs[0])
ax.axvspan(5, 8, color=FAINT, lw=0, zorder=0)
resE = ipsE - (ips[0]["target_energy_eV"] - raw["target_energy_eV"]); resF = ipsF - (ips[0]["target_radial_force_eV_A"] - raw["target_radial_force_eV_A"])
mk_ = np.unique(np.searchsorted(r, np.arange(2.3, 8.01, 0.3)))
ax.plot(r, ips[0]["target_energy_eV"], color=RES, lw=0.8, ls=(0, (1, 1.2)), alpha=0.8, label="IPS target (learned)")
ax.plot(r, raw["predicted_interaction_energy_eV"], color=UNT, lw=1.3, label="MACE")
ax.plot(r, resE, color=RES, lw=1.3, label="MACE + IPS")
ax.plot(r[mk_], raw["target_energy_eV"][mk_], ls="none", marker="o", ms=2.6, mfc="white", mec=INK, mew=0.6, label="reference (Coulomb + LJ)", zorder=4)
ax.axvline(5, color=MUTE, lw=0.6, ls=(0, (2, 2)))
ax.set_xlim(3, 8); ax.set_ylim(-4.6, 0.5); ax.set_xlabel("Na–Cl distance (Å)"); ax.set_ylabel("Interaction energy (eV)")
ax.text(7.9, -4.45, "beyond $R_\\mathrm{c}$", ha="right", va="bottom", fontsize=5.8, color=MUTE)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=4, fontsize=5.8, columnspacing=1.2, handlelength=2.2)
ax = fig.add_subplot(gs[1])
mk2 = np.unique(np.searchsorted(r, np.arange(4.0, 5.51, 0.1)))
ax.axvspan(5, 5.5, color=FAINT, lw=0, zorder=0)
ax.plot(r, ips[0]["target_radial_force_eV_A"], color=RES, lw=0.8, ls=(0, (1, 1.2)), alpha=0.8)
ax.plot(r, raw["predicted_radial_force_eV_A"], color=UNT, lw=1.3)
ax.plot(r, resF, color=RES, lw=1.3)
ax.plot(r[mk2], raw["target_radial_force_eV_A"][mk2], ls="none", marker="o", ms=2.6, mfc="white", mec=INK, mew=0.6, zorder=4)
ax.axvline(5, color=MUTE, lw=0.6, ls=(0, (2, 2)))
m = (r >= 4) & (r <= 5.5); yy = np.concatenate([raw["predicted_radial_force_eV_A"][m], raw["target_radial_force_eV_A"][m], resF[m], ips[0]["target_radial_force_eV_A"][m]])
ax.set_xlim(4, 5.5); ax.set_ylim(yy.min() - 0.1, yy.max() + 0.15); ax.set_xticks([4, 4.5, 5, 5.5])
ax.set_xlabel("Na–Cl distance (Å)"); ax.set_ylabel("Radial force on Na (eV Å$^{-1}$)")
ax.annotate("MACE:\nwrong force throughout,\noscillates near $R_\\mathrm{c}$", (4.7, raw["predicted_radial_force_eV_A"][np.searchsorted(r, 4.7)]), (4.05, 0.62),
            fontsize=5.8, color=UNT, va="top", arrowprops=dict(arrowstyle="-", lw=0.45, color=UNT))
ax = fig.add_subplot(gs[2])
q = [("atomic_force_rmse_eV_A", "Atomic\nforce", "meV Å$^{-1}$"), ("molecular_net_force_rmse_eV_A", "Molecular\nforce", "meV Å$^{-1}$"),
     ("molecular_torque_rmse_eV", "Torque", "meV"), ("energy_rmse_per_atom_centered_eV", "Energy", "meV atom$^{-1}$")]
x = np.arange(len(q)); w = 0.42
for j, (tgt, col, labl) in enumerate([("pme", UNT, "MACE"), ("ips5", RES, "MACE + IPS")]):
    vals = [TD[tgt]["overall"][k] * 1e3 for k, _, _ in q]
    ax.bar(x + (j - 0.5) * w, vals, w * 0.86, color=col, label=labl)
    for xi, v in zip(x + (j - 0.5) * w, vals): ax.text(xi + 0.05 * (j - 0.5) * 2, v * 1.12, fmtv(v), ha="center", va="bottom", fontsize=6, color=INK)
ax.set_yscale("log"); ax.set_ylim(0.005, 40); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator()); ax.set_xticks(x); ax.set_xticklabels([f"{a}\n({u})" for _, a, u in q], fontsize=5.6)
ax.set_ylabel("Held-out error (TIP3P water)"); ax.legend(loc="upper right", ncol=2, fontsize=5.8)
letters(fig, fig.axes, lift=0.30); save(fig, "fig2_classical")

# ================= Fig 3: DFT screening =================
rows = J("farfield/farfield_rows.json")
R = np.array([x["r"] for x in rows]); Q = np.array([x["kQ"] for x in rows]); Cc = np.array([x["kC"] for x in rows]); G = np.array([x["config"] * 1000 + x["ion"] for x in rows])
Dl = np.array([x["kD"] for x in rows]); cfg = np.array([x["config"] for x in rows]); ug = np.unique(G); rng = np.random.default_rng(0)
Lbox = {110: 15.104, 143: 13.727, 148: 15.529, 177: 14.797, 180: 15.785, 182: 14.35}
def slope(idx): qq = Q[idx].ravel(); cc = Cc[idx].ravel(); return qq @ cc / (cc @ cc)
def ci(mask, B=2000):
    v = []
    for _ in range(B):
        gsel = rng.choice(ug, len(ug)); idx = np.concatenate([np.where((G == g) & mask)[0] for g in gsel]); v.append(slope(idx))
    return np.percentile(v, [2.5, 97.5])
far = R > 5; lam = slope(np.where(far)[0])
fig = plt.figure(figsize=(FULL, 4.3))
gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.095, top=0.935, wspace=0.30, hspace=0.45)
ax = fig.add_subplot(gs[0, 0])
xx = Cc[far].ravel() * 1e3; yv = Q[far].ravel() * 1e3; lim = np.abs(xx).max() * 1.05; t = np.linspace(-lim, lim, 10)
ax.scatter(xx, yv, s=2.5, color="#7F7F7F", lw=0, alpha=0.55, rasterized=True)
ax.plot(t, t, ls=(0, (3.5, 2)), color=REF, lw=0.9)
ax.plot(t, lam * t, color=INK, lw=1.2)
ax.text(0.62 * lim, 0.62 * lim + 0.06 * lim, "formal charges", rotation=0, fontsize=6, color=REF, ha="right", va="bottom")
ax.annotate(f"DFT fit, slope {lam:.2f}", (0.78 * lim, lam * 0.78 * lim), (0.95 * lim, -0.45 * lim), fontsize=6, color=INK, ha="right", va="center", arrowprops=dict(arrowstyle="-", lw=0.45, color=INK))
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Coulomb response, formal charges (meV Å$^{-2}$)"); ax.set_ylabel("DFT response (meV Å$^{-2}$)")
ax.text(-0.95 * lim, 0.95 * lim, "1,012 ion pairs beyond 5 Å,\n6 melts; 3 Cartesian components", fontsize=5.8, color=MUTE, va="top")
ax = fig.add_subplot(gs[0, 1])
bins = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 12)]
ax.axhspan(0.40, 0.44, color=RES, alpha=0.13, lw=0)
ax.axhline(1.0, color=REF, lw=0.9, ls=(0, (3.5, 2))); ax.axhline(1 / 1.9213, color=MUTE, lw=0.7, ls=(0, (1, 1.5)))
for b in bins:
    mm = (R > b[0]) & (R <= b[1]); s_ = slope(np.where(mm)[0]); l_, h_ = ci(mm, 1000)
    ax.hlines(s_, b[0] + 0.06, b[1] - 0.06, color="#9A9A9A", lw=0.7, zorder=1)
    ax.errorbar(np.mean(b), s_, yerr=[[s_ - l_], [h_ - s_]], fmt="o", ms=3.4, color=INK, capsize=1.8, zorder=3, **EB)
ax.text(11.9, 1.0 - 0.025, "formal charges", ha="right", va="top", fontsize=6, color=REF)
ax.text(11.9, 1 / 1.9213 - 0.018, "$1/\\varepsilon_\\infty$ (finite field)", ha="right", va="top", fontsize=6, color=MUTE)
ax.text(4.25, 0.385, "minimum of the learning error", fontsize=6, color=RES, va="top")
ax.set_xlim(4, 12); ax.set_ylim(0.2, 1.08); ax.set_xlabel("Distance from displaced ion (Å)"); ax.set_ylabel("Screening strength $\\lambda_\\mathrm{far}$")
ax = fig.add_subplot(gs[1, 0])
edges = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 13.0]); mid = 0.5 * (edges[1:] + edges[:-1])
rms = lambda a: np.sqrt(np.mean(a ** 2)) * 1e3
prof = lambda A: np.array([rms(A[(R > lo) & (R <= hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
ax.axvspan(5, 13, color=FAINT, lw=0, zorder=0); ax.axvline(5, color=MUTE, lw=0.6, ls=(0, (2, 2)))
ax.plot(mid, prof(Cc), color=REF, ls=(0, (3.5, 2)), lw=1.0)
ax.plot(mid, prof(Q), "o-", color=INK, ms=2.8, lw=1.1)
ax.plot(mid, prof(Q + 1.0 * Dl), "s-", color=FORM, ms=2.6, lw=1.0, mfc="white", mew=0.8)
ax.plot(mid, prof(Q + lam * Dl), "o-", color=RES, ms=2.8, lw=1.2)
ax.set_yscale("log"); ax.set_xlim(2.5, 13); ax.set_ylim(1.5, 2000)
ax.set_xlabel("Distance from displaced ion (Å)"); ax.set_ylabel("RMS force response (meV Å$^{-2}$)")
for yv_, txt, col in [(prof(Cc)[-1] * 1.25, "bare Coulomb, formal charges", REF), (prof(Q)[-1] * 1.1, "DFT reference", INK),
                      (prof(Q + Dl)[-1] * 0.62, "IPS target, formal ($\\lambda=1$)", FORM), (prof(Q + lam * Dl)[-1] * 0.62, f"IPS target, screened ($\\lambda={lam:.2f}$)", RES)]:
    pass
ax.legend(handles=[Line2D([], [], color=REF, ls=(0, (3.5, 2)), lw=1.0, label="bare Coulomb, formal charges"),
                   Line2D([], [], color=INK, marker="o", ms=2.8, lw=1.1, label="DFT reference"),
                   Line2D([], [], color=FORM, marker="s", ms=2.6, mfc="white", lw=1.0, label="IPS target, formal ($\\lambda=1$)"),
                   Line2D([], [], color=RES, marker="o", ms=2.8, lw=1.2, label=f"IPS target, screened ($\\lambda={lam:.2f}$)")],
          loc="upper right", fontsize=5.6)
ax.text(5.15, 2.6, "$R_\\mathrm{c}$", fontsize=6, color=MUTE)
ax = fig.add_subplot(gs[1, 1])
lg = np.linspace(0, 1.05, 50); ax.axhline(0, color="#BBBBBB", lw=0.5)
ax.plot(lg, lam - lg, color=INK, lw=1.2)
p = J("proj_scan.json")["test"]
for l in ["0", "0.2", "0.3", "0.4", "0.5", "0.6"]:
    v = [p[k]["iv_mean"] for k in p if k == f"lamscan/L{l}" or k.startswith(f"lamscan/L{l}_s")]
    ax.errorbar(float(l), np.mean(v), yerr=np.std(v, ddof=1) if len(v) > 1 else 0, fmt="o", ms=3.4, color=RES, mec="white", mew=0.4, capsize=1.2, zorder=3, **EB)
ax.plot([1.0], [lam - 1.0], "x", ms=5, color=FORM, mew=1.1)
ax.annotate("formal correction:\nwrong-sign remainder", (0.985, lam - 1.0), (0.80, -0.56), ha="right", va="center", fontsize=5.8, color=FORM, arrowprops=dict(arrowstyle="-", lw=0.45, color=FORM))
ax.text(0.05, lam - 0.05 + 0.04, "DFT: $\\lambda_\\mathrm{far}-\\lambda$", fontsize=6, color=INK, va="bottom")
ax.annotate("learned models: $\\lambda\\!'-\\lambda$", (0.2, 0.205), (0.03, -0.28), fontsize=6, color=RES, va="center", arrowprops=dict(arrowstyle="-", lw=0.45, color=RES))
ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.66, 0.68); ax.set_xlabel("Correction strength $\\lambda$"); ax.set_ylabel("Long-range coupling left in $T_\\lambda$")
letters(fig, fig.axes); save(fig, "fig3_dft")
# SI: density dependence
fig, ax = plt.subplots(figsize=(HALF, 2.3), layout="constrained")
for c_, Lc in Lbox.items():
    mm = far & (cfg == c_); rho = 4.852e-21 / (Lc ** 3 * 1e-24); ax.plot(rho, slope(np.where(mm)[0]), "o", ms=3.6, color=INK)
ax.plot([1.5], [1 / 1.9213], "^", ms=4.2, mfc="white", mec=MUTE, mew=0.8, label="$1/\\varepsilon_{\\infty}$ (finite field)")
ax.axhline(lam, color=INK, lw=0.6, ls=(0, (3, 2)), label=f"all configurations, {lam:.2f}")
ax.set_xlabel("Density (g cm$^{-3}$)"); ax.set_ylabel("$\\lambda_\\mathrm{far}$ per configuration"); ax.set_ylim(0.45, 0.65); ax.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65]); ax.legend(loc="upper right")
save(fig, "figS_density")

# ================= Fig 4: scan, projection, confirmation =================
fig = plt.figure(figsize=(FULL, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1.25], left=0.065, right=0.99, bottom=0.22, top=0.80, wspace=0.40)
ax = fig.add_subplot(gs[0])
order = ["NaCl", "CsCl", "BaCl2", "SrCl2", "ss11", "ssmgna", "water"]
for k in order:
    col, mk, lsty = SYS[k]; ls_ = sorted(scan[k], key=float)
    xv = np.array([float(l) for l in ls_]); yv = np.array([scan[k][l][0] for l in ls_]); ev = np.array([scan[k][l][1] for l in ls_])
    ax.errorbar(xv, yv / yv[0], yerr=ev / yv[0], marker=mk, ms=3.0 if k != "NaCl" else 3.4, lw=1.5 if k == "NaCl" else 0.9, ls=lsty, color=col,
                mfc="white" if k in ("water", "ssmgna") else col, capsize=1.2, label=tex(k), zorder=3 if k == "NaCl" else 2, **EB)
ax.errorbar([0.5205], [baseF[0] / scan["NaCl"]["0"][0]], yerr=[baseF[1] / scan["NaCl"]["0"][0]], marker="D", ms=3.4, mfc="white", mec=BAS, mew=1.0,
            color=BAS, capsize=1.2, ls="none", label="NaCl, baseline", zorder=4, **EB)
ax.axhline(1, color="#BBBBBB", lw=0.5, zorder=0)
ax.set_xlabel("Correction strength $\\lambda$"); ax.set_ylabel("Force RMSE relative to $\\lambda=0$")
ax.set_ylim(0.45, 1.18); ax.set_xlim(-0.03, 1.03); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=5.6, columnspacing=0.9, handlelength=1.6)
ax = fig.add_subplot(gs[1])
ax.plot([0.2, 0.6], [0.2, 0.6], color="#BBBBBB", lw=0.6, zorder=0); ax.text(0.59, 0.555, "1:1", fontsize=5.8, color=MUTE, ha="right", va="top")
for k in ["NaCl", "CsCl", "BaCl2", "SrCl2", "ss11", "ssmgna"]:
    col, mk, _ = SYS[k]
    ax.errorbar(vertex[k], iv[k][0], yerr=iv[k][1], marker=mk, ms=3.6, color=col, mfc="white" if k == "ssmgna" else col, ls="none", capsize=1.2, **EB)
lab_pos = {"NaCl": (0.405, 0.392, "left", False), "CsCl": (0.487, 0.517, "right", False), "BaCl2": (0.365, 0.350, "left", False),
           "SrCl2": (0.300, 0.392, "right", True), "ss11": (0.318, 0.272, "left", True), "ssmgna": (0.278, 0.355, "right", True)}
for k, (tx, ty, ha_, lead) in lab_pos.items():
    if lead: ax.annotate("NaCl–\nMgCl$_2$" if k == "ssmgna" else tex(k), (vertex[k], iv[k][0]), (tx, ty), fontsize=5.5, color=SYS[k][0], ha=ha_, va="center",
                         arrowprops=dict(arrowstyle="-", lw=0.4, color=SYS[k][0], shrinkA=1, shrinkB=2.5))
    else: ax.text(tx, ty, tex(k), fontsize=5.5, color=SYS[k][0], ha=ha_, va="center")
ax.set_xlim(0.22, 0.6); ax.set_ylim(0.22, 0.6); ax.set_aspect("equal")
ax.set_xlabel("Scan minimum, $\\lambda_0$"); ax.set_ylabel("Three-training estimate, $\\lambda\\!'$")
ax = fig.add_subplot(gs[2])
rowsc = [("NaCl", cF["0"][640], cF["0.5205"][640], "0.52")]
for salt, l in [("CsCl", "0.5"), ("BaCl2", "0.3"), ("SrCl2", "0.3")]:
    rowsc.append((salt, ms([ss3[f"ss3/{salt}/L0_s{s}"]["test"]["F"] for s in seeds]), ms([ss3[f"ss3/{salt}/L{l}_s{s}"]["test"]["F"] for s in seeds]), l))
rowsc.append(("water", ms([wf["runs"][f"W0|{s}"]["test"]["F"] for s in seeds]), ms([wf["runs"][f"W0.5|{s}"]["test"]["F"] for s in seeds]), "0.5"))
x = np.arange(len(rowsc)); w = 0.38
ax.bar(x - w / 2, [r_[1][0] for r_ in rowsc], w * 0.95, yerr=[r_[1][1] for r_ in rowsc], color=UNT, capsize=1.2, error_kw=EB, label="MACE")
ax.bar(x + w / 2, [r_[2][0] for r_ in rowsc], w * 0.95, yerr=[r_[2][1] for r_ in rowsc], color=RES, capsize=1.2, error_kw=EB, label="MACE + screened IPS",
       hatch=[None] * 4 + ["/////"], edgecolor=["none"] * 4 + ["white"], lw=0)
for i, r_ in enumerate(rowsc):
    ax.text(i, max(r_[1][0] + r_[1][1], r_[2][0] + r_[2][1]) + 2.0, "−%.0f%%" % (100 * (1 - r_[2][0] / r_[1][0])), ha="center", va="bottom", fontsize=6, color=INK)
ax.set_xticks(x); ax.set_xticklabels([f"{tex(r_[0])}\n$\\lambda$ = {r_[3]}" for r_ in rowsc], fontsize=5.8)
ax.set_ylabel("Test force RMSE (meV Å$^{-1}$)"); ax.set_ylim(0, 80)
ax.legend(handles=[Patch(color=UNT, label="MACE"), Patch(color=RES, label="MACE + screened IPS"), Patch(facecolor=RES, hatch="/////", edgecolor="white", label="exploratory")],
          loc="upper right", fontsize=5.6)
letters(fig, fig.axes, dy=0.0); save(fig, "fig4_scan")
# SI: explained fraction
fig, ax = plt.subplots(figsize=(HALF, 2.2), layout="constrained")
keys = ["NaCl", "CsCl", "BaCl2", "SrCl2", "water", "perm"]
vals = [100 * expl[k][0] for k in keys]
ax.bar(range(len(keys)), vals, yerr=[100 * expl[k][1] if expl[k][1] > 0 else np.nan for k in keys], width=0.62, capsize=1.2, error_kw=EB, color=[UNT] * 5 + ["#D0D0D0"])
ax.text(5, 2, "0%", ha="center", fontsize=6)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([tex(k) for k in keys[:5]] + ["NaCl,\npermuted"], fontsize=5.8)
ax.set_ylabel("Residual variance explained by $\\mathbf{N}$ (%)"); save(fig, "figS_explained")

def gapline(ax, nn, ys, es, **kw):
    """Markers at all available points; connect only consecutive members of NS."""
    lab = kw.pop("label", None); ax.errorbar(nn, ys, yerr=es, ls="none", label=lab, **kw)
    base = [min(NS, key=lambda m: abs(np.log(m) - np.log(v))) for v in nn]
    for k_ in range(len(nn) - 1):
        if NS.index(base[k_ + 1]) - NS.index(base[k_]) == 1: ax.plot(nn[k_:k_ + 2], ys[k_:k_ + 2], color=kw.get("color"), lw=1.1)

# ================= Fig 5: comparisons =================
fig = plt.figure(figsize=(FULL, 4.25))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], left=0.07, right=0.99, bottom=0.115, top=0.92, wspace=0.42, hspace=0.62)
gsb = gs[1, :].subgridspec(1, 2, wspace=0.22)
ax = fig.add_subplot(gs[0, 0:2])
k52 = [("MACE", scan["NaCl"]["0"], None, UNT), ("baseline", baseF, None, BAS),
       ("local control", grp(rv, "rev/plus0.52_s{}")[0], None, "LOCAL"),
       ("shifted force", grp(alt, "alt/sf_s{}")[0], None, None), ("DSF", grp(alt, "alt/dsf_s{}")[0], grp(rv, "rev/dsf0.4_s{}")[0], None),
       ("Δ-learning (Ewald)", grp(alt, "alt/noewald_s{}")[0], grp(rv, "rev/dl0.4_s{}")[0], None),
       ("IPS", scan["NaCl"]["0.5"], scan["NaCl"]["0.4"], None), ("MACE + LES", grp(les5, "les/L0_s{}")[0], None, LES)]
w = 0.38
for i, (nm, a52, a40, col) in enumerate(k52):
    if a40 is None:
        if col == "LOCAL": ax.bar(i, a52[0], 0.6, yerr=a52[1], facecolor="white", edgecolor=BAS, lw=0.9, capsize=1.2, error_kw=EB)
        else: ax.bar(i, a52[0], 0.6, yerr=a52[1], color=col or RES, capsize=1.2, error_kw=EB)
    else:
        ax.bar(i - w / 2, a52[0], w * 0.95, yerr=a52[1], color=RES, capsize=1.2, error_kw=EB)
        ax.bar(i + w / 2, a40[0], w * 0.95, yerr=a40[1], color=NAVY, capsize=1.2, error_kw=EB)
ax.axhline(scan["NaCl"]["0"][0], color=UNT, lw=0.6, ls=(0, (3, 2)), zorder=0)
ax.set_xticks(range(len(k52))); ax.set_xticklabels([k[0].replace(" (", "\n(") if "(" in k[0] else (k[0].replace(" + ", " +\n") if " + " in k[0] else k[0].replace(" ", "\n", 1)) for k in k52], fontsize=5.8)
ax.set_ylabel("Test force RMSE (meV Å$^{-1}$)"); ax.set_ylim(0, 23.5)
ax.legend(handles=[Patch(color=RES, label="$\\lambda\\approx0.52$"), Patch(color=NAVY, label="$\\lambda=0.4$")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8)
ax.text(0.0, 1.06, "5,000 updates", transform=ax.transAxes, fontsize=6, color=MUTE, va="bottom")
ax = fig.add_subplot(gs[0, 2])
n = np.array(NS)
ax.errorbar(n, [cF["0"][i][0] for i in NS], yerr=[cF["0"][i][1] for i in NS], marker="o", ms=3, color=UNT, capsize=1.2, label="MACE", **EB)
ax.errorbar(n, [cF["0.5205"][i][0] for i in NS], yerr=[cF["0.5205"][i][1] for i in NS], marker="o", ms=3, color=RES, capsize=1.2, label="MACE + screened IPS, $\\lambda=0.52$", **EB)
ax.errorbar(n, [cF["0.44"][i][0] for i in NS], yerr=[cF["0.44"][i][1] for i in NS], marker="s", ms=2.8, color=NAVY, capsize=1.2, label="MACE + screened IPS, $\\lambda=0.44$", **EB)
if lesN:
    nn = sorted(lesN); gapline(ax, nn, [lesN[i][0][0] for i in nn], [lesN[i][0][1] for i in nn], marker="D", ms=2.8, color=LES, capsize=1.2, label="MACE + LES", **EB)
ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks(NS); ax.set_xticklabels([str(i) for i in NS])
ax.yaxis.set_minor_locator(NullLocator()); ax.set_yticks([8, 10, 15, 20, 30]); ax.set_yticklabels(["8", "10", "15", "20", "30"]); ax.set_ylim(7.5, 66)
ax.set_xlabel("Training configurations"); ax.set_ylabel("Test force RMSE (meV Å$^{-1}$)"); ax.legend(loc="upper right", fontsize=5.6)
ax.text(0.0, 1.06, "20,000 updates", transform=ax.transAxes, fontsize=6, color=MUTE, va="bottom")
full = [("MACE", grp(n6, "lc/L0_n640_s{}"), UNT, None), ("screened IPS\n$\\lambda$ = 0.52", grp(n6, "lc/L0.5205_n640_s{}"), RES, None),
        ("screened IPS\n$\\lambda$ = 0.44", grp(lc, "lc/L0.44_n640_s{}")[:2], RES, "/////"), ("MACE +\nLES", grp(lf, "lesf/L0_n640_s{}"), LES, None),
        ("MACE + LES\n+ screened IPS", grp(l2, "lesf2/L0.4_n640_s{}")[:2], LESI, "/////")]
for j, (yl, kk) in enumerate([("Test force RMSE (meV Å$^{-1}$)", 0), ("Test energy RMSE (meV atom$^{-1}$)", 1)]):
    ax = fig.add_subplot(gsb[0, j])
    for i, (nm, g, col, h) in enumerate(full):
        ax.bar(i, g[kk][0], 0.66, yerr=g[kk][1], color=col, capsize=1.2, error_kw=EB, hatch=h, edgecolor="white" if h else "none", lw=0)
    ax.set_xticks(range(len(full))); ax.set_xticklabels([f[0] for f in full], fontsize=5.8); ax.set_ylabel(yl)
    if j == 0: ax.text(0.0, 1.06, "640 configurations, 20,000 updates", transform=ax.transAxes, fontsize=6, color=MUTE, va="bottom")
fig.axes[-1].set_ylim(0, 4.0); fig.axes[-2].set_ylim(0, 19)
fig.axes[-2].legend(handles=[Patch(color=UNT, label="MACE"), Patch(color=RES, label="MACE + screened IPS"), Patch(color=LES, label="MACE + LES"),
                    Patch(color=LESI, label="MACE + LES + screened IPS"), Patch(facecolor="#BBBBBB", hatch="/////", edgecolor="white", label="exploratory")],
           loc="upper right", ncol=2, fontsize=5.8, handlelength=1.4, columnspacing=1.0)
letters(fig, fig.axes[:4]); save(fig, "fig5_methods")
# SI: learning-curve energies
fig, ax = plt.subplots(figsize=(HALF, 2.2), layout="constrained")
ax.errorbar(n, [cE["0"][i][0] for i in NS], yerr=[cE["0"][i][1] for i in NS], marker="o", ms=3, color=UNT, capsize=1.2, label="MACE", **EB)
ax.errorbar(n, [cE["0.5205"][i][0] for i in NS], yerr=[cE["0.5205"][i][1] for i in NS], marker="o", ms=3, color=RES, capsize=1.2, label="MACE + screened IPS, $\\lambda=0.52$", **EB)
ax.errorbar(n, [cE["0.44"][i][0] for i in NS], yerr=[cE["0.44"][i][1] for i in NS], marker="s", ms=2.8, color=NAVY, capsize=1.2, label="MACE + screened IPS, $\\lambda=0.44$", **EB)
if lesN:
    nn = sorted(lesN); gapline(ax, [x_ * 10**0.012 for x_ in nn], [lesN[i][1][0] for i in nn], [lesN[i][1][1] for i in nn], marker="D", ms=2.8, color=LES, capsize=1.2, label="MACE + LES", **EB)
ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticks(NS); ax.set_xticklabels([str(i) for i in NS])
ax.set_xlabel("Training configurations"); ax.set_ylabel("Test energy RMSE (meV atom$^{-1}$)"); ax.legend(loc="upper right"); save(fig, "figS_lc_energy")

# ================= Fig 6: molecular dynamics =================
def md(tag):
    d = J(f"md/{tag}_an.json"); reps = [v for v in d.values() if "g_NaCl" in v]
    return (np.array(reps[0]["r"]), {g: np.mean([r_[g] for r_ in reps], 0) for g in ["g_NaCl", "g_NaNa", "g_ClCl"]},
            {qq: ms([r_[qq] for r_ in reps]) for qq in ["D_Na_1e-9m2s", "D_Cl_1e-9m2s", "CN_NaCl"]})
rg, gU, qU = md("M0"); _, gR, qR = md("MA"); _, gB, qB = md("FA")
# --- SI: local structure and diffusion (unchanged) ---
fig, axs = plt.subplots(1, 2, figsize=(FULL * 0.75, 2.1), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
for g_, lsty in [("g_NaCl", "-"), ("g_NaNa", (0, (3.5, 1.5))), ("g_ClCl", (0, (1, 1.2)))]:
    ax.plot(rg, gU[g_], color=UNT, ls=lsty, lw=1.8); ax.plot(rg, gR[g_], color=RES, ls=lsty, lw=0.9); ax.plot(rg, gB[g_], color=BAS, ls=lsty, lw=0.7)
ax.legend(handles=[Line2D([], [], color=UNT, lw=1.8, label="MACE"), Line2D([], [], color=RES, lw=0.9, label="MACE + screened IPS"),
                   Line2D([], [], color=BAS, lw=0.7, label="baseline"), Line2D([], [], color=INK, lw=0.8, ls="-", label="Na–Cl"),
                   Line2D([], [], color=INK, lw=0.8, ls=(0, (3.5, 1.5)), label="Na–Na"), Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.2)), label="Cl–Cl")],
          loc="upper right", fontsize=5.8, ncol=2, columnspacing=1.0, handlelength=2.2)
ax.set_xlabel("$r$ (Å)"); ax.set_ylabel("$g(r)$"); ax.set_xlim(1.8, 7.5); ax.set_ylim(0, 4.6)
ax = axs[1]
for i, (qq, col, labl) in enumerate([(qU, UNT, "MACE"), (qR, RES, "MACE + screened IPS"), (qB, BAS, "baseline")]):
    for j, k in enumerate(["D_Na_1e-9m2s", "D_Cl_1e-9m2s"]):
        ax.errorbar(j + (i - 1) * 0.18, qq[k][0], yerr=qq[k][1], marker="D" if labl == "baseline" else "o", ms=3.4, color=col, mfc="white" if labl == "baseline" else col, mew=1.0, capsize=1.5, label=labl if j == 0 else None, **EB)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Na", "Cl"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(3, 7.5)
ax.set_ylabel("Self-diffusion coefficient (10$^{-9}$ m$^2$ s$^{-1}$)"); ax.legend(loc="upper right", fontsize=5.8)
letters(fig, axs); save(fig, "figS_md_local")

# --- Fig 6: long-wavelength charge physics ---
fig = plt.figure(figsize=(FULL, 4.3))
gs = fig.add_gridspec(2, 6, left=0.09, right=0.985, bottom=0.095, top=0.93, wspace=1.7, hspace=0.50)
ax = fig.add_subplot(gs[1, 4:6])
for i, tag in enumerate(["CH_r1", "CH_r2", "CH_r3"]):
    rr = J(f"md/{tag}.json"); t = np.array([x_["step"] for x_ in rr]) * 2e-3; sp = np.array([x_["spread"] for x_ in rr]) * 1e3
    ax.plot(t, sp, color=HAM, lw=0.9, label="IPS Hamiltonian" if i == 0 else None)
    ax.plot(t[-1], min(sp[-1], 1200), "x", color=HAM, ms=4, mew=1.0, clip_on=False)
rr = J("md/CR_r1.json"); t = np.array([x_["step"] for x_ in rr]) * 2e-3; sp = np.array([x_["spread"] for x_ in rr]) * 1e3
ax.plot(t, sp, color=RES, lw=0.9, label="MACE + screened IPS")
ax.set_yscale("log"); ax.set_ylim(2, 1500); ax.set_xlim(0, 5); ax.set_xlabel("Time (ps)"); ax.set_ylabel("Force s.d. across 3 models (meV Å$^{-1}$)")
ax.text(1.95, 900, "× failed", fontsize=5.8, color=HAM, va="center"); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=5.6)
ax = fig.add_subplot(gs[0, 0:3]); sz = J("md/szz_100.json")
if os.path.exists(f"{D}/md/szz_800b.json"): sz["B6"] = J("md/szz_800b.json")["B6"]
if os.path.exists(f"{D}/md/szz_800c.json"): sz["B7"] = J("md/szz_800c.json")["B7"]
kk_ = np.linspace(0, 0.62, 100); dens = 100 / 15.4**3; kB = 8.617333e-5 * 1200
kD2s = 4 * np.pi * 0.5205 * 14.3996 * dens / kB; kD2f = 4 * np.pi * 1.0 * 14.3996 * dens / kB
ax.plot(kk_, kk_**2 / kD2s, color=INK, lw=0.8, ls=(0, (1, 1.2)), label="SL, screened")
ax.plot(kk_, kk_**2 / kD2f, color=REF, lw=0.7, ls=(0, (3.5, 2)), label="SL, formal")
for tag, col, mk, labl, dx_ in [("M0", UNT, "o", "MACE", -0.006), ("FA", BAS, "D", "baseline model", 0.0), ("MA", RES, "o", "MACE + screened IPS", 0.006),
                                 ("B0", UNT, "^", None, -0.004), ("BR", RES, "^", None, 0.004), ("B6", INK, "v", "MACE, 6 Å cutoff", 0.0), ("B7", INK, "<", "MACE, 7 Å cutoff", 0.0)]:
    if tag not in sz: continue
    k_ = np.array(sz[tag]["k"]); S_ = np.array(sz[tag]["S"]); mm = k_ <= 0.62
    ax.plot(k_[mm] + dx_, S_[mm], mk, ms=2.8, color=col, mfc="white" if mk in ("D", "v") else col, mec=col, mew=0.8, label=labl, ls="none")
ax.set_xlim(0, 0.62); ax.set_ylim(0, 0.026); ax.set_xlabel("Wavevector $k$ (Å$^{-1}$)"); ax.set_ylabel("Charge structure factor $S_{ZZ}(k)$")
ax.text(0.33, 0.0248, "circles: 100 atoms\ntriangles: 800 atoms", fontsize=5.4, color=MUTE, ha="left", va="top")
ax.legend(loc="upper left", fontsize=5.2, ncol=1, handlelength=1.2)
# c: force-error spectrum
km = {}
for f_ in ["errk_mace.json", "errk_les.json", "errk_mace44.json", "errk_delta.json"]:
    if os.path.exists(f"{D}/kspec/{f_}"): km.update(J(f"kspec/{f_}"))
def kspec_rel(prefix, key="PL", bw=0.16):
    runs = [v["own"] for k, v in km.items() if k.startswith(prefix)]
    k = np.array(runs[0]["k"]); cnt = np.array(runs[0]["count"]); P = np.mean([r[key] for r in runs], 0)
    edges = np.arange(0.3, 3.3, bw); kc, Pc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (k >= lo) & (k < hi)
        if sel.any(): kc.append(np.average(k[sel], weights=cnt[sel])); Pc.append(np.average(P[sel], weights=cnt[sel]))
    kc, Pc = np.array(kc), np.array(Pc); return kc, Pc / Pc[kc > 2.0].mean()
ax = fig.add_subplot(gs[0, 3:6])
for pre, lab, col, mk in [("lamscan/Ldelta_s", "$\\mathbf{N}$", INK, "x"), ("lc/L0_n640", "MACE", UNT, "o"),
                          ("lc/L0.5205_n640", "MACE + screened IPS, $\\lambda=0.52$", RES, "s"), ("lc/L0.44_n640", "MACE + screened IPS, $\\lambda=0.44$", NAVY, "v"), ("lesf/L0_n640", "MACE + LES", LES, "D")]:
    if any(k.startswith(pre) for k in km):
        k, P = kspec_rel(pre); ax.plot(k, P, marker=mk, ms=2.4, lw=0.9, color=col, label=lab)
k, P = kspec_rel("lc/L0_n640", "PT"); ax.plot(k, P, color=UNT, lw=0.8, ls=(0, (2, 1.5)), label="MACE, transverse")
ax.axhline(1, color="#BBBBBB", lw=0.5); ax.set_yscale("log"); ax.set_xlim(0.3, 3.2); ax.set_ylim(0.6, 14)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Wavevector $k$ (Å$^{-1}$)"); ax.set_ylabel("Force-error spectrum, relative")
ax.legend(loc="upper right", fontsize=5.2, handlelength=1.4)
# d,e: charge-current spectra
cc = {}
for f_ in sorted(os.listdir(f"{D}/kspec")):
    if f_.startswith("ccur_"): cc.update(J(f"kspec/{f_}"))
GRP = [("MACE", "/NU_", UNT), ("MACE, 6 Å cutoff", "/N6_", INK), ("MACE + screened IPS", "/NR_", RES), ("MACE + LES", "/NL_", LES)]
smo = lambda y, w=5: np.convolve(y, np.ones(w) / w, mode="same")
ax = fig.add_subplot(gs[1, 0:2])
for lab, tag, col in GRP:
    ks = [k for k in cc if tag in k]
    if not ks: continue
    f = np.array(cc[ks[0]]["1"]["f"]); S = smo(np.mean([cc[k]["1"]["SL"] for k in ks], 0)); Sn = S / np.trapezoid(np.clip(S, 0, None), f)
    ax.plot(f, Sn, color=col, lw=1.1 if tag != "/N6_" else 0.9, ls="-" if tag != "/N6_" else (0, (3, 1.5)), label=lab)
ax.set_xlim(0, 12); ax.set_ylim(0, 0.47); ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("Longitudinal charge-current spectrum\n(normalized), $k=0.20$ Å$^{-1}$")
ax.legend(loc="upper right", fontsize=5.6)
ax = fig.add_subplot(gs[1, 2:4])
for lab, tag, col in GRP:
    ks = [k for k in cc if tag in k]
    if not ks: continue
    kk, fL, fLe = [], [], []
    for n in ["1", "2", "3", "4"]:
        f = np.array(cc[ks[0]][n]["f"]); s = f > 0.5
        pk = [f[s][np.argmax(smo(np.array(cc[k][n]["SL"]))[s])] for k in ks]
        kk.append(cc[ks[0]][n]["k"]); fL.append(np.mean(pk)); fLe.append(np.std(pk, ddof=1) / np.sqrt(len(pk)) if len(pk) > 1 else 0)
    ax.errorbar(kk, fL, yerr=fLe, marker="o" if tag != "/N6_" else "v", ms=3, color=col, mfc=col if tag != "/N6_" else "white", capsize=1.2, label=lab, ls="-" if tag != "/N6_" else (0, (3, 1.5)), **EB)
ax.set_xlabel("Wavevector $k$ (Å$^{-1}$)"); ax.set_ylabel("Longitudinal peak frequency (THz)"); ax.set_ylim(3, 8.6); ax.set_xlim(0.15, 0.86)
ax.legend(loc="upper right", fontsize=5.6)
A_ = fig.axes; letters(fig, [A_[1], A_[2], A_[3], A_[4], A_[0]]); save(fig, "fig6_md")

# SI: TIP3P geometry-source check
fig, ax = plt.subplots(figsize=(HALF, 2.1), layout="constrained")
for j_, (tgt, col, labl) in enumerate([("pme", UNT, "MACE"), ("ips5", RES, "MACE + IPS")]):
    vals = {t_["trajectory"]["source_origin"]: t_["atomic_force_rmse_eV_A"] * 1e3 for t_ in TD[tgt]["trajectories"]}
    xs2 = np.arange(2) + (j_ - 0.5) * 0.38; v2 = [vals["pme"], vals["ips5"]]
    ax.bar(xs2, v2, 0.34, color=col, label=labl)
    for xi, v in zip(xs2, v2): ax.text(xi, v + 0.3, fmtv(v), ha="center", va="bottom", fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["test geometries\nfrom PME MD", "test geometries\nfrom IPS MD"]); ax.set_ylabel("Atomic force RMSE (meV Å$^{-1}$)")
ax.set_ylim(0, 17); ax.legend(loc="upper left"); save(fig, "figS_tip3p")
print("lam_far", lam, "LES n:", sorted(lesN))
