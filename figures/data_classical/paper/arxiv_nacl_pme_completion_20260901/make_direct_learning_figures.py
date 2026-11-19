#!/usr/bin/env python3
"""Reproduce direct-learning figures from the portable held-out metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
DATA = json.loads((OUT / "tip3p_direct_learning_data.json").read_text())
DYN = json.loads((OUT / "tip3p_dynamics_contact_data.json").read_text())
COLORS = ["#34698b", "#bd6c25"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})
labels = ["PME-trained", "IPS5-trained"]

def bars(ax, keys, names, ylabel, title):
    for j, target in enumerate(("pme", "ips5")):
        ys = [DATA[target]["overall"][key] * 1000 for key in keys]
        xs = [i + (j - .5)*.34 for i in range(len(keys))]
        b = ax.bar(xs, ys, .32, label=labels[j], color=COLORS[j])
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(range(len(keys)), names)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1]*1.2)

fig, axs = plt.subplots(2, 2, figsize=(8, 6.6), layout="constrained")
bars(axs[0,0], ["atomic_force_rmse_eV_A", "molecular_net_force_rmse_eV_A"],
     ["Atomic", "Molecular net"], r"Force RMSE (meV / Å)", "(a) Force")
bars(axs[0,1], ["molecular_torque_rmse_eV"], ["Molecular COM"],
     "Torque RMSE (meV)", "(b) Torque")
bars(axs[1,0], ["energy_rmse_per_atom_centered_eV"], ["Pooled centering"],
     "Energy RMSE (meV / atom)", "(c) Energy")
ax = axs[1,1]
for j, target in enumerate(("pme", "ips5")):
    vals = {x["trajectory"]["source_origin"]: x["atomic_force_rmse_eV_A"]*1000
            for x in DATA[target]["trajectories"]}
    b = ax.bar([i+(j-.5)*.34 for i in range(2)],
               [vals["pme"], vals["ips5"]], .32, color=COLORS[j])
    ax.bar_label(b, fmt="%.3f", fontsize=8, padding=3)
ax.set_xticks([0,1], ["PME origin", "IPS5 origin"])
ax.set_ylabel(r"Force RMSE (meV / Å)")
ax.set_title("(d) Geometry-source check", loc="left", fontweight="bold")
ax.set_ylim(0, 17)
axs[0,0].legend(frameon=False, fontsize=9)
fig.suptitle("Same geometries and training budget; each model scored on its own target",
             fontsize=11)
fig.savefig(OUT / "tip3p_direct_learning.pdf")
fig.savefig(OUT / "tip3p_direct_learning.png", dpi=160)
plt.close(fig)

def box(ax, x, y, w, h, text, color="#eaf0f4", fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.015",
                               facecolor=color, edgecolor="#82909a", linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize)

def arrow(ax, a, b, color="#586570"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,
                                linewidth=1,color=color))

fig, ax = plt.subplots(figsize=(8,4.2))
ax.set(xlim=(0,10), ylim=(0,5))
ax.axis("off")
box(ax,.05,1.95,1.7,1.2,"Identical\nwater geometries")
box(ax,2.4,3.05,2.05,.85,"PME + LJ labels")
box(ax,2.4,1.2,2.05,.85,"IPS5 + LJ labels","#faeddf")
box(ax,5.1,3.05,2.05,.85,"PME-trained GNN")
box(ax,5.1,1.2,2.05,.85,"IPS5-trained GNN","#faeddf")
box(ax,7.7,3.05,2.2,.85,"Own-target error\n10.916 meV / Å")
box(ax,7.7,1.2,2.2,.85,"Own-target error\n1.040 meV / Å","#faeddf")
for y in (3.475,1.625):
    arrow(ax,(1.75,2.55),(2.4,y))
    arrow(ax,(4.45,y),(5.1,y))
    arrow(ax,(7.15,y),(7.7,y))
ax.text(6.1,4.4,"Same R5 architecture, data count and update budget",
        ha="center",fontsize=10,fontweight="bold")
ax.text(6.1,2.55,"Direct learning: no analytical add-back",
        ha="center",fontsize=10)
ax.text(5,.5,"Separate checks: classical IPS vs PME properties;\n"
                   "learned IPS5 vs parent properties (three completed replicas)",
        ha="center",va="center",fontsize=10,color="#48535b")
fig.tight_layout()
fig.savefig(OUT / "direct_learning_workflow.pdf")
fig.savefig(OUT / "direct_learning_workflow.png", dpi=160)
plt.close(fig)

# Learned-property inheritance and the failure-precursor contact scan.
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.15), layout="constrained")
for target, color, label in (("pme", COLORS[0], "PME-trained"),
                             ("ips5", COLORS[1], "IPS5-trained")):
    rows = DYN[target]["rows"]
    x = [float(row["tagged_HH_separation_A"]) for row in rows]
    force = [float(row["learned_radial_molecular_force_eV_A"]) for row in rows]
    reference = [float(row["reference_radial_molecular_force_eV_A"]) for row in rows]
    energy = [float(row["learned_delta_energy_eV"]) for row in rows]
    ref_energy = [float(row["reference_delta_energy_eV"]) for row in rows]
    axs[0].plot(x, force, color=color, lw=2, label=label)
    axs[0].plot(x, reference, color=color, lw=1.2, ls="--", alpha=.8)
    axs[1].plot(x, energy, color=color, lw=2, label=label)
    axs[1].plot(x, ref_energy, color=color, lw=1.2, ls="--", alpha=.8)
for ax in axs[:2]:
    ax.axvspan(.34, DYN["training_HH_min_A"], color="#d8d8d8", alpha=.45,
               label="outside training range" if ax is axs[0] else None)
    ax.axvline(DYN["training_HH_min_A"], color="#555555", lw=.9, ls=":")
    ax.set_xlim(.35, 1.5)
    ax.set_xlabel(r"Tagged H--H separation (Å)")
axs[0].axhline(0, color="black", lw=.6)
axs[0].set_yscale("symlog", linthresh=2)
axs[0].set_ylabel(r"Radial molecular force (eV / Å)")
axs[0].set_title("(a) Contact force", loc="left", fontweight="bold")
axs[0].legend(frameon=False, fontsize=7, loc="upper left")
axs[1].axhline(0, color="black", lw=.6)
axs[1].set_yscale("symlog", linthresh=1)
axs[1].set_ylabel(r"Energy change (eV)")
axs[1].set_title("(b) Contact energy", loc="left", fontweight="bold")
axs[1].text(.39, 50, "solid: GNN\ndashed: parent", fontsize=7, va="top")
metrics = [row for row in DYN["properties"]["comparisons"]
           if row["reference_target"] == "ips5"]
names = {"coord_OO_lt_3p5A": "Coordination",
         "tetrahedral_q": "Tetrahedrality",
         "oxygen_D_50_100_ps_1e9_m2_s": "Diffusion"}
vals = [100*row["relative_difference"] for row in metrics]
bars_out = axs[2].bar(range(3), vals, color=COLORS[1], width=.62)
axs[2].bar_label(bars_out, labels=[f"{v:+.3f}%" for v in vals], fontsize=8, padding=3)
axs[2].axhline(0, color="black", lw=.7)
axs[2].set_xticks(range(3), [names[row["observable"]] for row in metrics], rotation=22)
axs[2].set_ylabel("Learned IPS5 vs parent (%)")
axs[2].set_title("(c) Property inheritance", loc="left", fontweight="bold")
axs[2].set_ylim(min(-.3, min(vals)-.2), max(1.6, max(vals)+.25))
fig.savefig(OUT / "tip3p_dynamics_extrapolation.pdf")
fig.savefig(OUT / "tip3p_dynamics_extrapolation.png", dpi=180)
plt.close(fig)

# Original vector TOC art, using the new direct-target story.
fig, ax = plt.subplots(figsize=(3.25,1.75))
ax.set(xlim=(0,3.25),ylim=(0,1.75)); ax.axis("off")
ax.text(1.625,1.59,"Statistical physics → learnable target",ha="center",
        fontsize=8,fontweight="bold")
box(ax,.08,.58,.82,.68,"Pair\nphysics",fontsize=7)
box(ax,1.16,.58,.82,.68,"IPS\napproximation","#faeddf",7)
box(ax,2.24,.58,.9,.68,"Local GNN\nlearning",fontsize=7)
arrow(ax,(.91,.92),(1.14,.92)); arrow(ax,(1.99,.92),(2.22,.92))
ax.text(1.625,.28,"Lower error • stable dynamics • parent properties retained",
        ha="center",fontsize=6.6)
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
fig.savefig(ROOT/"toc_graphic.pdf")
fig.savefig(ROOT/"toc_graphic.tif",dpi=300,
            pil_kwargs={"compression":"tiff_lzw"})
plt.close(fig)
