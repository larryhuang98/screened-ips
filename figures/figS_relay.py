"""SI Fig. S1: relayed-interaction learning, 3 versus 15 atoms (three seeds).
Values from paper/figures/mace_nacl_local_complexity_task3/specialist_seed_metrics.json (ips-mlip)."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
from pstyle import FULL, INK, MUTE, C, panel_labels

out = sys.argv[1] if len(sys.argv) > 1 else "../figures"
# rows: seed; columns: (3 atoms, 15 atoms)
E = np.array([[0.0026004666, 1.4037958665], [0.0012444571, 1.3323022070], [0.0011173230, 1.4829551495]])
Ec = np.array([[0.0005782155, 0.0690040012], [0.0003210152, 0.0647267979], [0.0003049914, 0.0602407488]])
F = np.array([[0.0120957813, 0.0568156511], [0.0079989397, 0.0564423015], [0.0077957665, 0.0556551755]]) * 1e3

fig, axes = plt.subplots(1, 3, figsize=(FULL * 0.78, 1.75), layout="constrained")
panels = [(E, "Tagged-energy RMSE (eV)"), (Ec, "Centred tagged-energy RMSE (eV)"), (F, "Endpoint force RMSE (meV Å$^{-1}$)")]
x = np.array([0, 1])
for ax, (Y, lab) in zip(axes, panels):
    for s, y in enumerate(Y):
        ax.plot(x, y, color=C["untransformed"], lw=0.7, marker="o", ms=2.6, mfc="white", mew=0.7, zorder=2,
                label="single seeds" if s == 0 else None)
    m = Y.mean(0)
    ax.plot(x, m, color=INK, lw=0, marker="_", ms=11, mew=1.4, zorder=3, label="mean")
    for xi, v in zip(x, m):
        ax.annotate(f"{v:.2g}", (xi, v), xytext=(7, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=5.6, color=INK)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["3 atoms", "15 atoms"]); ax.set_xlim(-0.4, 1.6)
    ax.set_ylabel(lab)
axes[2].yaxis.set_major_locator(FixedLocator([5, 10, 20, 50])); axes[2].yaxis.set_minor_formatter(NullFormatter())
axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); axes[2].set_ylim(5, 80)
axes[0].legend(loc="lower right", fontsize=5.6)
fig.get_layout_engine().set(w_pad=0.04, rect=(0, 0, 1, 0.93))
fig.canvas.draw(); r = fig.canvas.get_renderer()
for ax, t in zip(axes, "abc"):
    bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    fig.text(bb.x0, 0.995, t, fontsize=8, weight="bold", color=INK, va="top", ha="left")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}/figS_relay.{ext}")
print("ok")
