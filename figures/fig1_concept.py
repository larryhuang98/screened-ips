"""Fig. 1: a, classical IPS target; b, QM screening and the scaled correction; c, workflow.
Pair curves: electrostatic part of one +/- ion pair in units of |q_i q_j|/R_c (schematic, no fitted data)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pstyle import FULL, INK, MUTE, FAINT, C, PANEL

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
REF, IPS, FORMAL = C["reference"], C["restored"], C["formal"]
LAMF = 0.56
P = lambda x: (-35 + 35 * x**2 - 21 * x**4 + 5 * x**6) / 16
x = np.linspace(0.2, 2.0, 900); inn = x < 1
def target(lref, lam):
    return np.where(inn, lref / x + lam * P(x), (lref - lam) / x)

fig = plt.figure(figsize=(FULL, 3.3))
top, hgt = 0.915, 0.56
axa = fig.add_axes([0.082, top - hgt, 0.40, hgt]); axb = fig.add_axes([0.572, top - hgt, 0.40, hgt])
axc = fig.add_axes([0.0, 0.0, 1.0, 0.175]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)

def frame(ax, title, ylabel=True):
    ax.axvspan(1, 2.0, color=FAINT, lw=0, zorder=0)
    ax.axvline(1, color=MUTE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=1)
    ax.set_xlim(0.2, 2.0); ax.set_ylim(-0.95, 2.5)
    ax.set_xticks([0.5, 1, 1.5, 2]); ax.set_xticklabels(["0.5", "1", "1.5", "2"]); ax.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Pair distance, $r/R_\\mathrm{c}$")
    if ylabel: ax.set_ylabel("Electrostatic pair energy ($|q_iq_j|/R_\\mathrm{c}$)")
    ax.text(1.5, 2.40, "beyond $R_\\mathrm{c}$: not seen\nby a local model", ha="center", va="top", fontsize=5.8, color=MUTE, linespacing=1.15)
    ax.set_title(title, fontsize=6.5, weight="normal", color=INK, loc="left", pad=5)
def lab(ax, xy, text, color, xytext, ha="left", va="center"):
    ax.annotate(text, xy, xytext, ha=ha, va=va, fontsize=6.2, color=color, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.45, color=color, shrinkA=1.5, shrinkB=1.5))

frame(axa, "Classical reference: known charges")
axa.plot(x, 1 / x, color=REF, lw=1.1, ls=(0, (3.5, 2)))
axa.plot(x, target(1.0, 1.0), color=IPS, lw=1.5, solid_capstyle="round")
axa.text(1.97, 0.80, "Coulomb, $1/r$", color=REF, fontsize=6.2, ha="right", va="bottom")
axa.text(1.04, 0.06, "IPS target ($\\lambda=1$): no tail", color=IPS, fontsize=6.2, ha="left", va="bottom")
lab(axa, (0.93, target(1.0, 1.0)[np.searchsorted(x, 0.93)]), "smooth zero\nat $R_\\mathrm{c}$", IPS, (0.62, 0.62))

frame(axb, "QM reference: electronically screened tail", ylabel=False)
axb.plot(x, 1 / x, color=REF, lw=0.8, ls=(0, (3.5, 2)), alpha=0.45)
axb.plot(x, LAMF / x, color=INK, lw=1.0)
axb.text(1.97, 0.60, "bare Coulomb, $1/r$", color=REF, alpha=0.8, fontsize=5.8, ha="right", va="bottom")
axb.plot(x, target(LAMF, 1.0), color=FORMAL, lw=1.3, solid_capstyle="round")
axb.plot(x, target(LAMF, LAMF), color=IPS, lw=1.5, solid_capstyle="round")
lab(axb, (1.30, LAMF / 1.30), "screened Coulomb, $\\lambda_\\mathrm{far}/r$\n($\\lambda_\\mathrm{far}=0.56$, from DFT)", INK, (1.10, 1.05))
axb.text(1.04, 0.06, "screened correction ($\\lambda=\\lambda_\\mathrm{far}$): no tail", fontsize=6.0, color=IPS, ha="left", va="bottom")
lab(axb, (1.30, (LAMF - 1) / 1.30), "formal correction ($\\lambda=1$):\ntail $-0.44/r$, wrong sign", FORMAL, (1.22, -0.70))

def box(x0, w, head, body, ec, fc):
    y0, h = 0.08, 0.80
    axc.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.008", fc=fc, ec=ec, lw=0.7, mutation_aspect=9.0))
    axc.text(x0 + w / 2, y0 + h - 0.10, head, ha="center", va="top", fontsize=6.5, weight="bold", color=INK)
    axc.text(x0 + w / 2, y0 + h - 0.34, body, ha="center", va="top", fontsize=6.2, color=INK, linespacing=1.3)
    return y0 + h / 2
xs, w = [0.035, 0.280, 0.525, 0.770], 0.195
ym = box(xs[0], w, "Reference", "$E_\\mathrm{ref}$ from DFT", "#9A9A9A", "#F5F5F5")
box(xs[1], w, "IPS target", "$T_\\lambda=E_\\mathrm{ref}+\\lambda\\Delta$,  $\\Delta=E_\\mathrm{IPS}-E_\\mathrm{Ew}$\n$\\lambda$ from DFT or three trainings", IPS, "#EAF2F9")
box(xs[2], w, "Local GNN (MACE)", "learns $E_\\theta\\approx T_\\lambda$\nwithin $R_\\mathrm{c}$", "#9A9A9A", "#F5F5F5")
box(xs[3], w, "MACE + screened IPS", "$E_\\theta-\\lambda\\Delta$\nused for simulation", IPS, "#EAF2F9")
for i in range(3):
    axc.annotate("", (xs[i + 1] - 0.008, ym), (xs[i] + w + 0.008, ym),
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTE, mutation_scale=7, shrinkA=0, shrinkB=0))

# panel letters on the title baseline (a, b) and at the box top (c)
xa_ = None
for ax, t in [(axa, "a"), (axb, "b")]:
    fig.canvas.draw(); bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    tb = ax.title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x0 - 0.008, tb.y0, t, fontsize=8, weight="bold", color=INK, va="baseline", ha="left"); xa_ = xa_ if xa_ is not None else bb.x0 - 0.008
fig.text(xa_, 0.175, "c", fontsize=8, weight="bold", color=INK, va="baseline", ha="left")
fig.savefig(f"{OUT}/fig1_concept.pdf"); fig.savefig(f"{OUT}/fig1_concept.png", dpi=300)
