"""Shared figure style for the manuscript (npj Comput Mater; 180 mm full width = 7.09 in, 88 mm single column = 3.46 in).
All figures are drawn at their printed size, so font sizes below are final point sizes."""
import matplotlib as mpl

FULL, HALF = 7.09, 3.46
FONT = "Liberation Sans"          # Arial-metric; math in the same face
mpl.rcParams.update({
    "font.family": FONT, "font.size": 6.5, "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT,
    "axes.labelsize": 6.5, "axes.titlesize": 6.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelpad": 2.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "errorbar.capsize": 1.5,
    "patch.linewidth": 0.5, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300, "figure.dpi": 150,
})

INK, MUTE, FAINT = "#222222", "#666666", "#EFEFEF"
# Okabe-Ito based roles, fixed across the paper
C = {
    "untransformed": "#8C8C8C",   # local model on the untransformed target
    "restored": "#0072B2",        # screened IPS target, exterior restored
    "baseline": "#E69F00",        # IPS baseline target
    "hamiltonian": "#009E73",     # IPS Hamiltonian deployment
    "les": "#CC79A7",             # learned charges (LES)
    "les_ips": "#882255",         # LES trained on the IPS target
    "formal": "#D55E00",          # formal-charge correction (lambda = 1)
    "reference": "#4D4D4D",       # reference / Coulomb
    "NaCl": "#0072B2", "CsCl": "#E69F00", "BaCl2": "#009E73", "SrCl2": "#CC79A7",
    "ss11": "#D55E00", "ssmgna": "#56B4E9", "water": "#666666",
}
PANEL = dict(fontsize=8, weight="bold", color=INK, va="top", ha="left")

def panel_labels(fig, axes, letters="abcdefgh", dx=-0.012, dy=0.0):
    """Place bold panel letters at the top-left of each axes' tight bbox (figure coordinates)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax, t in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        fig.text(bb.x0 + dx, bb.y1 + dy, t, **PANEL)
