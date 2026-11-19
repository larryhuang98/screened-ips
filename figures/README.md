# Figures

All figures of the paper can be redrawn from the data in this folder:

```bash
cd figures
python make_figures.py data data_classical output   # Figs. 2-6 and Supplementary Figs. S2-S6
python fig1_concept.py output                       # Fig. 1 (schematic)
python figS_relay.py output                         # Supplementary Fig. S1
```

`pstyle.py` holds the shared style (Liberation Sans, sizes at print width, colour roles).

| Folder | Content |
|---|---|
| `data/` | Evaluated errors, scans, projections, DFT far-field response, spectra and MD analyses produced by the scripts in `../scripts` (JSON). |
| `data/kspec/` | Force-error k-spectra (`errk_*.json`) and charge-current spectra (`ccur_*.json`). |
| `data/md/` | Charge structure factors `S_ZZ(k)` (`szz_*.json`), committee spreads and local structure. |
| `data/rcut/` | Cutoff control (5, 6 and 7 A plain MACE; 6 A screened IPS) and same-node timings. |
| `data/farfield/` | Finite-displacement DFT response used for the screening strength (Fig. 3). |
| `data_classical/` | Results of the classical studies (ion pair, TIP3P water) used in Fig. 2 and Supplementary Figs. S1-S2. |
