# -*- coding: utf-8 -*-
"""
demo_paper_viz.py
=================

The Giunti & Munch running example (Examples 2.17 / 3.7 / 3.12) with the SAME
interactive treatment as the fMRI canopy: an HTML with two synced 3D panels
driven by one time slider --
  * left : the embedded complex L (square 1-2-3-4 + diagonal, triangles c, d);
           simplices APPEAR and DISAPPEAR in the sublevel set K_s(t) as t moves;
  * right: the canopy dim-1 trajectories in (birth, death, time), with the
           structural (non-Hausdorff) event at t = 1/3 marked in red.

We also keep the plot we were already generating (can.plot dim=1 -> PNG).

Complex L (labels chosen to reproduce the paper's pairings):
    a = (1,4)  b = (1,3)  c = (1,3,4)  d = (1,2,3)
Fibered filtration (Fig. 5):  f_a = 2t, f_b = 1-t, f_c = 2+t, f_d = 3-2t
Expected: structural (non-Hausdorff) crossing at t = 1/3, benign one at t = 3/4.
"""
import os

from canopies.core import Complex, PLFiltration, compute_canopy
from canopies.html_viz import export_filtration_html, export_dual_sweep_html

OUTDIR = os.environ.get("CANOPIES_OUTDIR", os.getcwd())

LABELS = {"a": (1, 4), "b": (1, 3), "c": (1, 3, 4), "d": (1, 2, 3)}
SIMPLICES = [(1,), (2,), (3,), (4,),
             (1, 2), (2, 3), (3, 4), (1, 4), (1, 3),
             (1, 2, 3), (1, 3, 4)]
# tree edges / vertices enter early at distinct values (paper sets them to 0)
BASE = {(1,): -8, (2,): -7, (3,): -6, (4,): -5,
        (1, 2): -3, (2, 3): -2, (3, 4): -1}

cx = Complex(SIMPLICES, LABELS)
spec = dict(BASE)
spec["a"] = [(0, 0), (1, 2)]   # 2t
spec["b"] = [(0, 1), (1, 0)]   # 1-t
spec["c"] = [(0, 2), (1, 3)]   # 2+t
spec["d"] = [(0, 3), (1, 1)]   # 3-2t
plf = PLFiltration(cx, spec)

can = compute_canopy(cx, plf, policy="follow_death")
print(can.summary())

# --- the plot we already generate -----------------------------------------
png = can.plot(dim=1, fname=os.path.join(OUTDIR, "paper_vineyard_dim1.png"))
print("wrote", png)

# --- the new interactive HTML (simplices appearing/disappearing + canopy) ---
# Square embedding; diagonal 1-3 splits it into triangles d=(1,2,3), c=(1,3,4).
POS = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0)}
# Level chosen so triangle c (=2+t) leaves at t=1/2 and d (=3-2t) enters at
# t=1/4: over [0,1] the two faces swap, so a hole opens/closes around the
# structural point t=1/3.
LEVEL = 2.5
html = export_filtration_html(
    cx, plf, can, POS, os.path.join(OUTDIR, "paper_canopy_viz.html"),
    level=LEVEL, frames=61, dims=(1,),
    title="Giunti–Munch running example L — sublevel simplices + canopy")
print("wrote", html)

# --- dual-sweep HTML: independent t and s sliders --------------------------
html2 = export_dual_sweep_html(
    cx, plf, can, POS, os.path.join(OUTDIR, "paper_canopy_dualsweep.html"),
    frames=61, dims=(1,), s_default=1.5,   # start inside the hole window
    title="Giunti–Munch example L — t and s sweeps + canopy")
print("wrote", html2)
