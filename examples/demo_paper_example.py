# -*- coding: utf-8 -*-
"""
Demo: the running example of Giunti & Munch (Examples 2.17, 3.7, 3.12).

Complex L = square 1-2-3-4 with diagonal 13 and triangles 123, 134.
Labels chosen so the chosen pairings reproduce the paper exactly:
    a = edge (1,4),  b = edge (1,3),  c = triangle (1,3,4),  d = triangle (1,2,3)
(so that with a < b both triangles' reduced boundaries have pivot b, giving
the switch between {(a,d),(b,c)} and {(a,c),(b,d)}).

Fibered filtration (Fig. 5):  f_a = 2t, f_b = 1-t, f_c = 2+t, f_d = 3-2t.
Expected: crossings at t = 1/3 (structural, non-Hausdorff) and t = 3/4 (benign).
"""
from fractions import Fraction as Fr

from canopies.core import (Complex, PLFiltration, compute_canopy, detect_structural,
                    find_first_structural, format_pairing)

LABELS = {"a": (1, 4), "b": (1, 3), "c": (1, 3, 4), "d": (1, 2, 3)}
SIMPLICES = [(1,), (2,), (3,), (4,),
             (1, 2), (2, 3), (3, 4), (1, 4), (1, 3),
             (1, 2, 3), (1, 3, 4)]
# vertices and tree edges enter at distinct early times (the paper sets them
# to 0 and ignores them; distinct values avoid spurious diagonal ambiguity)
BASE = {(1,): -8, (2,): -7, (3,): -6, (4,): -5,
        (1, 2): -3, (2, 3): -2, (3, 4): -1}

cx = Complex(SIMPLICES, LABELS)
spec = dict(BASE)
spec["a"] = [(0, 0), (1, 2)]   # 2t
spec["b"] = [(0, 1), (1, 0)]   # 1-t
spec["c"] = [(0, 2), (1, 3)]   # 2+t
spec["d"] = [(0, 3), (1, 1)]   # 3-2t
plf = PLFiltration(cx, spec)

print("=== crossing events =========================================")
events, intervals = plf.crossings()
print("event times:", events, "  tie intervals:", intervals)

print("\n=== lazy detection at t = 1/3 (Example 3.12) ===============")
det = detect_structural(cx, plf.values(Fr(1, 3)))
print(det.describe(cx))
print("structural diagram points (A):", det.structural_points(plf.values(Fr(1, 3))))

print("\n=== lazy detection at t = 3/4 (Hausdorff, Example 3.12) ====")
det34 = detect_structural(cx, plf.values(Fr(3, 4)))
print(det34.describe(cx))

print("\n=== fully lazy first-witness scan over the whole path ======")
t, d = find_first_structural(cx, plf)
print(f"first non-Hausdorff point at t = {t}")

print("\n=== full canopy over [0,1] =================================")
can = compute_canopy(cx, plf, policy="follow_death")
print(can.summary())

print("\ntrajectories (policy follow_death, as in the paper's Example 3.7):")
for tr in can.trajectories:
    print("  " + tr.describe(cx))

print("\nA-diagram at t = 1/2, dim 1:", can.diagram(Fr(1, 2), dim=1))

out = can.plot(dim=1, fname="vineyard_dim1.png")
print("\nplot written to", out)
