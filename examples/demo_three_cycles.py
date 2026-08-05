# -*- coding: utf-8 -*-
"""
Demo: three 1-cycles opening and closing alternately over t in [0,1].

Complex (two connected components):
  * shared block  = the paper's L: square 1-2-3-4 with diagonal 13.
      cycle z_a (whole square)   born at edge a = (1,4)
      cycle z_b (triangle 123)   born at edge b = (1,3)
      -- they SHARE the tree edges 12, 23 (and interact through the
      triangles c = (1,3,4), d = (1,2,3), whose reduced boundaries are
      z_a + z_b and z_b).
  * disjoint block = triangle 5-6-7, not touching the others:
      cycle z_e born at edge e = (5,7), killed by triangle f = (5,6,7).

Time design (piecewise linear):
      f_a = 2t          f_b = 1 - t        f_c = 2 + t       f_d = 3 - 2t
      f_e = 1/2         f_f : 1/2 on [0,1/4], up to 7/2 at 3/4, back to 1/2 at 1
  * the square cycle is widest at t=0 and shrinks;
  * the shared triangle cycle grows, and the two shared cycles SWITCH their
    pairing at t = 1/3 (both a,b and c,d tie simultaneously) -> structural,
    non-Hausdorff, two valid trajectory continuations;
  * the disjoint cycle is CLOSED (on the diagonal, persistence 0) until
    t = 1/4, widest at t = 3/4, closed again at t = 1;
  * at t = 1/2 the disjoint bar coincides EXACTLY with the (b,d) bar:
    the diagram point (1/2, 2) has multiplicity 2 -- but it is COINCIDENTAL
    (Example 3.16): no re-ordering exchanges the two pairs, not structural.
"""
from fractions import Fraction as Fr

from canopies.core import (Complex, PLFiltration, compute_canopy, detect_structural,
                    format_pairing)

# ---- complex ---------------------------------------------------------------
SIMPLICES = [(1,), (2,), (3,), (4,), (5,), (6,), (7,),
             (1, 2), (2, 3), (3, 4), (1, 4), (1, 3),
             (5, 6), (6, 7), (5, 7),
             (1, 2, 3), (1, 3, 4), (5, 6, 7)]
LABELS = {"a": (1, 4), "b": (1, 3), "c": (1, 3, 4), "d": (1, 2, 3),
          "e": (5, 7), "f": (5, 6, 7)}
cx = Complex(SIMPLICES, LABELS)

# ---- fibered filtration ----------------------------------------------------
spec = {(1,): -14, (2,): -13, (3,): -12, (4,): -11, (5,): -10, (6,): -9, (7,): -8,
        (1, 2): -3, (2, 3): -2.9, (3, 4): -2.8, (5, 6): -2.7, (6, 7): -2.6}
spec["a"] = [(0, 0), (1, 2)]                                   # 2t
spec["b"] = [(0, 1), (1, 0)]                                   # 1-t
spec["c"] = [(0, 2), (1, 3)]                                   # 2+t
spec["d"] = [(0, 3), (1, 1)]                                   # 3-2t
spec["e"] = 0.5
spec["f"] = [(0, 0.5), (Fr(1, 4), 0.5), (Fr(3, 4), 3.5), (1, 0.5)]
plf = PLFiltration(cx, spec)

# ---- the three bars over time ----------------------------------------------
print("=== the three 1-cycles over time (D-diagrams, dim 1) ===============")
can = compute_canopy(cx, plf, policy="follow_death")
for t in (Fr(0), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(1)):
    bars = can.diagram(t, dim=1, mode="A")
    txt = ",  ".join(f"[{b}, {d}]" + ("  (closed)" if b == d else "")
                     for b, d in bars)
    print(f"  t = {str(t):4}   {txt}")

# ---- events, classified -----------------------------------------------------
print("\n=== events =========================================================")
print(can.summary())

# ---- the two interesting parameter values -----------------------------------
print("\n=== t = 1/3: the two SHARED cycles -> structural (Ex. 3.12) ========")
det = detect_structural(cx, plf.values(Fr(1, 3)))
print(det.describe(cx))

print("\n=== t = 1/2: shared vs DISJOINT cycle -> coincidental (Ex. 3.16) ===")
fv = plf.values(Fr(1, 2))
det2 = detect_structural(cx, fv)
print(det2.describe(cx))
bars = can.diagram(Fr(1, 2), dim=1, mode="D")
print(f"D-diagram at t=1/2, dim 1: {bars}")
mult = bars.count((Fr(1, 2), Fr(2)))
print(f"multiplicity of the point (1/2, 2): {mult}  "
      f"(two bars coincide, but NOT structural: the pairs (b,d) and (e,f)\n"
      f" belong to different components -- no re-ordering can exchange them)")

# ---- trajectories ------------------------------------------------------------
print("\n=== dim-1 trajectories (policy follow_death) =======================")
for tr in can.trajectories:
    if cx.dim[tr.starts[0][1][0]] == 1:
        print("  " + tr.describe(cx))

# ---- self-checks -------------------------------------------------------------
struct = [e.t for e in can.structural_events()]
assert struct == [Fr(1, 3)], struct
ev = {e.t: e for e in can.events}
assert not ev[Fr(1, 2)].structural and not ev[Fr(1, 2)].is_switch
assert det.structural is True and det2.structural is False
assert mult == 2
assert len(ev[Fr(1, 3)].matchings) == 2            # genuine non-Hausdorff choice
b0 = can.diagram(Fr(0), dim=1, mode="A")
assert (Fr(1, 2), Fr(1, 2)) in b0                  # disjoint cycle starts closed
b1 = can.diagram(Fr(1), dim=1, mode="A")
assert (Fr(1, 2), Fr(1, 2)) in b1                  # ...and ends closed
print("\nall self-checks passed")

out = can.plot(dim=1, fname="three_cycles_dim1.png")
print("plot written to", out)

# ---- richer visualizations ---------------------------------------------------
from canopies.viz import (animate_sublevel, animate_vineyard_3d, plot_persistence,
                          plot_snapshots, plot_sublevel_snapshots,
                          plot_vineyard_3d)

# vertex embedding: square with diagonal on the left, disjoint triangle right
POS = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0), 4: (0.0, 1.0),
       5: (1.9, 0.1), 6: (2.9, 0.1), 7: (2.4, 0.95)}

print(plot_vineyard_3d(can, dim=1, fname="three_cycles_vineyard3d.png"))
print(animate_vineyard_3d(can, "three_cycles_vineyard3d.gif", dim=1))
print(plot_persistence(can, dim=1, fname="three_cycles_persistence.png"))
print(plot_snapshots(can, POS, times=[0, Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(3, 4), 1],
                     dim=1, fname="three_cycles_snapshots.png"))

# sublevel-set view at a fixed level s: simplices appear/disappear, holes
# outlined by representative cycles in trajectory colors
LEVEL = Fr(7, 4)
print(plot_sublevel_snapshots(can, POS, times=[0, Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(3, 4), 1],
                              level=LEVEL, dim=1,
                              fname="three_cycles_sublevel.png"))
try:
    print(animate_sublevel(can, POS, "three_cycles_sublevel.gif",
                           level=LEVEL, dim=1, frames=80, fps=10))
except Exception as exc:                                    # pillow missing etc.
    print(f"gif skipped: {exc}")
