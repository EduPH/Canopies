# -*- coding: utf-8 -*-
"""Tests for canopy.py, mostly against Giunti-Munch Examples 2.19, 3.7, 3.12, 3.16."""
from fractions import Fraction as Fr

from canopies.core import (Complex, PLFiltration, RUDecomposition, STAR, _bubble_to,
                    compute_canopy, detect_structural, diagram_of_pairing,
                    find_first_structural, pairing_for_order,
                    total_consistent_orders)


# --------------------------------------------------------------------------
# The paper's running example L (Fig. 1), realized concretely:
# square 1-2-3-4 with diagonal 13, triangles 123 and 134.
#   a := edge 14   (boundary of c reduces to z_a + z_b)
#   b := edge 13   (boundary of d reduces to z_b)
#   c := triangle 134,  d := triangle 123
# Tree edges 12, 23, 34 and all vertices enter at distinct early values.
# --------------------------------------------------------------------------

LABELS = {"a": (1, 4), "b": (1, 3), "c": (1, 3, 4), "d": (1, 2, 3)}
SIMPLICES = [(1,), (2,), (3,), (4,),
             (1, 2), (2, 3), (3, 4), (1, 4), (1, 3),
             (1, 2, 3), (1, 3, 4)]
BASE = {(1,): -8, (2,): -7, (3,): -6, (4,): -5,
        (1, 2): -3, (2, 3): -2, (3, 4): -1}


def make_L():
    return Complex(SIMPLICES, LABELS)


def fvals(cx, a, b, c, d):
    vals = [None] * cx.n
    for s, v in BASE.items():
        vals[cx.id(s)] = Fr(v)
    for lbl, v in zip("abcd", (a, b, c, d)):
        vals[cx.id(lbl)] = Fr(v)
    return vals


def order_with(cx, suffix_labels):
    """Base simplices in value order, then a,b,c,d in the requested order."""
    base_ids = [cx.id(s) for s, _ in sorted(BASE.items(), key=lambda kv: kv[1])]
    return base_ids + [cx.id(l) for l in suffix_labels]


def named(cx, P):
    return {(cx.name(s), cx.name(t)) for s, t in P
            if cx.name(s) in "abcd" or cx.name(t) in "abcd"}


# --------------------------------------------------------------------------
# chosen pairings (Lemma 2.11 / Example 2.19)
# --------------------------------------------------------------------------

def test_pairing_filled_triangle():
    cx = Complex([(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)])
    f = [0, 0, 0, 0, 0, 1, 2]  # edge 23 enters at 1, triangle at 2
    order = next(total_consistent_orders(cx, f))
    P = pairing_for_order(cx, order)
    d1 = diagram_of_pairing(cx, f, P, dim=1)
    assert d1 == [(1, 2)]
    d0 = diagram_of_pairing(cx, f, P, dim=0, mode="D")
    assert d0 == [(0, float("inf"))]  # only the essential 0-class survives off-diagonal


def test_example_2_19_function_g():
    """g(a)=1, g(b)=2, g(c)=g(d)=3: abcd -> {(a,d),(b,c)}, abdc -> {(a,c),(b,d)}."""
    cx = make_L()
    P1 = pairing_for_order(cx, order_with(cx, "abcd"))
    P2 = pairing_for_order(cx, order_with(cx, "abdc"))
    assert named(cx, P1) == {("a", "d"), ("b", "c")}
    assert named(cx, P2) == {("a", "c"), ("b", "d")}


def test_example_2_19_function_f():
    """f(a)=f(b)=1, f(c)=f(d)=2: order badc -> {(a,c),(b,d)}; abcd -> {(a,d),(b,c)}."""
    cx = make_L()
    P = pairing_for_order(cx, order_with(cx, "badc"))
    assert named(cx, P) == {("a", "c"), ("b", "d")}
    PS = pairing_for_order(cx, order_with(cx, "abcd"))
    assert named(cx, PS) == {("a", "d"), ("b", "c")}


def test_diagram_order_independent():
    """Lemma 2.14: the A-diagram is the same for every total consistent order."""
    cx = make_L()
    f = fvals(cx, 1, 1, 2, 2)
    diags = set()
    n = 0
    for order in total_consistent_orders(cx, f):
        P = pairing_for_order(cx, order)
        diags.add(tuple(diagram_of_pairing(cx, f, P)))
        n += 1
    assert n == 4  # blocks {a,b} and {c,d}, no face relation inside either
    assert len(diags) == 1


# --------------------------------------------------------------------------
# structural / non-Hausdorff detection (Def. 3.15, Examples 3.12 & 3.16)
# --------------------------------------------------------------------------

def test_detect_structural_double_tie():
    """The t=1/3 situation of Example 3.12: f(a)=f(b), f(c)=f(d) -> structural."""
    cx = make_L()
    f = fvals(cx, Fr(2, 3), Fr(2, 3), Fr(7, 3), Fr(7, 3))
    det = detect_structural(cx, f)
    assert det.structural is True
    # lazy: stops at the first disagreement (here after the 2-transposition
    # pre-pass plus a few enumerated orders; 4 total orders exist)
    assert det.orders_checked <= 6
    assert det.structural_points(f) == {(Fr(2, 3), Fr(7, 3))}
    assert det.structural_points(f, mode="D") == {(Fr(2, 3), Fr(7, 3))}


def test_detect_structural_g():
    """g from Fig. 1 (a=1,b=2,c=d=3) is structural (two orders disagree)."""
    cx = make_L()
    det = detect_structural(cx, fvals(cx, 1, 2, 3, 3))
    assert det.structural is True


def test_detect_not_structural_distinct_values():
    cx = make_L()
    det = detect_structural(cx, fvals(cx, 2, 0, 3, 1))
    assert det.structural is False
    assert det.orders_checked == 1  # unique total consistent order


def test_detect_not_structural_t_three_quarters():
    """t=3/4 of Example 3.7: f(a)=f(d)=3/2 tie, but NOT structural (Hausdorff)."""
    cx = make_L()
    f = fvals(cx, Fr(3, 2), Fr(1, 4), Fr(11, 4), Fr(3, 2))
    det = detect_structural(cx, f)
    assert det.structural is False
    assert det.orders_checked <= 3  # the a/d swap is the only freedom


def test_coincidental_example_3_16():
    """Two disjoint edges with identical values: diagram point of multiplicity 2
    but coincidental, not structural."""
    cx = Complex([(1,), (2,), (3,), (4,), (1, 2), (3, 4)])
    f = [0, 1, 0, 1, 2, 2]
    order = next(total_consistent_orders(cx, f))
    P = pairing_for_order(cx, order)
    inf = float("inf")
    assert diagram_of_pairing(cx, f, P, dim=0, mode="D") == \
        [(0, inf), (0, inf), (1, 2), (1, 2)]  # (1,2) has multiplicity 2
    det = detect_structural(cx, f)
    assert det.structural is False


def test_structural_with_essential_classes():
    """A circle: two tied edges swap which one creates the essential 1-class."""
    cx = Complex([(1,), (2,), (3,), (1, 2), (2, 3), (1, 3)])
    f = [-3, -2, -1, 1, 1, 0]  # edges 12 and 23 tied at 1
    det = detect_structural(cx, f)
    assert det.structural is True
    diff = {frozenset((cx.name(s), cx.name(t)) for s, t in
            (det.base_pairing ^ det.witness_pairing))}
    stars = [pr for pr in det.base_pairing ^ det.witness_pairing if pr[1] == STAR]
    assert stars  # the essential pair (sigma, STAR) is among the switched pairs


# --------------------------------------------------------------------------
# PL path of Example 3.7 / Fig. 5:  a=2t, b=1-t, c=2+t, d=3-2t
# --------------------------------------------------------------------------

def make_path(cx):
    spec = dict(BASE)
    spec["a"] = [(0, 0), (1, 2)]
    spec["b"] = [(0, 1), (1, 0)]
    spec["c"] = [(0, 2), (1, 3)]
    spec["d"] = [(0, 3), (1, 1)]
    return PLFiltration(cx, spec)


def test_crossings_exact():
    cx = make_L()
    plf = make_path(cx)
    events, intervals = plf.crossings()
    assert events == [Fr(1, 3), Fr(3, 4)]
    assert intervals == []


def test_canopy_end_to_end_follow_death():
    cx = make_L()
    plf = make_path(cx)
    can = compute_canopy(cx, plf, policy="follow_death")
    ev = {e.t: e for e in can.events}

    e13 = ev[Fr(1, 3)]
    assert e13.is_switch and e13.structural
    assert not e13.forced and len(e13.matchings) == 2   # genuine non-Hausdorff choice
    assert named(cx, e13.removed) == {("a", "d"), ("b", "c")}
    assert named(cx, e13.added) == {("b", "d"), ("a", "c")}

    e34 = ev[Fr(3, 4)]
    assert not e34.is_switch and not e34.structural      # benign crossing

    # paper's trajectory (Example 3.7 discretization): (t,a,d) -> (t,b,d)
    tr_ad = next(tr for tr in can.trajectories
                 if named(cx, {tr.starts[0][1]}) == {("a", "d")})
    assert named(cx, {tr_ad.pair_at(Fr(1, 2))}) == {("b", "d")}

    # structural points as diagram points (Fig. 6 / Ex. 3.12): (2/3, 7/3)
    assert can.structural_points(mode="A") == [(Fr(1, 3), {(Fr(2, 3), Fr(7, 3))})]

    # diagram at t=1/2 (dim 1): {(f(b),f(d)), (f(a),f(c))} = {(1/2,2), (1,5/2)}
    assert can.diagram(Fr(1, 2), dim=1) == [(Fr(1, 2), Fr(2)), (Fr(1), Fr(5, 2))]


def test_canopy_follow_birth_takes_other_branch():
    cx = make_L()
    plf = make_path(cx)
    can = compute_canopy(cx, plf, policy="follow_birth")
    tr_ad = next(tr for tr in can.trajectories
                 if named(cx, {tr.starts[0][1]}) == {("a", "d")})
    assert named(cx, {tr_ad.pair_at(Fr(1, 2))}) == {("a", "c")}


# --------------------------------------------------------------------------
# Morozov (CEM) transposition updates: RUDecomposition
# --------------------------------------------------------------------------

def _random_complex():
    return Complex([(1,), (2,), (3,), (4,), (5,),
                    (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (1, 4),
                    (1, 2, 3), (3, 4, 5)])


def test_ru_transpositions_match_reduction():
    """After every random valid adjacent transposition, the pairing read off
    the updated decomposition must equal the from-scratch reduction (and the
    boolean switch report must match whether the pairing changed).
    Exercises CEM Cases 1-4 including the essential-class (STAR) variants."""
    import random
    random.seed(11)
    cx = _random_complex()
    # start from a consistent order: sort by (dim, id)
    order = sorted(range(cx.n), key=lambda s: (cx.dim[s], s))
    ru = RUDecomposition(cx, order)
    P = pairing_for_order(cx, order)
    assert ru.pairing() == P
    done = 0
    while done < 400:
        i = random.randrange(cx.n - 1)
        a, b = ru.order[i], ru.order[i + 1]
        if cx.is_face(a, b) or cx.is_face(b, a):
            continue
        switched = ru.transpose(i)
        Pnew = pairing_for_order(cx, ru.order)
        assert ru.pairing() == Pnew, f"pairing mismatch after transposing pos {i}"
        assert switched == (Pnew != P), f"switch flag wrong at pos {i}"
        P = Pnew
        done += 1


def test_ru_bubble_to_matches_reduction():
    import random
    random.seed(23)
    cx = _random_complex()
    order = sorted(range(cx.n), key=lambda s: (cx.dim[s], s))
    ru = RUDecomposition(cx, order)
    dims = cx.dim
    for _ in range(50):
        # random consistent target: shuffle within dimension classes
        # (dimension-sorted orders are always face-consistent)
        target = sorted(range(cx.n), key=lambda s: (dims[s], random.random()))
        _bubble_to(ru, target)
        assert ru.order == target
        assert ru.pairing() == pairing_for_order(cx, target)


def test_detection_with_shared_ru_matches_fresh():
    cx = make_L()
    plf = make_path(cx)
    for t in (Fr(1, 3), Fr(3, 4)):
        fv = plf.values(t)
        ru = RUDecomposition(cx, next(total_consistent_orders(cx, fv)))
        det_shared = detect_structural(cx, fv, ru=ru)
        det_fresh = detect_structural(cx, fv)
        assert det_shared.structural == det_fresh.structural


def test_dataset_loader_roundtrip():
    """example_dataset.txt encodes the three-cycles experiment exactly."""
    import os
    from canopies.io import load_dataset
    from canopies import example_dataset_path
    path = str(example_dataset_path())
    cx2, plf2 = load_dataset(path)
    assert cx2.n == 18
    can = compute_canopy(cx2, plf2)
    assert [e.t for e in can.structural_events()] == [Fr(1, 3)]
    assert can.diagram(Fr(1, 2), dim=1, mode="D").count((Fr(1, 2), Fr(2))) == 2


def test_lower_star_loader():
    from canopies.io import from_vertex_functions
    tri = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    vv = {1: [0, 0], 2: [1, 2], 3: [2, 1]}       # vertices 2,3 swap heights
    cx2, plf2 = from_vertex_functions(tri, [0, 1], vv)
    can = compute_canopy(cx2, plf2)
    assert len(can.trajectories) == 4            # 1 essential + 2 pairs + edge/tri
    # dim-0 diagram at t=0: vertex 2 dies at f(edge 12)=1... check it runs
    assert can.diagram(Fr(1, 2), dim=0, mode="A")


def test_find_first_structural_is_lazy():
    cx = make_L()
    plf = make_path(cx)
    res = find_first_structural(cx, plf)
    assert res is not None
    t, det = res
    assert t == Fr(1, 3) and det.structural is True


if __name__ == "__main__":
    import sys
    ok = True
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                ok = False
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                ok = False
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)
