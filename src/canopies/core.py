# -*- coding: utf-8 -*-
"""
canopy.py
=========

Augmented canopies (Giunti & Munch, "Canopies: A Generalization of Vines and
Vineyards for Parameterized Persistence") for 1-parameter fibered filtration
functions  g : K x [0,1] -> R  with piecewise-linear g_sigma.

What is implemented
-------------------
* Chosen pairings.  For a *total* consistent order the chosen pairing P_< is
  unique (Lemma 2.11); it is computed by standard Z/2 boundary-matrix
  reduction, which realizes the split conditions SC1-3.

* Structural / non-Hausdorff point detection (Definition 3.15).  Criterion:

      p in B is structural  <=>  the base pairing P0 of g_p has two pairs
                                 e = (s,t), e' = (s',t') with eta(e) = eta(e')
                                 (same birth dimension) and some total
                                 consistent order of g_p pairs s with t' and
                                 s' with t  (a witness).

  Then swapping e and e' is a filtration equivalence of P0 with itself
  (Def. 2.18 with both orders equal; Example 2.19 for f), the mechanism of
  Example 3.12.  Equal eta is necessary because eta : E -> H is continuous
  (Def. 3.1) and H is Hausdorff, so fiber points with different diagram
  values are always separated (cf. the proof of Prop. 5.2).  Non-Hausdorff
  partners lie in the same fiber, and near p every order consistent with g_q
  is consistent with g_p, so only re-orderings of g_p itself matter.

  "Two consistent orders give different pairings" is necessary but NOT
  sufficient: a single tie with a vineyard switch changes the pairing, yet
  the equivalence is unique (Example 2.19 for g; Example 3.14 calls such ties
  benign).

  The search is *lazy*: total consistent orders are enumerated one at a time
  (products of linear extensions of the "tie blocks") and the search stops at
  the first disagreement.  A cheap adjacent-transposition pre-pass catches
  the typical switch immediately.

* Trajectories (Definition 3.6) along B = [0,1]: vineyard-style tracking of
  pairs across crossing events, in the spirit of Cohen-Steiner /
  Edelsbrunner / Morozov, but with *all* valid continuations enumerated at
  each switch -- these are exactly the non-Hausdorff choice points.

* Morozov updates (RUDecomposition): an updatable RU-decomposition D = R.U
  implementing the transposition case analysis of Cohen-Steiner /
  Edelsbrunner / Morozov, "Vines and Vineyards by Updating Persistence in
  Linear Time".  One full O(n^3) reduction is done at the start of the path
  (or of a detection); every subsequent order change costs O(n) per adjacent
  transposition.  Used by compute_canopy (across events), by the detection
  pre-pass (test a transposition, read off whether the pairing switched,
  undo), and by the enumeration fall-back (successive orders reached by
  transpositions instead of re-reducing).

All arithmetic is exact (fractions.Fraction), so ties are exact.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "STAR", "Complex", "PLFiltration", "Detection", "SwitchEvent",
    "Trajectory", "Canopy", "RUDecomposition",
    "pairing_for_order", "total_consistent_orders", "value_blocks",
    "check_filtration", "detect_structural", "compute_canopy",
    "find_first_structural", "format_pairing",
]

STAR = "*"                     # the symbol for the pairing partner of essential classes; f(STAR) = +oo
INF = float("inf")

Pair = Tuple[int, Union[int, str]]
Pairing = FrozenSet[Pair]


def _to_frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        # decimal-literal intent: 0.1 -> 1/10
        return Fraction(str(x))
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError(f"cannot convert {x!r} to Fraction")


def _pair_key(pr: Pair):
    s, t = pr
    return (s, 1, -1) if t == STAR else (s, 0, t)


# --------------------------------------------------------------------------
# Simplicial complexes
# --------------------------------------------------------------------------

class Complex:
    """A finite simplicial complex with optional simplex labels.

    `simplices` is an iterable of vertex tuples; the complex must be closed
    under taking faces. `labels` optionally maps a label string to a simplex,
    e.g. {"a": (1, 4)}.
    Internally simplices are integer ids; labels are used for pretty output
    and may be used wherever a simplex is expected in the public API.
    """

    def __init__(self, simplices: Iterable[Sequence], labels: Optional[Dict[str, Sequence]] = None):
        simps = sorted({tuple(sorted(s)) for s in simplices}, key=lambda s: (len(s), s))
        if not simps:
            raise ValueError("empty complex")
        self.simplices: List[tuple] = simps
        self.index: Dict[tuple, int] = {s: i for i, s in enumerate(simps)}
        self.n = len(simps)
        self.dim = [len(s) - 1 for s in simps]
        self.facets: List[List[int]] = []
        for s in simps:
            if len(s) == 1:
                self.facets.append([])
                continue
            fs = []
            for fc in combinations(s, len(s) - 1):
                if fc not in self.index:
                    raise ValueError(f"complex not closed under faces: {fc} (face of {s}) missing")
                fs.append(self.index[fc])
            self.facets.append(fs)
        # strict-face relation sigma < tau  (sigma a proper face of tau)
        self._face = [[False] * self.n for _ in range(self.n)]
        for i, si in enumerate(simps):
            ssi = set(si)
            for j, sj in enumerate(simps):
                if len(si) < len(sj) and ssi.issubset(sj):
                    self._face[i][j] = True
        self.label_of: Dict[int, str] = {}
        self.id_of: Dict[str, int] = {}
        if labels:
            for name, s in labels.items():
                i = self.index[tuple(sorted(s))]
                self.label_of[i] = name
                self.id_of[name] = i

    def is_face(self, i: int, j: int) -> bool:
        """True iff simplex i is a proper face of simplex j."""
        return self._face[i][j]

    def id(self, s) -> int:
        """Resolve a label / vertex tuple / internal id to an internal id."""
        if isinstance(s, int):
            return s
        if isinstance(s, str):
            return self.id_of[s]
        return self.index[tuple(sorted(s))]

    def name(self, i) -> str:
        if i == STAR:
            return STAR
        return self.label_of.get(i, str(self.simplices[i]))

    def pair_name(self, pr: Pair) -> str:
        return f"({self.name(pr[0])},{self.name(pr[1])})"


def format_pairing(cx: Complex, P: Pairing, dims: Optional[Iterable[int]] = None) -> str:
    prs = sorted(P, key=_pair_key)
    if dims is not None:
        dims = set(dims)
        prs = [p for p in prs if cx.dim[p[0]] in dims]
    return "{" + ", ".join(cx.pair_name(p) for p in prs) + "}"


def check_filtration(cx: Complex, fvals: Sequence) -> None:
    """Ensure f(sigma) <= f(tau) whenever sigma is a facet of tau (Def. 2.6)."""
    for j in range(cx.n):
        for i in cx.facets[j]:
            if fvals[i] > fvals[j]:
                raise ValueError(
                    f"not a filtration function: f({cx.name(i)})={fvals[i]} > f({cx.name(j)})={fvals[j]}")


# --------------------------------------------------------------------------
# Chosen pairing of a total consistent order (Lemma 2.11 / SC1-3)
# --------------------------------------------------------------------------

def pairing_for_order(cx: Complex, order: Sequence[int]) -> Pairing:
    """The unique chosen pairing of a total consistent order (Lemma 2.11).

    Standard Z/2 column reduction of the boundary matrix in the given order;
    the resulting (pivot row, column) pairs are exactly the pairs satisfying
    the split conditions SC1-3. Simplices whose column reduces to zero and
    which never appear as a pivot are paired with STAR.
    """
    pos = [0] * cx.n
    for k, s in enumerate(order):
        pos[s] = k
    for j in range(cx.n):
        for i in cx.facets[j]:
            if pos[i] > pos[j]:
                raise ValueError(
                    f"order not consistent: face {cx.name(i)} after coface {cx.name(j)}")
    R: Dict[int, int] = {}        # pivot position -> reduced column (bitmask over positions)
    killer: Dict[int, int] = {}   # pivot position -> simplex id of the killing column
    for k in range(cx.n):
        j = order[k]
        col = 0
        for i in cx.facets[j]:
            col ^= 1 << pos[i]
        while col:
            piv = col.bit_length() - 1
            if piv not in R:
                break
            col ^= R[piv]
        if col:
            piv = col.bit_length() - 1
            R[piv] = col
            killer[piv] = j
    pairs = set()
    paired_creators = set()
    killers = set(killer.values())
    for piv, j in killer.items():
        pairs.add((order[piv], j))
        paired_creators.add(order[piv])
    for s in order:
        if s not in paired_creators and s not in killers:
            pairs.add((s, STAR))
    return frozenset(pairs)


# --------------------------------------------------------------------------
# RU-decomposition with O(n/word) transpositions
# (Cohen-Steiner, Edelsbrunner & Morozov, "Vines and Vineyards by Updating
#  Persistence in Linear Time", Cases 1-4)
# --------------------------------------------------------------------------

class RUDecomposition:
    """Maintains D = R*U for the current total consistent order.

    Rows and columns are indexed by simplex id; the order lives only in the
    pos/order maps, so a transposition never moves matrix data -- it just
    repairs the decomposition per the CEM case analysis.  R is stored as
    column bitmasks and U as ROW bitmasks, so every repair operation
    (column addition on R / row addition on U) is a single big-int xor.

    Invariants: R reduced (distinct pivots), U upper-triangular w.r.t. the
    current positions, D = R*U.  The pairing read off R's pivots is THE
    chosen pairing of the current order (pairing uniqueness lemma /
    Lemma 2.11), independent of which valid decomposition we hold.
    """

    def __init__(self, cx: Complex, order: Sequence[int]):
        n = cx.n
        self.cx = cx
        self.order = list(order)
        self.pos = [0] * n
        for k, s in enumerate(self.order):
            self.pos[s] = k
        self.R: List[int] = [0] * n          # column id -> bitmask over row ids
        self.U: List[int] = [1 << r for r in range(n)]   # ROW id -> bitmask over column ids
        self.low: List[Optional[int]] = [None] * n       # column id -> pivot row id
        self.lowinv: Dict[int, int] = {}                  # pivot row id -> column id
        self.version = 0                     # bumped whenever the pairing changes
        for s in self.order:                 # initial reduction, O(n^3 / w)
            col = 0
            for f in cx.facets[s]:
                col |= 1 << f
            while col:
                piv = self._max_pos_bit(col)
                other = self.lowinv.get(piv)
                if other is None:
                    break
                col ^= self.R[other]
                self.U[other] ^= self.U[s]   # col_s += col_other  <=>  row_other += row_s
            self.R[s] = col
            if col:
                piv = self._max_pos_bit(col)
                self.low[s] = piv
                self.lowinv[piv] = s

    def _max_pos_bit(self, mask: int) -> int:
        """Row id of the entry with maximal position ('low') in a column."""
        best_id, best_pos = -1, -1
        m = mask
        while m:
            b = (m & -m).bit_length() - 1
            if self.pos[b] > best_pos:
                best_pos, best_id = self.pos[b], b
            m &= m - 1
        return best_id

    def pairing(self) -> Pairing:
        pairs = set()
        for c in range(self.cx.n):
            if self.R[c]:
                pairs.add((self.low[c], c))
        for s in range(self.cx.n):
            if not self.R[s] and s not in self.lowinv:
                pairs.add((s, STAR))
        return frozenset(pairs)

    def copy(self) -> "RUDecomposition":
        new = object.__new__(RUDecomposition)
        new.cx = self.cx
        new.order = list(self.order)
        new.pos = list(self.pos)
        new.R = list(self.R)
        new.U = list(self.U)
        new.low = list(self.low)
        new.lowinv = dict(self.lowinv)
        new.version = self.version
        return new

    def transpose(self, i: int) -> bool:
        """Swap the simplices at positions i, i+1 and repair D = R*U.
        Returns True iff the pairing changed (a switch).  O(n/word)."""
        a, b = self.order[i], self.order[i + 1]
        if self.cx.is_face(a, b):
            raise ValueError("cannot transpose a face past its coface")
        switched = False
        a_positive = self.R[a] == 0
        b_positive = self.R[b] == 0
        Uab = (self.U[a] >> b) & 1

        def do_swap():
            self.order[i], self.order[i + 1] = b, a
            self.pos[a], self.pos[b] = i + 1, i

        if a_positive and b_positive:                       # Case 1
            if Uab:
                self.U[a] &= ~(1 << b)                      # safe: column a of R is zero
            l = self.lowinv.get(b)
            k = self.lowinv.get(a)
            if l is not None and (self.R[l] >> a) & 1:
                if k is not None:
                    if self.pos[k] < self.pos[l]:           # Case 1.1.1: no switch
                        self.R[l] ^= self.R[k]              # col_l += col_k
                        self.U[k] ^= self.U[l]              # row_k += row_l
                        do_swap()                           # lows keep their row ids
                    else:                                   # Case 1.1.2: SWITCH
                        self.R[k] ^= self.R[l]              # col_k += col_l
                        self.U[l] ^= self.U[k]
                        do_swap()
                        self.low[k], self.low[l] = b, a
                        self.lowinv[b], self.lowinv[a] = k, l
                        switched = True
                else:                                       # star switch: (b,l) -> (a,l)
                    do_swap()
                    self.low[l] = a
                    del self.lowinv[b]
                    self.lowinv[a] = l
                    switched = True
            else:                                           # Case 1.2
                do_swap()
        elif not a_positive and not b_positive:             # Case 2
            if Uab:
                self.U[a] ^= self.U[b]                      # row_a += row_b
                self.R[b] ^= self.R[a]                      # col_b += col_a
                la, lb = self.low[a], self.low[b]
                if self.pos[la] < self.pos[lb]:             # Case 2.1.1: no switch
                    do_swap()
                else:                                       # Case 2.1.2: SWITCH
                    do_swap()
                    self.R[a] ^= self.R[b]                  # col_a += col_b (post-swap)
                    self.U[b] ^= self.U[a]
                    self.low[a], self.low[b] = lb, la
                    self.lowinv[lb], self.lowinv[la] = a, b
                    switched = True
            else:                                           # Case 2.2
                do_swap()
        elif not a_positive and b_positive:                 # Case 3: a negative, b positive
            if Uab:                                         # Case 3.1: SWITCH
                la = self.low[a]
                l = self.lowinv.get(b)
                self.U[a] ^= self.U[b]
                self.R[b] ^= self.R[a]                      # col b <- old col a
                do_swap()
                self.R[a] ^= self.R[b]                      # col a <- 0 (a turns positive)
                self.U[b] ^= self.U[a]
                self.low[b], self.low[a] = la, None
                self.lowinv[la] = b
                if l is not None:                           # row pairing (b,l) -> (a,l)
                    assert (self.R[l] >> a) & 1, "invalid state in Case 3.1"
                    self.low[l] = a
                    self.lowinv[a] = l
                if b in self.lowinv:
                    del self.lowinv[b]
                switched = True
            else:                                           # Case 3.2
                do_swap()
        else:                                               # Case 4: a positive, b negative
            if Uab:
                self.U[a] &= ~(1 << b)                      # safe: column a of R is zero
            do_swap()
        if switched:
            self.version += 1
        return switched


def _bubble_to(ru: RUDecomposition, target: Sequence[int]) -> bool:
    """Transform ru's order into `target` by adjacent transpositions.  Only
    inverted pairs are swapped; between two linear extensions of the same
    poset every inverted pair is incomparable (and, for two orders consistent
    with the same f, tied in value), so all intermediate orders are valid.
    Cost: (#inversions) * O(n/word).  Returns True iff any switch occurred."""
    switched = False
    for idx in range(len(target)):
        s = target[idx]
        p = ru.pos[s]
        while p > idx:
            switched |= ru.transpose(p - 1)
            p -= 1
    return switched


def eta_of_pair(fvals: Sequence, pr: Pair):
    """eta(p, sigma, tau) = (f(sigma), f(tau)), with f(STAR) = +oo."""
    s, t = pr
    return (fvals[s], INF if t == STAR else fvals[t])


def diagram_of_pairing(cx: Complex, fvals: Sequence, P: Pairing,
                       dim: Optional[int] = None, mode: str = "A") -> List[tuple]:
    """Multiset (sorted list) of diagram points of a pairing.
    mode 'A' keeps diagonal points, mode 'D' drops them (Def. 2.12)."""
    pts = []
    for pr in P:
        if dim is not None and cx.dim[pr[0]] != dim:
            continue
        b, d = eta_of_pair(fvals, pr)
        if mode == "D" and b == d:
            continue
        pts.append((b, d))
    return sorted(pts)



# --------------------------------------------------------------------------
# Total consistent orders (Def. 2.7), enumerated lazily
# --------------------------------------------------------------------------

def value_blocks(fvals: Sequence, tol=None) -> List[List[int]]:
    """Simplex ids grouped by filtration value ("tie blocks"), sorted by value.
    tol=None means exact equality; otherwise consecutive values within tol are merged."""
    idx = sorted(range(len(fvals)), key=lambda i: fvals[i])
    blocks, cur = [], [idx[0]]
    for i in idx[1:]:
        same = (fvals[i] == fvals[cur[-1]]) if tol is None else (fvals[i] - fvals[cur[-1]] <= tol)
        if same:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)
    return blocks


def _linear_extensions(items: Sequence[int], cx: Complex) -> Iterator[Tuple[int, ...]]:
    """All linear extensions of the face relation restricted to `items` (lazy)."""
    items = list(items)
    if len(items) == 1:
        yield (items[0],)
        return
    preds = {x: frozenset(y for y in items if cx.is_face(y, x)) for x in items}

    def rec(remaining: FrozenSet[int], placed: FrozenSet[int], acc: tuple):
        if not remaining:
            yield acc
            return
        for x in sorted(remaining):
            if preds[x] <= placed:
                yield from rec(remaining - {x}, placed | {x}, acc + (x,))

    yield from rec(frozenset(items), frozenset(), ())


def total_consistent_orders(cx: Complex, fvals: Sequence, tol=None) -> Iterator[List[int]]:
    """Lazily enumerate every total consistent order of `fvals` (Def. 2.7):
    the product over tie blocks of the linear extensions of the face poset.
    Orders across blocks are forced by the values; only ties can vary."""
    blocks = value_blocks(fvals, tol)

    def rec(k: int, acc: tuple):
        if k == len(blocks):
            yield list(acc)
            return
        for ext in _linear_extensions(blocks[k], cx):
            yield from rec(k + 1, acc + ext)

    yield from rec(0, ())


# --------------------------------------------------------------------------
# Structural (non-Hausdorff) point detection -- lazy
# --------------------------------------------------------------------------

@dataclass
class Detection:
    """Result of the lazy structural-point check at one parameter value.

    ``structural`` is True iff the base pairing admits a non-identity filtration
    equivalence onto itself (Def. 2.18 with both orders equal, as for f in
    Example 2.19 of Giunti--Munch): two distinct pairs e = (s, t), e' = (s', t')
    with eta(e) = eta(e'), equal birth dimension, and a consistent order (the
    *witness*) whose pairing contains the crossed pairs (s, t') and (s', t).
    Such e, e' are non-Hausdorff partners; ``swappable`` records them.
    """
    structural: Optional[bool]        # True / False / None (search truncated by max_orders)
    orders_checked: int
    base_order: List[int]
    base_pairing: Pairing
    witness_order: Optional[List[int]] = None
    witness_pairing: Optional[Pairing] = None
    swappable: Optional[Tuple[Pair, Pair]] = None   # the two inseparable pairs
    reason: str = ""

    @property
    def differing_pairs(self) -> FrozenSet[Pair]:
        """Pairs of the base and witness pairings that differ (the switch that
        realizes the crossed pairing)."""
        if not self.structural or self.witness_pairing is None:
            return frozenset()
        return self.base_pairing ^ self.witness_pairing

    def structural_points(self, fvals: Sequence, mode: str = "A") -> set:
        """Structural diagram points (b, d): eta of the inseparable pairs.
        mode 'A': A-canopy notion (Def. 3.15), diagonal included.
        mode 'D': D-canopy notion (Def. 4.3), diagonal dropped."""
        pts = set()
        if not self.structural or self.swappable is None:
            return pts
        for pr in self.swappable:
            b, d = eta_of_pair(fvals, pr)
            if mode == "D" and b == d:
                continue
            pts.add((b, d))
        return pts

    def describe(self, cx: Complex) -> str:
        if self.structural is None:
            return f"inconclusive after {self.orders_checked} orders"
        if not self.structural:
            return f"not structural after {self.orders_checked} orders ({self.reason})"
        e, e2 = self.swappable
        out = (f"STRUCTURAL after {self.orders_checked} orders: "
               f"{cx.pair_name(e)} and {cx.pair_name(e2)} are inseparable\n"
               f"  base order {[cx.name(i) for i in self.base_order]}\n"
               f"    -> {format_pairing(cx, self.base_pairing)}")
        if self.witness_order is not None:
            out += (f"\n  witness order {[cx.name(i) for i in self.witness_order]}\n"
                    f"    -> {format_pairing(cx, self.witness_pairing)}")
        return out


def _multiplicity_classes(cx: Complex, fvals: Sequence, P: Pairing, eq) -> List[List[Pair]]:
    """Pairs of P grouped by (eta, birth dimension); only groups of size >= 2.
    A non-Hausdorff partner of e must have the same eta as e (eta : E -> H is
    continuous, Def. 3.1, and H is Hausdorff), so these are the only candidates."""
    groups: List[List[Pair]] = []
    keys: List[tuple] = []
    for pr in sorted(P, key=_pair_key):
        b, d = eta_of_pair(fvals, pr)
        dm = cx.dim[pr[0]]
        for k, (b2, d2, dm2) in enumerate(keys):
            if dm == dm2 and eq(b, b2) and ((d == INF and d2 == INF) or (d != INF and d2 != INF and eq(d, d2))):
                groups[k].append(pr)
                break
        else:
            keys.append((b, d, dm))
            groups.append([pr])
    return [g for g in groups if len(g) >= 2]


def _crossed_witness(groups: List[List[Pair]], Q: Pairing) -> Optional[Tuple[Pair, Pair]]:
    """Return (e, e') from one group whose crossed pairs both lie in Q, if any."""
    for g in groups:
        for (s, t), (s2, t2) in combinations(g, 2):
            if (s, t2) in Q and (s2, t) in Q:
                return (s, t), (s2, t2)
    return None


def detect_structural(cx: Complex, fvals: Sequence, tol=None,
                      max_orders: Optional[int] = None,
                      transposition_prepass: bool = True,
                      ru: Optional[RUDecomposition] = None,
                      essential_vacuous: bool = False) -> Detection:
    """Lazily decide whether the parameter with filtration values `fvals` is a
    structural (non-Hausdorff) point of the A-canopy (Def. 3.15).

    Criterion (Def. 2.18 with both orders equal, Example 2.19 for f): p is
    structural iff the base pairing P0 contains two distinct pairs
    e = (s, t), e' = (s', t') with
      (i)  eta(e) = eta(e') and equal birth dimension  -- a diagram point of
           multiplicity >= 2 (necessary: eta is continuous, Def. 3.1), and
      (ii) some total consistent order of g_p whose pairing contains the
           crossed pairs (s, t') and (s', t)  -- the witness.
    Then swapping e and e' is a filtration equivalence of P0 with itself, so
    e and e' lie in every neighbourhood of each other.

    NOTE. "Two consistent orders give different pairings" is necessary but
    NOT sufficient: a single tie with a vineyard switch changes the pairing
    while the induced equivalence is unique and eta separates the fiber
    points (Example 2.19 for g; Example 3.14 calls such ties benign).

    Essential pairs (t = t' = STAR): the crossed pairs equal the original
    ones, so (ii) is vacuous.  Read literally, Def. 2.18 then makes any two
    essential classes born at the same value inseparable -- which would make
    Example 3.16 structural through its two (0, inf) points, contradicting the
    paper's reading of that example ("separate connected components").  By
    default (essential_vacuous=False) equal-eta essential pairs are therefore
    NOT treated as inseparable; set essential_vacuous=True for the literal
    reading.

    Laziness: if no diagram point has multiplicity >= 2 the answer is NOT
    structural immediately (one reduction).  Otherwise consistent orders are
    enumerated (CEM transpositions on one RU-decomposition) until a witness is
    found; proving NOT structural (coincidental multiplicity, Ex. 3.16)
    exhausts the tie-block extensions.  `max_orders` truncates the search,
    returning structural=None.

    `ru`, if given, must hold a decomposition whose order is consistent with
    `fvals`; it is copied, so the caller's decomposition is untouched.
    """
    check_filtration(cx, fvals)
    if ru is None:
        work = RUDecomposition(cx, next(total_consistent_orders(cx, fvals, tol)))
    else:
        work = ru.copy()
    base = list(work.order)
    P0 = work.pairing()
    v0 = work.version
    checked = 1
    eq = (lambda a, b: a == b) if tol is None else (lambda a, b: abs(a - b) <= tol)

    groups = _multiplicity_classes(cx, fvals, P0, eq)
    if essential_vacuous:
        for g in groups:                   # literal Def. 2.18: (ii) is vacuous
            ess = [pr for pr in g if pr[1] == STAR]
            if len(ess) >= 2:
                return Detection(True, checked, base, P0, swappable=(ess[0], ess[1]),
                                 reason="equal-eta essential classes (literal Def. 2.18)")
    groups = [g for g in ([pr for pr in g if pr[1] != STAR] for g in groups) if len(g) >= 2]
    if not groups:
        return Detection(False, checked, base, P0,
                         reason="no finite diagram point of multiplicity >= 2")
    if transposition_prepass:
        for k in range(cx.n - 1):
            i, j = work.order[k], work.order[k + 1]
            if eq(fvals[i], fvals[j]) and not cx.is_face(i, j) and not cx.is_face(j, i):
                switched = work.transpose(k)
                checked += 1
                if switched:
                    Q = work.pairing()
                    hit = _crossed_witness(groups, Q)
                    if hit is not None:
                        return Detection(True, checked, base, P0, list(work.order), Q,
                                         swappable=hit, reason="witness order found")
                work.transpose(k)          # revert (pairing is a function of the order)
    for order in total_consistent_orders(cx, fvals, tol):
        if order == base:
            continue
        if max_orders is not None and checked >= max_orders:
            return Detection(None, checked, base, P0, reason="max_orders reached")
        _bubble_to(work, order)
        checked += 1
        if work.version != v0:             # a switch happened somewhere en route
            Q = work.pairing()
            if Q != P0:
                hit = _crossed_witness(groups, Q)
                if hit is not None:
                    return Detection(True, checked, base, P0, list(order), Q,
                                     swappable=hit, reason="witness order found")
    return Detection(False, checked, base, P0,
                     reason="multiplicity is coincidental: no witness order")


# --------------------------------------------------------------------------
# Piecewise-linear fibered filtration functions on B = [0,1]
# --------------------------------------------------------------------------

class PLFiltration:
    """A fibered filtration function g : K x [0,1] -> R with each g_sigma
    piecewise linear.

    `spec` maps a simplex (label string or vertex tuple) either to a constant
    value or to a list of (t, value) breakpoints whose times span [0, 1].
    Values are converted to exact Fractions (floats via their decimal
    literal, e.g. 0.1 -> 1/10).
    """

    def __init__(self, cx: Complex, spec: Dict):
        self.cx = cx
        self.bp: List[Optional[List[Tuple[Fraction, Fraction]]]] = [None] * cx.n
        for key, v in spec.items():
            i = cx.id(key)
            if isinstance(v, (int, float, str, Fraction)):
                pts = [(Fraction(0), _to_frac(v)), (Fraction(1), _to_frac(v))]
            else:
                pts = sorted((_to_frac(t), _to_frac(val)) for t, val in v)
                if pts[0][0] != 0 or pts[-1][0] != 1:
                    raise ValueError(f"breakpoints for {cx.name(i)} must span t=0..1")
            self.bp[i] = pts
        missing = [cx.name(i) for i in range(cx.n) if self.bp[i] is None]
        if missing:
            raise ValueError(f"no values given for simplices: {missing}")
        # validate the filtration property at the union of all breakpoint
        # times (sufficient for PL functions).
        knots = sorted({t for pts in self.bp for t, _ in pts})
        for t in knots:
            check_filtration(cx, self.values(t))

    def value(self, i: int, t: Fraction) -> Fraction:
        pts = self.bp[i]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return v0
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        raise ValueError(f"t={t} outside [0,1]")

    def values(self, t) -> List[Fraction]:
        t = _to_frac(t)
        return [self.value(i, t) for i in range(self.cx.n)]

    def crossings(self, return_pairs: bool = False):
        """Candidate event times: t in (0,1) where two NON-face-related
        simplices share a value.  (Face-related ties never affect the order,
        so e.g. lower-star inputs do not flood the event list.)

        Returns (event_times sorted, tie_intervals) where a tie interval
        (t0, t1, (i, j)) records g_i == g_j on the whole subinterval
        (degenerate input; order there is ambiguous everywhere).
        With return_pairs=True additionally returns {t: set of tied (i, j)},
        which lets the canopy driver update the order incrementally.
        """
        cx = self.cx
        events: Dict[Fraction, set] = {}
        intervals = []

        def add(t, i, j):
            if 0 < t < 1:
                events.setdefault(t, set()).add((i, j))

        for i in range(cx.n):
            for j in range(i + 1, cx.n):
                if cx.is_face(i, j) or cx.is_face(j, i):
                    continue
                ts = sorted({t for t, _ in self.bp[i]} | {t for t, _ in self.bp[j]})
                for t0, t1 in zip(ts, ts[1:]):
                    d0 = self.value(i, t0) - self.value(j, t0)
                    d1 = self.value(i, t1) - self.value(j, t1)
                    if d0 == 0 and d1 == 0:
                        intervals.append((t0, t1, (i, j)))
                    elif d0 == 0:
                        add(t0, i, j)
                    elif d1 == 0:
                        add(t1, i, j)
                    elif (d0 < 0) != (d1 < 0):
                        add(t0 + d0 * (t1 - t0) / (d0 - d1), i, j)
        times = sorted(events)
        if return_pairs:
            return times, intervals, events
        return times, intervals


# --------------------------------------------------------------------------
# Trajectories and the 1-parameter canopy
# --------------------------------------------------------------------------

def _valid_matchings(removed: Iterable[Pair], added: Iterable[Pair],
                     cx: Complex, fvals: Sequence) -> List[Dict[Pair, Pair]]:
    """All bijections removed -> added that can extend trajectories through
    the event: matched pairs must have equal eta at the event time (the
    filtration-equivalence constraint of Def. 2.18) and equal birth-simplex
    dimension."""
    removed = sorted(removed, key=_pair_key)
    out: List[Dict[Pair, Pair]] = []

    def rec(k: int, remaining: set, acc: list):
        if k == len(removed):
            out.append(dict(acc))
            return
        r = removed[k]
        for a in sorted(remaining, key=_pair_key):
            if (eta_of_pair(fvals, r) == eta_of_pair(fvals, a)
                    and cx.dim[r[0]] == cx.dim[a[0]]):
                remaining.remove(a)
                acc.append((r, a))
                rec(k + 1, remaining, acc)
                acc.pop()
                remaining.add(a)

    rec(0, set(added), [])
    return out


def _choose_matching(matchings: List[Dict[Pair, Pair]], policy: str) -> Dict[Pair, Pair]:
    def score(m):
        s = 0
        for r, a in m.items():
            if policy == "follow_death" and r[1] == a[1]:
                s += 1
            if policy == "follow_birth" and r[0] == a[0]:
                s += 1
        return s

    def det_key(m):
        return tuple(sorted((_pair_key(r), _pair_key(a)) for r, a in m.items()))

    return max(matchings, key=lambda m: (score(m), det_key(m)))


@dataclass
class SwitchEvent:
    """One crossing event of the path.  If `removed` is non-empty the pairing
    changed across the event (a switch).  A switch alone does NOT make the
    event structural: for a single tie the continuation is forced and the
    fiber points stay separated by eta (Example 2.19 for g / Example 3.14 of
    Giunti--Munch).  The event is structural iff the full detection at t says
    so (deep=True), or, without detection, iff more than one trajectory
    continuation is valid (`matchings` has more than one element)."""
    t: Fraction
    old_pairing: Pairing
    new_pairing: Pairing
    removed: Pairing
    added: Pairing
    matchings: List[Dict[Pair, Pair]]
    chosen: Optional[Dict[Pair, Pair]]
    detection: Optional[Detection]     # full lazy check at exactly t (may catch latent cases)

    @property
    def is_switch(self) -> bool:
        return bool(self.removed)

    @property
    def structural(self) -> bool:
        if self.detection is not None:
            return bool(self.detection.structural)
        return self.is_switch and not self.forced

    @property
    def forced(self) -> bool:
        return len(self.matchings) <= 1

    def structural_points(self, fvals: Sequence, mode: str = "A") -> set:
        if not self.structural:
            return set()
        if self.detection is not None:
            return self.detection.structural_points(fvals, mode)
        pts = set()                        # deep=False: pairs with >1 valid target
        for r in self.removed:
            if len({m[r] for m in self.matchings}) > 1:
                b, d = eta_of_pair(fvals, r)
                if mode == "D" and b == d:
                    continue
                pts.add((b, d))
        return pts


@dataclass
class Trajectory:
    """A trajectory (Def. 3.6) presented as a section over [0,1]:
    piecewise-constant simplex pairs, switching only at events."""
    starts: List[Tuple[Fraction, Pair]]      # (t_start, (sigma, tau)), first t_start = 0
    t_end: Fraction = Fraction(1)

    def segments(self) -> List[Tuple[Fraction, Fraction, Pair]]:
        out = []
        for k, (t0, pr) in enumerate(self.starts):
            t1 = self.starts[k + 1][0] if k + 1 < len(self.starts) else self.t_end
            out.append((t0, t1, pr))
        return out

    def pair_at(self, t) -> Pair:
        t = _to_frac(t)
        pr = self.starts[0][1]
        for t0, p in self.starts:
            if t0 <= t:
                pr = p
            else:
                break
        return pr

    def describe(self, cx: Complex) -> str:
        return "  ->  ".join(f"[{t0},{t1}]:{cx.pair_name(pr)}" for t0, t1, pr in self.segments())


@dataclass
class Canopy:
    """The computed 1-parameter A-canopy: one chosen section per fiber point,
    all events, and the non-Hausdorff (structural) information."""
    cx: Complex
    plf: PLFiltration
    events: List[SwitchEvent]
    trajectories: List[Trajectory]
    tie_intervals: List[Tuple[Fraction, Fraction, Tuple[int, int], Optional[Detection]]]
    policy: str

    def structural_events(self) -> List[SwitchEvent]:
        return [e for e in self.events if e.structural]

    def first_structural(self) -> Optional[SwitchEvent]:
        for e in self.events:
            if e.structural:
                return e
        return None

    def diagram(self, t, dim: Optional[int] = None, mode: str = "A") -> List[tuple]:
        """A- or D-diagram at time t (well defined independently of order
        choices by Lemma 2.14)."""
        fv = self.plf.values(t)
        P = pairing_for_order(self.cx, next(total_consistent_orders(self.cx, fv)))
        return diagram_of_pairing(self.cx, fv, P, dim=dim, mode=mode)

    def structural_points(self, mode: str = "A") -> List[tuple]:
        """All detected structural points as (t, {(b, d), ...})."""
        out = []
        for e in self.structural_events():
            pts = e.structural_points(self.plf.values(e.t), mode)
            if pts:
                out.append((e.t, pts))
        return out

    def summary(self) -> str:
        cx = self.cx
        lines = [f"events: {len(self.events)}   structural: {len(self.structural_events())}"
                 f"   policy: {self.policy}"]
        for e in self.events:
            if e.is_switch:
                alt = "FORCED" if e.forced else f"{len(e.matchings)} valid continuations (choice!)"
                kind = ("structural, non-Hausdorff" if e.structural
                        else "not structural: Hausdorff")
                lines.append(f"  t={e.t}: SWITCH ({kind}) {alt}")
                lines.append(f"     {format_pairing(cx, e.removed)}  ->  {format_pairing(cx, e.added)}")
                for m in e.matchings:
                    tag = " (chosen)" if m == e.chosen else ""
                    lines.append("       " + ", ".join(
                        f"{cx.pair_name(r)}->{cx.pair_name(a)}" for r, a in sorted(m.items(), key=lambda kv: _pair_key(kv[0]))) + tag)
            elif e.structural:
                lines.append(f"  t={e.t}: latent structural point (pairing continuous on the "
                             f"chosen section, but some order at t realizes a crossed pairing)")
            else:
                lines.append(f"  t={e.t}: benign crossing (not structural)")
        for (t0, t1, (i, j), det) in self.tie_intervals:
            s = det.structural if det else "?"
            lines.append(f"  tie interval [{t0},{t1}] on ({cx.name(i)},{cx.name(j)}): structural={s}")
        return "\n".join(lines)

    # -- plotting ----------------------------------------------------------
    def plot(self, dim: Optional[int] = None, fname: Optional[str] = None, npoints: int = 200):
        """Vineyard-style plot: birth and death curves per trajectory, with
        structural events marked by dashed vertical lines."""
        import matplotlib
        if fname:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        knots = sorted({float(e.t) for e in self.events}
                       | {k / (npoints - 1) for k in range(npoints)})
        fig, (axb, axd) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        cmap = plt.get_cmap("tab10")
        ci = 0
        for tr in self.trajectories:
            s0 = tr.starts[0][1][0]
            if dim is not None and self.cx.dim[s0] != dim:
                continue
            color = cmap(ci % 10)
            ci += 1
            bs, ds, ts = [], [], []
            for t in knots:
                fv = self.plf.values(Fraction(t).limit_denominator(10 ** 9))
                b, d = eta_of_pair(fv, tr.pair_at(Fraction(t).limit_denominator(10 ** 9)))
                ts.append(t)
                bs.append(float(b))
                ds.append(None if d == INF else float(d))
            lbl = self.cx.pair_name(tr.starts[0][1]) + " -> " + self.cx.pair_name(tr.starts[-1][1])
            axb.plot(ts, bs, color=color, label=lbl)
            if any(d is not None for d in ds):
                axd.plot(ts, [d for d in ds], color=color)
        for e in self.structural_events():
            for ax in (axb, axd):
                ax.axvline(float(e.t), ls="--", c="k", alpha=0.5)
        axb.set_title("births  f(sigma)(t)")
        axd.set_title("deaths  f(tau)(t)")
        axb.set_xlabel("t")
        axd.set_xlabel("t")
        axb.legend(fontsize=7)
        fig.tight_layout()
        if fname:
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            return fname
        return fig


def _order_after_event(plf: PLFiltration, order: Sequence[int], pos: Sequence[int],
                       tied_pairs: Iterable[Tuple[int, int]], t: Fraction,
                       t_next: Fraction) -> List[int]:
    """The consistent order of the segment just after event t, computed
    incrementally from the left-limit order: only simplices tied at t can
    move.  Tied pairs are merged into groups (union-find); within each group
    the new order is the value order just after t (evaluated at the next
    segment midpoint, for group members only), stable under persistent ties
    so face-forced relative positions are preserved.  Cost: O(|tied| + g log g)
    exact-value operations instead of a full O(n log n) re-sort."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in tied_pairs:
        parent.setdefault(i, i)
        parent.setdefault(j, j)
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: Dict[int, List[int]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    mid = (t + t_next) / 2
    new_order = list(order)
    for members in groups.values():
        slots = sorted(pos[x] for x in members)
        members.sort(key=lambda x: (plf.value(x, mid), pos[x]))  # stable in pos
        for p, x in zip(slots, members):
            new_order[p] = x
    return new_order


def compute_canopy(cx: Complex, plf: PLFiltration, policy: str = "follow_death",
                   deep: bool = True) -> Canopy:
    """Compute the 1-parameter A-canopy of a PL fibered filtration function.

    * Segments between crossing events carry a constant total consistent
      order, hence a constant chosen pairing (recomputed per segment; the
      diagram itself is order-independent by Lemma 2.14).
    * At each event the old/new pairings are compared.  A change (switch)
      does NOT by itself make the event structural (a single tie switches
      the pairing but the continuation is forced).  All valid
      continuations are enumerated; `policy` ('follow_death'/'follow_birth')
      picks one so a concrete section is always produced, exactly as
      Morozov-style vineyard algorithms implicitly do.
    * With deep=True, a full lazy detection runs at every event time, which
      decides structurality exactly (Detection) and also catches latent
      structural points (pairing unchanged across t, e.g. tangential ties).
      With deep=False an event is reported structural iff more than one
      continuation is valid.
    """
    event_times, raw_intervals, pairs_at = plf.crossings(return_pairs=True)
    knots = [Fraction(0)] + list(event_times) + [Fraction(1)]

    # one RU-decomposition maintained along the whole path: O(n^3/w) once,
    # then O(#inversions * n/w) per event via CEM transpositions.  The order
    # is also updated incrementally: only the simplices tied at an event can
    # move, so no per-segment re-sort of all n exact values is needed.
    mid0 = (knots[0] + knots[1]) / 2
    ru = RUDecomposition(cx, next(total_consistent_orders(cx, plf.values(mid0))))
    seg_pairings: List[Pairing] = [ru.pairing()]

    trajs: Dict[Pair, Trajectory] = {}
    all_trajs: List[Trajectory] = []
    for pr in sorted(seg_pairings[0], key=_pair_key):
        tr = Trajectory([(Fraction(0), pr)])
        trajs[pr] = tr
        all_trajs.append(tr)

    events: List[SwitchEvent] = []
    for k, t in enumerate(event_times):
        fv_t = plf.values(t) if deep else None
        # the left-limit order held by `ru` is consistent AT t (ties relax
        # constraints), so the detection can reuse the decomposition (copied)
        det = detect_structural(cx, fv_t, ru=ru) if deep else None
        target = _order_after_event(plf, ru.order, ru.pos, pairs_at[t], t, knots[k + 2])
        switched = _bubble_to(ru, target)
        # rebuild the pairing only if some transposition actually switched
        seg_pairings.append(ru.pairing() if switched else seg_pairings[-1])
        Pold, Pnew = seg_pairings[k], seg_pairings[k + 1]
        removed = frozenset(Pold - Pnew)
        added = frozenset(Pnew - Pold)
        matchings: List[Dict[Pair, Pair]] = []
        chosen = None
        if removed:
            if fv_t is None:
                fv_t = plf.values(t)
            matchings = _valid_matchings(removed, added, cx, fv_t)
            if not matchings:
                raise RuntimeError(f"no valid trajectory continuation at t={t}; "
                                   f"this should not happen")
            chosen = _choose_matching(matchings, policy)
            for r in sorted(removed, key=_pair_key):
                tr = trajs.pop(r)
                a = chosen[r]
                tr.starts.append((t, a))
                trajs[a] = tr
        events.append(SwitchEvent(t, Pold, Pnew, removed, added, matchings, chosen, det))

    tie_intervals = []
    for (t0, t1, ij) in raw_intervals:
        mid = (t0 + t1) / 2
        det = detect_structural(cx, plf.values(mid)) if deep else None
        if det and det.structural:
            warnings.warn(f"degenerate tie interval [{t0},{t1}] on "
                          f"({cx.name(ij[0])},{cx.name(ij[1])}) is structural: the order "
                          f"is ambiguous on a whole interval")
        tie_intervals.append((t0, t1, ij, det))

    return Canopy(cx, plf, events, all_trajs, tie_intervals, policy)


def find_first_structural(cx: Complex, plf: PLFiltration, deep: bool = True):
    """Fully lazy scan: walk the crossing events in time order and return the
    first structural (non-Hausdorff) point as (t, Detection), or None.  Stops
    as soon as one witness is found -- no further orderings, events or
    trajectories are examined."""
    event_times, intervals = plf.crossings()
    candidates = sorted(set(event_times) | {(t0 + t1) / 2 for t0, t1, _ in intervals})
    for t in candidates:
        det = detect_structural(cx, plf.values(t))
        if det.structural:
            return t, det
    return None
