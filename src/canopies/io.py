# -*- coding: utf-8 -*-
"""
canopy_io.py -- dataset loaders for the canopies package.

Dataset format (.txt), compatible with (and extending) the format used by
A. Hickok's vineyard implementation:

    # comment lines start with '#'
    times: 0 1/4 1/3 3/4 1          # optional; default: equally spaced in [0,1]
    a: 1 4 ; 0 1/2 2/3 3/2 2        # optional 'label:' prefix
    1 2 ; -3                        # constant value = repeated for all times
    5 6 7 ; 1/2 1/2 1 7/2 1/2       # vertices of the simplex ; value samples

Each line is  "[label:] v1 v2 ... vk ; val_1 [val_2 ... val_m]"  where the
values are sampled at the m listed times and interpolated linearly between
them (exact rational arithmetic; tokens may be integers, decimals, or
fractions like 2/3).  A single value means the simplex is constant in t.

Positions file (for embedding plots):  "vertex x y" per line.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .core import Complex, PLFiltration


def _frac(tok) -> Fraction:
    return Fraction(str(tok))


def load_dataset(path: str, times: Optional[Sequence] = None
                 ) -> Tuple[Complex, PLFiltration]:
    """Load a complex and its PL fibered filtration from a .txt dataset.
    `times` overrides any 'times:' header; default is equally spaced."""
    entries: List[Tuple[Optional[str], tuple, List[Fraction]]] = []
    header_times: Optional[List[Fraction]] = None
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith(("times:", "t:")):
                header_times = [_frac(x) for x in line.split(":", 1)[1].split()]
                continue
            label = None
            if ":" in line.split(";", 1)[0]:
                label, line = (s.strip() for s in line.split(":", 1))
            if ";" not in line:
                raise ValueError(f"bad line (no ';'): {raw.rstrip()}")
            left, right = line.split(";", 1)
            verts = tuple(sorted(int(v) for v in left.split()))
            vals = [_frac(x) for x in right.split()]
            entries.append((label, verts, vals))
    if times is not None:
        ts = [_frac(t) for t in times]
    elif header_times is not None:
        ts = header_times
    else:
        m = max(len(vals) for _, _, vals in entries if len(vals) > 1)
        m = max(m, 2)
        ts = [Fraction(k, m - 1) for k in range(m)]
    if ts[0] != 0 or ts[-1] != 1 or sorted(ts) != list(ts):
        raise ValueError(f"times must increase from 0 to 1, got {ts}")
    labels = {lab: verts for lab, verts, _ in entries if lab}
    cx = Complex([verts for _, verts, _ in entries], labels or None)
    spec: Dict[tuple, list] = {}
    for _, verts, vals in entries:
        if len(vals) == 1:
            vals = vals * len(ts)
        if len(vals) != len(ts):
            raise ValueError(f"simplex {verts}: {len(vals)} values for "
                             f"{len(ts)} times")
        spec[verts] = list(zip(ts, vals))
    return cx, PLFiltration(cx, spec)


def from_vertex_functions(simplices: Sequence[Sequence[int]], times: Sequence,
                          vertex_values: Dict[int, Sequence]
                          ) -> Tuple[Complex, PLFiltration]:
    """Lower-star input: a fixed complex and, for each vertex, its scalar
    value at each time.  Simplex values are max over vertices at each sample
    time, interpolated linearly in between (a PL approximation of the true
    lower-star family; it is a valid fibered filtration since the face
    inequalities hold at the sample times and are preserved by linear
    interpolation)."""
    ts = [_frac(t) for t in times]
    cx = Complex(simplices)
    spec: Dict[tuple, list] = {}
    for s in cx.simplices:
        vals = [max(_frac(vertex_values[v][k]) for v in s)
                for k in range(len(ts))]
        spec[s] = list(zip(ts, vals))
    return cx, PLFiltration(cx, spec)


def load_positions(path: str) -> Dict[int, Tuple[float, float]]:
    """Positions file: 'vertex x y' per line ('#' comments allowed)."""
    pos: Dict[int, Tuple[float, float]] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            v, x, y = line.split()
            pos[int(v)] = (float(x), float(y))
    return pos
