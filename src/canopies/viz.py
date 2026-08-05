# -*- coding: utf-8 -*-
"""
canopy_viz.py -- richer visualizations for 1-parameter canopies.

* plot_vineyard_3d : classic CEM vineyard picture: vines (b(t), d(t), t)
                     in the plane x time, with the diagonal wall b = d and
                     structural (non-Hausdorff) events marked.
* plot_persistence : persistence d(t) - b(t) per trajectory; 0 = diagonal,
                     so "cycles opening and closing" is read directly.
* plot_snapshots   : for chosen times, the EMBEDDED simplicial complex
                     (vertex positions given by the user) colored by the
                     current filtration values, next to the diagram at t.
* animate          : the same two panels as an animated GIF over t.

All functions take the `Canopy` returned by compute_canopy.  `positions`
maps each vertex id to (x, y).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .core import (Canopy, Complex, INF, STAR, eta_of_pair,
                    total_consistent_orders)

Point = Tuple[float, float]


def _frac(t) -> Fraction:
    return t if isinstance(t, Fraction) else Fraction(t).limit_denominator(10 ** 9)


def _use_agg(fname):
    import matplotlib
    if fname:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _traj_list(can: Canopy, dim: Optional[int]):
    return [tr for tr in can.trajectories
            if dim is None or can.cx.dim[tr.starts[0][1][0]] == dim]


def _sample_traj(can: Canopy, tr, npoints: int):
    """Sampled (t, b, d) along a trajectory, densified at events/breakpoints."""
    knots = sorted({float(e.t) for e in can.events}
                   | {k / (npoints - 1) for k in range(npoints)})
    ts, bs, ds = [], [], []
    for t in knots:
        tf = _frac(t)
        fv = can.plf.values(tf)
        b, d = eta_of_pair(fv, tr.pair_at(tf))
        ts.append(t)
        bs.append(float(b))
        ds.append(None if d == INF else float(d))
    return ts, bs, ds


def _finish(fig, fname, plt):
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        return fname
    return fig


# --------------------------------------------------------------------------
# vineyard views
# --------------------------------------------------------------------------

def _draw_vineyard_3d(ax, can: Canopy, dim, npoints, plt, legend=True):
    """Draw the vines (b(t), d(t), t) and the diagonal wall on a 3-D axes.
    Returns the shared (lo, hi) limit used for both the birth and death axes,
    so callers that redraw (e.g. the animation) keep a fixed frame."""
    import numpy as np
    cmap = plt.get_cmap("tab10")
    lo, hi = float("inf"), float("-inf")
    samples = []
    for tr in _traj_list(can, dim):
        ts, bs, ds = _sample_traj(can, tr, npoints)
        samples.append((tr, ts, bs, ds))
        vals = bs + [d for d in ds if d is not None]
        lo, hi = min([lo] + vals), max([hi] + vals)
    pad = 0.05 * (hi - lo or 1)
    lo, hi = lo - pad, hi + pad
    # diagonal wall b = d
    s = np.linspace(lo, hi, 2)
    T = np.linspace(0, 1, 2)
    S, TT = np.meshgrid(s, T)
    ax.plot_surface(S, S, TT, color="0.85", alpha=0.35, linewidth=0)
    for ci, (tr, ts, bs, ds) in enumerate(samples):
        color = cmap(ci % 10)
        segb, segd, segt = [], [], []
        for t, b, d in zip(ts, bs, ds):
            if d is None:
                continue
            segb.append(b)
            segd.append(d)
            segt.append(t)
        lbl = can.cx.pair_name(tr.starts[0][1]) + " -> " + can.cx.pair_name(tr.starts[-1][1])
        ax.plot(segb, segd, segt, color=color, lw=2, label=lbl)
    for e in can.structural_events():
        fv = can.plf.values(e.t)
        for pr in set(e.removed) | set(e.added):
            b, d = eta_of_pair(fv, pr)
            if d != INF:
                ax.scatter([float(b)], [float(d)], [float(e.t)],
                           color="red", s=45, depthshade=False, zorder=5)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_zlabel("t")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if legend:
        ax.legend(fontsize=7, loc="upper left")
    return lo, hi


def plot_vineyard_3d(can: Canopy, dim: Optional[int] = None, fname: Optional[str] = None,
                     npoints: int = 160, elev: float = 22, azim: float = -55):
    """Vines (b(t), d(t), t) in R^2 x [0,1] with the diagonal wall b = d.
    Structural events are marked with red dots on the affected vines."""
    plt = _use_agg(fname)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    _draw_vineyard_3d(ax, can, dim, npoints, plt)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("vineyard: vines (b(t), d(t), t); red = structural (non-Hausdorff)")
    return _finish(fig, fname, plt)


def animate_vineyard_3d(can: Canopy, fname: str, dim: Optional[int] = None,
                        npoints: int = 160, frames: int = 40, fps: int = 12,
                        elev: float = 22, azim0: float = -55,
                        figsize=(5.4, 4.6), dpi: int = 70, colors: int = 64):
    """Animated GIF of the 3-D vineyard with the camera orbiting a full turn.

    The geometry is drawn once and only the viewing angle changes per frame, so
    this costs one render of the vines plus `frames` rasterizations. Rotation
    is what makes the (birth, death, time) picture readable on a flat page: a
    single viewpoint hides which vines cross and which merely pass behind one
    another at a structural event.

    `dpi` and `colors` trade file size against fidelity -- the defaults aim at
    a GIF small enough to embed in a README. Set `colors=0` to skip the palette
    quantization pass (which needs pillow).
    """
    plt = _use_agg(fname)
    import matplotlib.animation as manim
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    _draw_vineyard_3d(ax, can, dim, npoints, plt, legend=False)
    ax.tick_params(labelsize=6, pad=0)
    for lab in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lab.set_size(7)
    # the default 3-D axes leave a wide margin; reclaim it so more of the GIF
    # is actually plot (smaller file at the same legibility). The legend goes
    # on the figure rather than the axes so the orbiting box never covers it.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=1.0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)),
                   fontsize=6, frameon=False, borderaxespad=0.2,
                   columnspacing=1.2, handlelength=1.4)
    fig.text(0.5, 0.985, "vines (b(t), d(t), t);  red = structural (non-Hausdorff)",
             ha="center", va="top", fontsize=8)

    def draw(k):
        ax.view_init(elev=elev, azim=azim0 + 360.0 * k / frames)
        return []

    ani = manim.FuncAnimation(fig, draw, frames=frames, blit=False)
    ani.save(fname, writer=manim.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    if colors:
        _shrink_gif(fname, colors=colors)
    return fname


def _shrink_gif(fname: str, colors: int = 96):
    """Re-encode a GIF with an adaptive palette. Roughly halves the size of the
    vineyard animation; silently does nothing if pillow is unavailable."""
    try:
        from PIL import Image
    except ImportError:                                  # pragma: no cover
        return fname
    im = Image.open(fname)
    frames, durations = [], []
    try:
        while True:
            durations.append(im.info.get("duration", 80))
            frames.append(im.convert("RGB").quantize(
                colors=colors, method=Image.MEDIANCUT, dither=Image.NONE))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    if frames:
        frames[0].save(fname, save_all=True, append_images=frames[1:],
                       loop=0, duration=durations, optimize=True)
    return fname


def plot_persistence(can: Canopy, dim: Optional[int] = None, fname: Optional[str] = None,
                     npoints: int = 300):
    """Persistence d(t) - b(t) of each trajectory over t.  Touching 0 means
    the cycle is closed (on the diagonal); dashed lines = structural events."""
    plt = _use_agg(fname)
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.get_cmap("tab10")
    for ci, tr in enumerate(_traj_list(can, dim)):
        ts, bs, ds = _sample_traj(can, tr, npoints)
        pers = [(d - b) if d is not None else None for b, d in zip(bs, ds)]
        if all(p is None for p in pers):
            continue
        lbl = can.cx.pair_name(tr.starts[0][1]) + " -> " + can.cx.pair_name(tr.starts[-1][1])
        ax.plot(ts, pers, color=cmap(ci % 10), lw=2, label=lbl)
    ax.axhline(0, color="k", lw=0.8)
    for e in can.structural_events():
        ax.axvline(float(e.t), ls="--", color="red", alpha=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("persistence  d - b")
    ax.set_title("cycles opening/closing over time (0 = on the diagonal)")
    ax.legend(fontsize=8)
    return _finish(fig, fname, plt)


# --------------------------------------------------------------------------
# embedded complex snapshots
# --------------------------------------------------------------------------

def _draw_complex(ax, cx: Complex, positions: Dict[int, Point], fvals,
                  norm, cmap, label: bool = True):
    from matplotlib.patches import Polygon
    grey = (0.62, 0.62, 0.62)

    def color(v):
        v = float(v)
        return grey if v < 0 else cmap(norm(v))

    for i in range(cx.n):        # triangles first (background)
        if cx.dim[i] == 2:
            pts = [positions[v] for v in cx.simplices[i]]
            ax.add_patch(Polygon(pts, closed=True, facecolor=color(fvals[i]),
                                 alpha=0.5, edgecolor="none", zorder=1))
    for i in range(cx.n):
        if cx.dim[i] == 1:
            (x1, y1), (x2, y2) = (positions[v] for v in cx.simplices[i])
            base = float(fvals[i]) < 0
            ax.plot([x1, x2], [y1, y2],
                    color=grey if base else color(fvals[i]),
                    lw=1.4 if base else 3.2, zorder=2,
                    solid_capstyle="round")
    for i in range(cx.n):
        if cx.dim[i] == 0:
            x, y = positions[cx.simplices[i][0]]
            ax.plot(x, y, "o", color="k", ms=4, zorder=3)
    if label:
        for i, name in cx.label_of.items():
            pts = [positions[v] for v in cx.simplices[i]]
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            ax.annotate(f"{name}={float(fvals[i]):.3g}", (mx, my), fontsize=8,
                        ha="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  alpha=0.8, ec="none"))
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_diagram(ax, pts, lims, title):
    lo, hi = lims
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    for (b, d), m in Counter(pts).items():
        if d == INF:
            continue
        c = "tab:red" if m > 1 else "tab:blue"
        ax.plot(float(b), float(d), "o", ms=8, color=c)
        if m > 1:
            ax.annotate(f"x{m}", (float(b), float(d)), textcoords="offset points",
                        xytext=(7, 4), fontsize=9, color="tab:red")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def _norm_over(can: Canopy, times) -> Tuple[float, float]:
    vmax = 0.0
    for t in times:
        for v in can.plf.values(_frac(t)):
            vmax = max(vmax, float(v))
    return 0.0, vmax


def plot_snapshots(can: Canopy, positions: Dict[int, Point], times: Sequence,
                   dim: Optional[int] = 1, fname: Optional[str] = None,
                   mode: str = "A"):
    """One row per time: the embedded complex colored by the filtration values
    g_t (grey = already-present base simplices), and the persistence diagram
    at t (red x m = multiplicity).  Colors share a global scale."""
    plt = _use_agg(fname)
    from matplotlib import colors as mcolors
    from matplotlib import cm
    times = [_frac(t) for t in times]
    _, vmax = _norm_over(can, times)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    # global diagram limits
    dlo, dhi = 0.0, 0.0
    diagrams = []
    for t in times:
        pts = can.diagram(t, dim=dim, mode=mode)
        diagrams.append(pts)
        for b, d in pts:
            if d != INF:
                dhi = max(dhi, float(d))
            dlo = min(dlo, float(b))
    pad = 0.06 * (dhi - dlo or 1)
    lims = (dlo - pad, dhi + pad)
    n = len(times)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 3.1 * n),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(hspace=0.45)
    if n == 1:
        axes = [axes]
    struct_t = {e.t for e in can.structural_events()}
    for row, (t, pts) in enumerate(zip(times, diagrams)):
        axc, axd = axes[row]
        _draw_complex(axc, can.cx, positions, can.plf.values(t), norm, cmap)
        tag = "   [STRUCTURAL]" if t in struct_t else ""
        axc.set_title(f"t = {t}{tag}", fontsize=10,
                      color="red" if t in struct_t else "black")
        _draw_diagram(axd, pts, lims, f"dgm at t = {t} (dim {dim})")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=[a for row in axes for a in row], shrink=0.5,
                 label="filtration value g_t")
    if fname:
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        return fname
    return fig


# --------------------------------------------------------------------------
# sublevel-set view: simplices appear/disappear at a fixed level s
# --------------------------------------------------------------------------

def _reduce_with_reps(cx: Complex, fvals):
    """Plain reduction that also tracks V (R = D.V), so every homology class
    has an explicit representative chain:
      * a killed class (sigma, tau): the reduced column R[tau] is a cycle
        born at sigma, and all its simplices have value <= f(sigma);
      * an essential class (sigma, *): V[sigma] is a cycle containing sigma.
    Returns (pairs, reps) with reps[pair] = list of simplex ids."""
    order = next(total_consistent_orders(cx, fvals))
    pos = {s: k for k, s in enumerate(order)}
    R: Dict[int, int] = {}
    V: Dict[int, int] = {}
    lowmap: Dict[int, int] = {}          # pivot position -> column simplex
    pairs, reps = [], {}
    for j in order:
        col = 0
        for f in cx.facets[j]:
            col ^= 1 << pos[f]
        v = 1 << pos[j]
        while col:
            piv = col.bit_length() - 1
            if piv not in lowmap:
                break
            k = lowmap[piv]
            col ^= R[k]
            v ^= V[k]
        R[j], V[j] = col, v
        if col:
            lowmap[col.bit_length() - 1] = j
    def bits(mask):
        out = []
        while mask:
            b = mask & -mask
            out.append(order[b.bit_length() - 1])
            mask ^= b
        return out
    for piv, tau in lowmap.items():
        sigma = order[piv]
        pairs.append((sigma, tau))
        reps[(sigma, tau)] = bits(R[tau])
    killed = {order[p] for p in lowmap}
    killers = set(lowmap.values())
    for s in order:
        if s not in killed and s not in killers:
            pairs.append((s, STAR))
            reps[(s, STAR)] = bits(V[s])
    return pairs, reps


def open_cycles(can: Canopy, t, level, dim: int = 1):
    """The classes of dimension `dim` alive at (t, level), i.e. with
    f(sigma) <= level < f(tau), together with representative cycles whose
    simplices are all present in the sublevel complex."""
    fv = can.plf.values(_frac(t))
    pairs, reps = _reduce_with_reps(can.cx, fv)
    out = []
    for (s, tau) in pairs:
        if can.cx.dim[s] != dim:
            continue
        b, d = eta_of_pair(fv, (s, tau))
        if b <= level < d:
            out.append(((s, tau), reps[(s, tau)]))
    return out


def _traj_color_map(can: Canopy, dim, plt):
    cmap = plt.get_cmap("tab10")
    return {id(tr): cmap(ci % 10) for ci, tr in enumerate(_traj_list(can, dim))}


def _color_of_pair(can: Canopy, t, pr, colors, dim):
    for tr in _traj_list(can, dim):
        if tr.pair_at(_frac(t)) == pr:
            return colors[id(tr)]
    return (0.3, 0.3, 0.3, 1.0)


def _draw_sublevel(ax, cx: Complex, positions: Dict[int, Point], fvals, level,
                   halos=(), label: bool = True):
    """Draw only the simplices with f <= level; absent ones are ghosted.
    `halos` = [(color, [simplex ids])] outlines representative open cycles."""
    from matplotlib.patches import Polygon
    ghost = (0.78, 0.78, 0.78)
    present_edge = (0.15, 0.15, 0.15)
    fill = (0.55, 0.68, 0.83)

    def here(i):
        return fvals[i] <= level

    for color, simps in halos:                       # cycle halos (underlay)
        for i in simps:
            if cx.dim[i] == 1:
                (x1, y1), (x2, y2) = (positions[v] for v in cx.simplices[i])
                ax.plot([x1, x2], [y1, y2], color=color, lw=9, alpha=0.4,
                        zorder=1.5, solid_capstyle="round")
    for i in range(cx.n):
        if cx.dim[i] == 2 and here(i):
            pts = [positions[v] for v in cx.simplices[i]]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fill, alpha=0.55,
                                 edgecolor="none", zorder=1))
    for i in range(cx.n):
        if cx.dim[i] == 1:
            (x1, y1), (x2, y2) = (positions[v] for v in cx.simplices[i])
            if here(i):
                ax.plot([x1, x2], [y1, y2], color=present_edge, lw=2.6,
                        zorder=2, solid_capstyle="round")
            else:
                ax.plot([x1, x2], [y1, y2], color=ghost, lw=1.2, ls=":",
                        zorder=2)
    for i in range(cx.n):
        if cx.dim[i] == 0:
            x, y = positions[cx.simplices[i][0]]
            if here(i):
                ax.plot(x, y, "o", color="k", ms=4.5, zorder=3)
            else:
                ax.plot(x, y, "o", mfc="none", mec=ghost, ms=4.5, zorder=3)
    if label:
        for i, name in cx.label_of.items():
            pts = [positions[v] for v in cx.simplices[i]]
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            ax.annotate(name, (mx, my), fontsize=10, ha="center", zorder=4,
                        color="black" if here(i) else ghost,
                        bbox=dict(boxstyle="round,pad=0.13", fc="white",
                                  alpha=0.75, ec="none"))
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_diagram_level(ax, pts_colored, lims, level, title):
    """Diagram with the viewing level marked: points with b <= s < d (open
    cycles) are drawn in their trajectory color, the rest in grey."""
    lo, hi = lims
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.axvline(level, color="0.6", lw=0.8, ls=":")
    ax.axhline(level, color="0.6", lw=0.8, ls=":")
    ax.fill_between([lo, level], [level] * 2, [hi] * 2, color="0.92", zorder=0)
    for (b, d), color in pts_colored:
        if d == INF:
            continue
        is_open = b <= level < d
        ax.plot(float(b), float(d), "o", ms=8,
                color=color if is_open else (0.65, 0.65, 0.65))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def _diag_lims(can: Canopy, times, dim, level):
    dlo, dhi = 0.0, float(level)
    for t in times:
        for b, d in can.diagram(_frac(t), dim=dim, mode="A"):
            if d != INF:
                dhi = max(dhi, float(d))
            dlo = min(dlo, float(b))
    pad = 0.06 * (dhi - dlo or 1)
    return (dlo - pad, dhi + pad)


def _sublevel_row(can, positions, t, level, dim, axc, axd, lims, colors):
    t = _frac(t)
    fv = can.plf.values(t)
    opened = open_cycles(can, t, level, dim=dim)
    halos = [(_color_of_pair(can, t, pr, colors, dim), rep) for pr, rep in opened]
    _draw_sublevel(axc, can.cx, positions, fv, level, halos=halos)
    pts_colored = []
    for tr in _traj_list(can, dim):
        pr = tr.pair_at(t)
        pts_colored.append((eta_of_pair(fv, pr), colors[id(tr)]))
    _draw_diagram_level(axd, pts_colored, lims, level,
                        f"dgm at t = {t} (dim {dim}), level s = {level}")
    return len(opened)


def plot_sublevel_snapshots(can: Canopy, positions: Dict[int, Point], times: Sequence,
                            level, dim: int = 1, fname: Optional[str] = None):
    """One row per time: LEFT the sublevel complex K_{t,s} = {f_t <= s}
    (simplices literally appear/disappear; ghosted = not yet present; open
    cycles outlined by a representative in their trajectory color), RIGHT
    the diagram at t with the level s marked (grey region = open at s)."""
    plt = _use_agg(fname)
    level = float(level)
    times = [_frac(t) for t in times]
    lims = _diag_lims(can, times, dim, level)
    n = len(times)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 3.1 * n),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(hspace=0.45)
    if n == 1:
        axes = [axes]
    colors = _traj_color_map(can, dim, plt)
    struct_t = {e.t for e in can.structural_events()}
    for row, t in enumerate(times):
        axc, axd = axes[row]
        k = _sublevel_row(can, positions, t, level, dim, axc, axd, lims, colors)
        tag = "   [STRUCTURAL]" if t in struct_t else ""
        axc.set_title(f"t = {t}   ({k} open cycle{'s' if k != 1 else ''} at s = {level}){tag}",
                      fontsize=10, color="red" if t in struct_t else "black")
    if fname:
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        return fname
    return fig


def animate_sublevel(can: Canopy, positions: Dict[int, Point], fname: str,
                     level, dim: int = 1, frames: int = 80, fps: int = 10):
    """Animated GIF of the sublevel complex K_{t,s} as t sweeps [0,1] at a
    fixed level s: simplices appear/disappear, open cycles are outlined."""
    plt = _use_agg(fname)
    import matplotlib.animation as manim
    level = float(level)
    ts = [Fraction(k, frames - 1) for k in range(frames)]
    lims = _diag_lims(can, ts, dim, level)
    colors = _traj_color_map(can, dim, plt)
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    struct_t = sorted(float(e.t) for e in can.structural_events())

    def draw(k):
        t = ts[k]
        axc.clear()
        axd.clear()
        nopen = _sublevel_row(can, positions, t, level, dim, axc, axd, lims, colors)
        near = any(abs(float(t) - s) < 1.5 / frames for s in struct_t)
        axc.set_title(f"t = {float(t):.3f}   ({nopen} open at s = {level})"
                      + ("   [STRUCTURAL]" if near else ""),
                      color="red" if near else "black")
        return []

    ani = manim.FuncAnimation(fig, draw, frames=frames, blit=False)
    ani.save(fname, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    return fname


def animate(can: Canopy, positions: Dict[int, Point], fname: str,
            dim: Optional[int] = 1, mode: str = "A", frames: int = 80, fps: int = 10):
    """Animated GIF: embedded complex + diagram sweeping t through [0,1]."""
    plt = _use_agg(fname)
    from matplotlib import colors as mcolors
    import matplotlib.animation as manim
    ts = [Fraction(k, frames - 1) for k in range(frames)]
    _, vmax = _norm_over(can, ts)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    dhi = 0.0
    for t in ts:
        for b, d in can.diagram(t, dim=dim, mode=mode):
            if d != INF:
                dhi = max(dhi, float(d))
    lims = (-0.1, dhi * 1.06)
    fig, (axc, axd) = plt.subplots(1, 2, figsize=(9.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    struct_t = sorted(float(e.t) for e in can.structural_events())

    def draw(k):
        t = ts[k]
        axc.clear()
        axd.clear()
        _draw_complex(axc, can.cx, positions, can.plf.values(t), norm, cmap)
        near = any(abs(float(t) - s) < 1.5 / frames for s in struct_t)
        axc.set_title(f"t = {float(t):.3f}" + ("   [STRUCTURAL]" if near else ""),
                      color="red" if near else "black")
        _draw_diagram(axd, can.diagram(t, dim=dim, mode=mode), lims,
                      f"dgm (dim {dim})")
        return []

    ani = manim.FuncAnimation(fig, draw, frames=frames, blit=False)
    ani.save(fname, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    return fname
