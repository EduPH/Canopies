# -*- coding: utf-8 -*-
"""
canopies.cli -- run the package directly on a dataset file.

Installed as the ``canopies`` console script; ``python -m canopies`` works too.

Examples:
    canopies example_dataset.txt
    canopies example_dataset.txt --first-only
    canopies example_dataset.txt --at 1/3
    canopies example_dataset.txt --dim 1 --plots out \
             --positions example_positions.txt --level 7/4 --gif

``canopies --example`` runs the bundled example dataset without needing a path.
"""
from __future__ import annotations

import argparse
from fractions import Fraction

from . import __version__
from .core import compute_canopy, detect_structural, find_first_structural
from .io import load_dataset, load_positions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="canopies",
        description="A-canopies for PL fibered filtrations: non-Hausdorff "
                    "(structural) point detection and trajectories on B=[0,1].")
    ap.add_argument("dataset", nargs="?", default=None,
                    help="dataset .txt (format documented in canopies.io); "
                         "omit it together with --example")
    ap.add_argument("--example", action="store_true",
                    help="use the dataset bundled with the package "
                         "(and its embedding, for --plots)")
    ap.add_argument("--version", action="version",
                    version=f"canopies {__version__}")
    ap.add_argument("--times", nargs="+", default=None,
                    help="override sample times, e.g. --times 0 1/4 1/3 3/4 1")
    ap.add_argument("--dim", type=int, default=1, help="homology dimension for plots/reports")
    ap.add_argument("--policy", choices=["follow_death", "follow_birth"],
                    default="follow_death", help="trajectory continuation at switches")
    ap.add_argument("--mode", choices=["A", "D"], default="A",
                    help="A-canopy (keep diagonal) or D-canopy (drop it) reporting")
    ap.add_argument("--no-deep", action="store_true",
                    help="skip the full lazy detection at every event (faster)")
    ap.add_argument("--first-only", action="store_true",
                    help="lazy scan: stop at the FIRST non-Hausdorff point")
    ap.add_argument("--at", default=None,
                    help="only run detection at this parameter value, e.g. --at 1/3")
    ap.add_argument("--max-orders", type=int, default=None,
                    help="truncate order enumeration (returns 'inconclusive')")
    ap.add_argument("--plots", metavar="PREFIX", default=None,
                    help="write PREFIX_vineyard3d/persistence[/snapshots/sublevel].png")
    ap.add_argument("--positions", default=None,
                    help="'vertex x y' file enabling embedding plots")
    ap.add_argument("--level", default=None,
                    help="sublevel viewing level s for the embedding plots")
    ap.add_argument("--gif", action="store_true",
                    help="also write animated GIF(s) (requires --plots)")
    args = ap.parse_args(argv)

    if args.example:
        from . import example_dataset_path, example_positions_path
        args.dataset = str(example_dataset_path())
        if args.positions is None and args.plots:
            args.positions = str(example_positions_path())
    elif args.dataset is None:
        ap.error("a dataset path is required (or pass --example)")

    cx, plf = load_dataset(args.dataset, times=args.times)
    print(f"loaded {args.dataset}: {cx.n} simplices "
          f"(dims {min(cx.dim)}..{max(cx.dim)})")

    if args.at is not None:
        t = Fraction(args.at)
        det = detect_structural(cx, plf.values(t), max_orders=args.max_orders)
        print(f"\ndetection at t = {t}:")
        print(det.describe(cx))
        if det.structural:
            pts = det.structural_points(plf.values(t), mode=args.mode)
            print(f"structural diagram points ({args.mode}): {sorted(pts)}")
        return 0

    if args.first_only:
        res = find_first_structural(cx, plf)
        if res is None:
            print("no non-Hausdorff point on the whole path")
        else:
            t, det = res
            print(f"FIRST non-Hausdorff point at t = {t}")
            print(det.describe(cx))
        return 0

    can = compute_canopy(cx, plf, policy=args.policy, deep=not args.no_deep)
    print()
    print(can.summary())
    print("\nstructural points "
          f"({args.mode}-mode): {can.structural_points(mode=args.mode)}")
    print("\ntrajectories:")
    for tr in can.trajectories:
        if cx.dim[tr.starts[0][1][0]] == args.dim:
            print("  " + tr.describe(cx))

    if args.plots:
        from . import viz
        print()
        print(viz.plot_vineyard_3d(can, dim=args.dim,
                                   fname=f"{args.plots}_vineyard3d.png"))
        print(viz.plot_persistence(can, dim=args.dim,
                                   fname=f"{args.plots}_persistence.png"))
        if args.positions:
            pos = load_positions(args.positions)
            ts = sorted({Fraction(0), Fraction(1)}
                        | {e.t for e in can.structural_events()})
            print(viz.plot_snapshots(can, pos, times=ts, dim=args.dim,
                                     fname=f"{args.plots}_snapshots.png"))
            if args.level is not None:
                lvl = Fraction(args.level)
                print(viz.plot_sublevel_snapshots(
                    can, pos, times=ts, level=lvl, dim=args.dim,
                    fname=f"{args.plots}_sublevel.png"))
                if args.gif:
                    print(viz.animate_sublevel(can, pos,
                                               f"{args.plots}_sublevel.gif",
                                               level=lvl, dim=args.dim))
            elif args.gif:
                print(viz.animate(can, pos, f"{args.plots}.gif", dim=args.dim))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
