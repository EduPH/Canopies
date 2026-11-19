# -*- coding: utf-8 -*-
"""
canopies -- augmented canopies for 1-parameter fibered filtrations.

Detection of non-Hausdorff (structural) points and computation of
trajectories, after Giunti & Munch, *Canopies: A Generalization of Vines and
Vineyards for Parameterized Persistence*, with the fast updates of
Cohen-Steiner, Edelsbrunner & Morozov, *Vines and Vineyards by Updating
Persistence in Linear Time*.

Layout
------
``canopies.core``            exact core: complexes, PL filtrations, RU
                             decomposition, pairings, detection, canopies
``canopies.io``              dataset loaders
``canopies.viz``             matplotlib plots and animations  (extra: ``viz``)
``canopies.html_viz``        self-contained interactive HTML views (no deps)
``canopies.cli``             the ``canopies`` console script

Quickstart
----------
>>> from canopies import Complex, PLFiltration, compute_canopy
>>> cx = Complex([(1,), (2,), (1, 2)])
>>> plf = PLFiltration(cx, {(1,): 0, (2,): [(0, 1), (1, -1)], (1, 2): 2})
>>> can = compute_canopy(cx, plf)

All core arithmetic is exact (``fractions.Fraction``): event times and ties
are computed exactly, never by float comparison.
"""
from __future__ import annotations

__version__ = "0.1.1"
__author__ = "Eduardo Paluzo-Hidalgo"
__license__ = "MIT"

from .core import (
    # constants
    INF,
    STAR,
    # data structures
    Canopy,
    Complex,
    Detection,
    PLFiltration,
    RUDecomposition,
    SwitchEvent,
    Trajectory,
    # pairings and orders
    check_filtration,
    diagram_of_pairing,
    eta_of_pair,
    format_pairing,
    pairing_for_order,
    total_consistent_orders,
    value_blocks,
    # analysis entry points
    compute_canopy,
    detect_structural,
    find_first_structural,
)
from .io import from_vertex_functions, load_dataset, load_positions

__all__ = [
    "__version__",
    # constants
    "INF",
    "STAR",
    # data structures
    "Canopy",
    "Complex",
    "Detection",
    "PLFiltration",
    "RUDecomposition",
    "SwitchEvent",
    "Trajectory",
    # pairings and orders
    "check_filtration",
    "diagram_of_pairing",
    "eta_of_pair",
    "format_pairing",
    "pairing_for_order",
    "total_consistent_orders",
    "value_blocks",
    # analysis entry points
    "compute_canopy",
    "detect_structural",
    "find_first_structural",
    # io
    "from_vertex_functions",
    "load_dataset",
    "load_positions",
    # bundled data
    "example_dataset_path",
    "example_positions_path",
    "load_example",
]


def _data(name: str):
    """Filesystem path of a file bundled in ``canopies/data``."""
    try:                                   # Python >= 3.9
        from importlib.resources import files
        return files(__package__).joinpath("data", name)
    except ImportError:                    # pragma: no cover - very old Pythons
        import os
        return os.path.join(os.path.dirname(__file__), "data", name)


def example_dataset_path():
    """Path to the bundled ``example_dataset.txt`` (the three-cycles run)."""
    return _data("example_dataset.txt")


def example_positions_path():
    """Path to the bundled ``example_positions.txt`` (2-D vertex embedding)."""
    return _data("example_positions.txt")


def load_example():
    """``(Complex, PLFiltration)`` of the bundled example dataset.

    >>> from canopies import load_example, compute_canopy
    >>> cx, plf = load_example()
    >>> can = compute_canopy(cx, plf)
    """
    return load_dataset(str(example_dataset_path()))
