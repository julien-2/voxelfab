"""Micro-benchmark : chronomètre chaque type d'étape sur une grille n³.

    vf bench [n]        # n = 256 -> 16,7 M voxels ; n = 430 -> ~80 M voxels
"""

from __future__ import annotations

import time

import numpy as np

from .grid import VoxelGrid
from .ops import get_step
from .ops.cpu import HAS_NUMBA


def bench(n: int = 256):
    g = VoxelGrid(n, n, n, pitch=1.0)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    SiN = g.add_material("SiN", "#8bc34a")
    substrate = get_step("cpu", "substrate")
    dep_conf = get_step("cpu", "deposit", "conformal")
    dep_dir = get_step("cpu", "deposit", "directional")
    etch_iso = get_step("cpu", "etch", "isotropic")
    etch_dir = get_step("cpu", "etch", "directional")
    cmp_step = get_step("cpu", "cmp")
    substrate(g, SiO2, n // 2)
    print(f"\n=== bench {g.nx}x{g.ny}x{g.nz} = "
          f"{g.nx * g.ny * g.nz / 1e6:.1f} M voxels (numba: {HAS_NUMBA}) ===")

    def timed(label, fn):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        print(f"  {label:<28} {dt * 1e3:8.0f} ms")

    timed("deposit conformal t=5", lambda: dep_conf(g, SiN, 5))
    timed("deposit conformal t=20", lambda: dep_conf(g, SiN, 20))
    timed("deposit directional t=10", lambda: dep_dir(g, SiN, 10))
    timed("etch isotropic t=5", lambda: etch_iso(g, 5))
    timed("etch isotropic t=20", lambda: etch_iso(g, 20))
    mask = np.zeros((g.nx, g.ny), bool)
    mask[::4, :] = True
    timed("etch directional t=30", lambda: etch_dir(g, 30, mask=mask))
    timed("cmp", lambda: cmp_step(g, n - 8, stop_on=SiO2))
    timed("height_map", g.height_map)
