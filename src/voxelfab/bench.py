"""Micro-benchmark : chronomètre chaque type d'étape sur une grille n³.

    vf bench [n] [--backend cpu|gpu|all]
        n = 256 -> 16,7 M voxels ; n = 430 -> ~80 M voxels
        --backend all : lance chaque backend disponible et imprime le
        comparatif (speedup) — typiquement cpu vs gpu (Taichi).
"""


from __future__ import annotations

import time

import numpy as np

from .grid import VoxelGrid
from .ops import available_backends, get_step
from .ops.cpu import HAS_NUMBA


def bench(n: int = 256, backend: str = "cpu"):
    backends = available_backends() if backend == "all" else [backend]
    results: dict[str, dict] = {}
    for be in backends:
        results[be] = _bench_one(n, be)
    if len(results) > 1:
        ref, last = backends[0], backends[-1]
        print(f"\n=== comparatif {ref} vs {last} (grille {n}³) ===")
        for label, t_ref in results[ref].items():
            t_last = results[last][label]
            print(f"  {label:<28} {t_ref * 1e3:9.0f} ms {t_last * 1e3:9.0f} ms"
                  f"   x{t_ref / t_last:5.1f}")


def _bench_one(n: int, backend: str) -> dict:
    g = VoxelGrid(n, n, n, pitch=1.0)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    SiN = g.add_material("SiN", "#8bc34a")
    substrate = get_step(backend, "substrate")
    dep_conf = get_step(backend, "deposit", "conformal")
    dep_dir = get_step(backend, "deposit", "directional")
    etch_iso = get_step(backend, "etch", "isotropic")
    etch_dir = get_step(backend, "etch", "directional")
    cmp_step = get_step(backend, "cmp")
    substrate(g, SiO2, n // 2)
    print(f"\n=== bench {g.nx}x{g.ny}x{g.nz} = "
          f"{g.nx * g.ny * g.nz / 1e6:.1f} M voxels "
          f"(backend: {backend}, numba: {HAS_NUMBA}) ===")

    steps = [
        ("deposit conformal t=5", lambda gg: dep_conf(gg, SiN, 5)),
        ("deposit conformal t=20", lambda gg: dep_conf(gg, SiN, 20)),
        ("deposit directional t=10", lambda gg: dep_dir(gg, SiN, 10)),
        ("etch isotropic t=5", lambda gg: etch_iso(gg, 5)),
        ("etch isotropic t=20", lambda gg: etch_iso(gg, 20)),
        ("etch directional t=30", lambda gg: etch_dir(gg, 30, mask=mask)),
        ("cmp", lambda gg: cmp_step(gg, n - 8, stop_on=SiO2)),
        ("height_map", lambda gg: gg.height_map()),
    ]
    mask = np.zeros((g.nx, g.ny), bool)
    mask[::4, :] = True

    if backend != "cpu":
        # warm-up JIT Taichi : une passe non chronométrée sur une grille
        # fictive de même shape (les noyaux sont compilés par shape)
        scratch = VoxelGrid(n, n, n, pitch=1.0)
        s2 = scratch.add_material("SiO2", "#c9c9c9")
        scratch.add_material("SiN", "#8bc34a")
        substrate(scratch, s2, n // 2)
        for _, fn in steps:
            fn(scratch)

    results: dict[str, float] = {}

    def timed(label, fn):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        results[label] = dt
        print(f"  {label:<28} {dt * 1e3:8.0f} ms")

    for label, fn in steps:
        timed(label, lambda gg=g, f=fn: f(gg))
    return results
