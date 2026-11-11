# voxelfab

**Virtual fabrication engine** — process emulation on multi-material voxel
grids, in the spirit of SEMulator3D: geometry-driven (morphological) process
steps, layouts in, 3D structures and virtual metrology out.

## What it does

- **Voxel process kernel** (NumPy/SciPy, Numba-accelerated): conformal and
  directional deposition, isotropic and directional etch, lithography, CMP,
  trench fill — all vectorized, calibrated for 1–100M-voxel grids.
- **Process decks** (`vfdeck.py`): the flow is a versioned YAML file
  (`steps:`, `materials:`, `outputs:`), with provenance + structure
  hashing for deterministic runs.
- **Layout import**: GDSII/OASIS masks (gdstk + shapely), with mask bias and
  voxel-center sampling.
- **3D demos**: planar MOSFET flow (STI -> gate stack -> spacers -> epitaxy ->
  contacts), cross-sections, VTK/STL export, virtual metrology reports.

## Quickstart

    pip install numpy scipy matplotlib pyyaml numba
    python voxelfab.py            # demo: simplified CMOS flow (2D cross-section)
    python voxelfab.py --mos3d    # demo: 3D MOSFET flow + views
    python voxelfab.py --bench 256  # performance micro-benchmark
    python vfdeck.py demo         # write + run the YAML process deck

Optional: `pip install gdstk shapely` (layout import), `pyvista` (VTK),
`scikit-image numpy-stl` (STL export).

## Layout (v0.1 — package split planned, see docs/ARCHITECTURE.md)

    voxelfab.py   voxel process kernel (grid, steps, GDS/OAS import, exports)
    vfdeck.py     YAML process-deck runner (deck DSL v1, CLI, reports)
    docs/ARCHITECTURE.md   product architecture & roadmap

## Roadmap

v0.1 deck runner (this) -> v0.2 Taichi GPU backend (voxel-exact vs CPU) ->
v0.3 viewer + metrology DSL -> v0.4 DOE sweeps -> v1.0 GUI client.

## Status

Experimental. License TBD.
