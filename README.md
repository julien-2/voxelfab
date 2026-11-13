# voxelfab

**Virtual fabrication engine** — process emulation on multi-material voxel
grids, in the spirit of SEMulator3D: geometry-driven (morphological) process
steps, layouts in, 3D structures and virtual metrology out.

## What it does

- **Voxel process kernel** (NumPy/SciPy, Numba-accelerated): conformal and
  directional deposition, isotropic and directional etch, lithography, CMP,
  trench fill — all vectorized, calibrated for 1–100M-voxel grids.
- **Process decks** (YAML): the flow is a versioned file (`steps:`,
  `materials:`, `outputs:`), with provenance + structure hashing for
  deterministic runs.
- **Layout import**: GDSII/OASIS masks (gdstk + shapely), with mask bias and
  voxel-center sampling.
- **Virtual metrology**: thickness, CD, top material, volume fractions —
  written to `report.json` with a full provenance block.
- **3D demo**: planar MOSFET flow (STI → gate stack → spacers → epitaxy →
  contacts), cross-sections, VTK/STL export.

## Install

    uv pip install .                 # core (numpy, scipy, matplotlib, pyyaml)
    uv pip install ".[layout]"       # + GDS/OASIS import (gdstk, shapely)
    uv pip install ".[view]"         # + VTK export (pyvista)
    uv pip install ".[bench]"        # + Numba acceleration (recommended)
    uv pip install -e ".[dev]"       # development (pytest)

## Quickstart

    vf demo                          # write + run the demo MOSFET deck
    vf run decks/demo_deck.yaml -o out/   # run any deck
    vf info decks/demo_deck.yaml     # validate + summarize a deck
    vf bench 256                     # micro-benchmark (16.7M voxels)

Outputs land in the output directory: `mos3d.png` (top view + cross-sections),
`report.json` (metrology + provenance + `structure_sha256`).

## Package layout

    src/voxelfab/
    ├── grid.py        VoxelGrid, materials registry, height map
    ├── masks.py       geometric lithography masks
    ├── ops/           step registry + REFERENCE SEMANTICS (base.py), cpu.py
    ├── deck.py        YAML deck loading, demo deck
    ├── run.py         orchestrator: flow execution, outputs, provenance
    ├── metro.py       virtual metrology
    ├── export.py      cross-sections, 3D views, VTK, STL
    ├── layout.py      GDS/OASIS import (extra [layout])
    └── cli.py         vf run | info | demo | bench

    decks/demo_deck.yaml   shipped example deck (planar MOSFET)
    tests/test_golden.py   golden structure test (byte-exact determinism)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full product
architecture, the deck DSL reference, and the roadmap.

## Roadmap

v0.1 package split + deck DSL (this) → v0.2 Taichi GPU backend (voxel-exact
vs CPU) → v0.3 viewer + metrology DSL → v0.4 DOE sweeps → v1.0 GUI client.

## License

[MIT](LICENSE) — © 2026 Julien Del.

## Status

Experimental.
