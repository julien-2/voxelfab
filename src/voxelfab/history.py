"""Historique du procédé dans UN seul fichier HDF5 (issue #9).

Un run avec `history: {file: history.h5, mode: delta}` écrit, après chaque
étape, l'état incrémental dans un fichier unique :

    /attrs            mode, pitch_nm, grid [nx,ny,nz], deck_sha256, backend…
    /materials        table (id, name, color), comme structure.h5
    /steps/step_0000/ état INITIAL (delta depuis une grille de zéros)
      attrs : index, op, params (JSON), n_changed, step_sha256 (hash chaîné)
      datasets (mode delta) : coords (n,3) uint32 + mats (n,) uint8, gzip
                 (mode full)  : grid (nx,ny,nz) uint8, gzip
                 (mode summary): aucun dataset, attrs seulement
    /steps/step_0001/ idem après l'étape 1, etc.

Hash chaîné : step_sha256 = sha256(prev_sha256 || op || params || payload),
ce qui rend toute modification d'une étape détectable (intégrité).

Reconstruction : `grid_at(path, k)` rejoue les deltas 0..k -> VoxelGrid
byte-identique à l'état après l'étape k.

Modes :
  - `summary` : métadonnées + hash par étape, quasi gratuit ;
  - `delta`   : coordonnées des voxels modifiés uniquement (défaut) — la
                taille dépend de la surface modifiée, pas de la grille ;
  - `full`    : grille complète par étape (gzip) — petites grilles.

h5py est importé paresseusement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .grid import Materials, VoxelGrid

MODES = ("summary", "delta", "full")


def _rle_x(changed, data):
    """Encode le masque `changed` (nx, ny, nz) en runs horizontaux (axe x) :
    retourne (runs (m, 4) [z, y, x0, x1_exclusif], mats (m,)). Une couche
    pleine coute 1 run par ligne (ny runs), pas nx*ny voxels — c'est ce qui
    rend le mode delta compact sur les procédés à couches uniformes."""
    nx, ny, nz = changed.shape
    cdt = np.uint16 if max(nx, ny, nz) <= 65535 else np.uint32
    all_runs, all_mats = [], []
    for z in range(nz):
        plan = changed[:, :, z].T                       # (ny, nx)
        if not plan.any():
            continue
        d = np.diff(plan.astype(np.int8), axis=1, prepend=0, append=0)
        s = np.argwhere(d == 1)                         # débuts de runs
        e = np.argwhere(d == -1)                        # fins (exclusives)
        if len(s) == 0:
            continue
        y0, x0, x1 = s[:, 0], s[:, 1], e[:, 1]
        all_runs.append(np.stack([np.full(len(s), z, dtype=cdt),
                                  y0.astype(cdt), x0.astype(cdt),
                                  x1.astype(cdt)], axis=1))
        all_mats.append(data[x0, y0, z].astype(np.uint8))
    if not all_runs:
        return np.zeros((0, 4), cdt), np.zeros(0, np.uint8)
    return np.concatenate(all_runs), np.concatenate(all_mats)


class HistoryWriter:
    """Écrit l'historique d'un run étape par étape dans un fichier HDF5.

    Usage (run.py) :
        with HistoryWriter(path, grid, provenance, mode="delta") as hw:
            hw.record(0, "init", {}, grid)            # état initial
            ... hw.record(i, op, params, grid) ...    # après chaque étape
    """

    def __init__(self, path, grid: VoxelGrid, provenance: dict | None = None,
                 mode: str = "delta", every: int = 1):
        import h5py
        if mode not in MODES:
            raise ValueError(f"mode d'historique {mode!r} invalide "
                             f"(summary | delta | full)")
        self.grid = grid
        self.mode = mode
        self.every = max(1, int(every))
        self._prev = np.zeros_like(grid.data)
        self._hash = "0" * 64
        self._count = 0
        self._f = h5py.File(str(path), "w")
        self._f.attrs["voxelfab_history_version"] = 1
        self._f.attrs["mode"] = mode
        self._f.attrs["pitch_nm"] = float(grid.pitch)
        self._f.attrs["grid"] = [grid.nx, grid.ny, grid.nz]
        for k in ("deck_sha256", "voxelfab_version", "backend"):
            if provenance and k in provenance:
                self._f.attrs[k] = provenance[k]
        str_dt = h5py.string_dtype("utf-8")
        mat_dt = np.dtype([("id", "u1"), ("name", str_dt), ("color", str_dt)])
        self._f.create_dataset("materials", data=np.array(
            [(mid, grid.materials.name(mid), grid.materials.color(mid))
             for mid in grid.materials.ids if mid != Materials.AIR],
            dtype=mat_dt))
        self._f.create_group("steps")

    def record(self, index: int, op: str, params: dict, grid: VoxelGrid) -> str:
        """Enregistre l'état courant (étape `index`) et retourne le hash
        chaîné de l'étape."""
        data = grid.data
        changed = data != self._prev
        n = int(changed.sum())
        if self.mode == "delta":
            runs, rmats = _rle_x(changed, data)
        h = hashlib.sha256()
        h.update(self._hash.encode())
        h.update(str(op).encode())
        h.update(json.dumps(params, sort_keys=True, default=str).encode())
        if self.mode == "delta":
            h.update(runs.tobytes())
            h.update(rmats.tobytes())
        elif self.mode == "full":
            h.update(data.tobytes())
        else:                                    # summary : hash de la grille
            h.update(hashlib.sha256(data.tobytes()).hexdigest().encode())
        step_sha = h.hexdigest()

        grp = self._f.create_group(f"steps/step_{index:04d}")
        grp.attrs["index"] = int(index)
        grp.attrs["op"] = str(op)
        grp.attrs["params"] = json.dumps(params, sort_keys=True, default=str)
        grp.attrs["n_changed"] = n
        grp.attrs["step_sha256"] = step_sha
        if self.mode == "delta":
            grp.create_dataset("runs", data=runs, compression="gzip",
                               compression_opts=4, shuffle=True)
            grp.create_dataset("runs_mats", data=rmats, compression="gzip",
                               compression_opts=4)
        elif self.mode == "full":
            grp.create_dataset("grid", data=data, compression="gzip",
                               compression_opts=4, shuffle=True)
        self._prev = data.copy()
        self._hash = step_sha
        self._count += 1
        return step_sha

    def close(self):
        if not self._f:
            return
        self._f.attrs["n_steps"] = self._count
        self._f.close()
        self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_history(path) -> dict:
    """Métadonnées + liste des étapes d'un fichier history.h5."""
    import h5py
    with h5py.File(str(path), "r") as f:
        meta = {k: f.attrs[k] for k in f.attrs.keys()}
        steps = []
        for name in sorted(f["steps"].keys()):
            g = f["steps"][name]
            steps.append({
                "index": int(g.attrs["index"]),
                "op": str(g.attrs["op"]),
                "params": json.loads(g.attrs["params"]),
                "n_changed": int(g.attrs["n_changed"]),
                "step_sha256": str(g.attrs["step_sha256"]),
            })
        meta["steps"] = steps
    return meta


def grid_at(path, k: int) -> VoxelGrid:
    """Reconstruit la grille à l'étape k (0 = état initial) : rejoue les
    deltas chaînés (mode delta) ou lit la grille (mode full). Byte-identique
    à l'état du run après l'étape k."""
    import h5py
    with h5py.File(str(path), "r") as f:
        nx, ny, nz = (int(v) for v in f.attrs["grid"])
        g = VoxelGrid(nx, ny, nz, pitch=float(f.attrs["pitch_nm"]))
        for row in f["materials"][()]:
            name = (row["name"].decode() if isinstance(row["name"], bytes)
                    else str(row["name"]))
            color = (row["color"].decode() if isinstance(row["color"], bytes)
                     else str(row["color"]))
            g.add_material(name, color)
        mode = str(f.attrs["mode"])
        for name in sorted(f["steps"].keys()):
            grp = f["steps"][name]
            if int(grp.attrs["index"]) > k:
                break
            if mode == "full":
                g.data = np.asarray(grp["grid"][()], dtype=np.uint8)
            elif mode == "delta":
                runs = grp["runs"][()]
                rmats = grp["runs_mats"][()]
                for (z, y, x0, x1), m in zip(runs, rmats):
                    g.data[x0:x1, y, z] = m
    return g
