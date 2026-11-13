"""Orchestrateur : exécution d'un deck de procédé sur le moteur voxel.

DeckRunner traduit les étapes du deck (nm, noms de matériaux) en appels
d'ops voxel via le registre ops.get_step(backend, ...), gère les masques,
les sorties et le rapport de provenance. Le backend CPU est la référence ;
le backend GPU (v0.2) devra produire des structures identiques octet pour
octet (tests d'équivalence).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ._version import __version__
from .deck import DeckError
from .grid import VoxelGrid
from .layout import grid_size_for, layout_info, masks_from_layout
from .masks import mask_bands, mask_rectangle
from .metro import Metrology
from .ops import get_step


class DeckRunner:
    """Exécute un deck (dict issu du YAML) et produit structure + rapport."""

    def __init__(self, deck: dict, deck_path: Path | None = None,
                 outdir: Path = Path("."), backend: str = "cpu"):
        self.deck = deck
        self.deck_path = deck_path
        self.outdir = outdir
        self.backend = backend
        self.timings: list[tuple] = []
        self.log: list[str] = []
        self.mat: dict = {}          # nom -> id
        self.layer_masks: dict = {}  # nom logique -> mask (nx, ny)

    # --- helpers -----------------------------------------------------------

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"  [flow] {msg}")

    def _mid(self, name: str) -> int:
        if name not in self.mat:
            raise DeckError(f"matériau inconnu : {name!r} "
                            f"(déclaré : {sorted(self.mat)})")
        return self.mat[name]

    def _mids(self, names) -> tuple:
        return tuple(self._mid(n) for n in (names or ()))

    def _thickness(self, step: dict, *keys: str):
        """Première clé présente ; le nm est l'unité obligatoire."""
        for k in keys:
            if k in step:
                return float(step[k])
        raise DeckError(f"épaisseur manquante ({'/'.join(keys)}) : {step}")

    def _mask(self, spec, where=""):
        """Construit un masque (nx, ny) : layer | rect | bands | invert."""
        g = self.g
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise DeckError(f"masque invalide {where} : {spec!r}")
        if "layer" in spec:
            name = spec["layer"]
            if name not in self.layer_masks:
                raise DeckError(f"couche inconnue {where} : {name!r} "
                                f"(layout: {sorted(self.layer_masks)})")
            return self.layer_masks[name]
        if "rect" in spec:
            x0, x1, y0, y1 = spec["rect"]
            return mask_rectangle(g.nx, g.ny, x0, x1, y0, y1)
        if "bands" in spec:
            return mask_bands(g.nx, g.ny, spec["bands"])
        if "invert" in spec:
            return ~self._mask(spec["invert"], where)
        raise DeckError(f"masque non compris {where} : {spec!r}")

    # --- setup -------------------------------------------------------------

    def setup(self):
        d = self.deck
        if d.get("version") != 1:
            raise DeckError(f"version de deck non supportée : "
                            f"{d.get('version')!r} (attendu : 1)")

        gr = d.get("grid", {})
        pitch = float(gr.get("pitch_nm", 1.0))
        nx, ny, nz = int(gr.get("nx", 0)), int(gr.get("ny", 0)), \
            int(gr.get("nz", 0))

        # layout optionnel : la grille est déduite de la bbox
        lay = d.get("layout")
        if lay:
            bbox, layers_found = layout_info(lay["file"], lay.get("cell"))
            print(f"[layout] {lay['file']} bbox={bbox} µm couches={layers_found}")
            nx, ny = grid_size_for(bbox, pitch)
            self.g = VoxelGrid(nx, ny, nz, pitch=pitch)
            wanted = tuple((int(l), int(dt))
                           for l, dt in lay.get("layers", {}).values())
            masks = masks_from_layout(lay["file"], self.g, layers=wanted,
                                      bbox=bbox, cell_name=lay.get("cell"),
                                      bias_nm=float(lay.get("bias_nm", 0.0)))
            names = lay.get("layers", {})
            self.layer_masks = {name: masks[(int(l), int(dt))]
                                for name, (l, dt) in names.items()}
        else:
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise DeckError("grid: nx/ny/nz requis (ou un layout)")
            self.g = VoxelGrid(nx, ny, nz, pitch=pitch)

        # matériaux
        for name, m in d.get("materials", {}).items():
            self.mat[name] = self.g.add_material(name, m.get("color", "#888888"))

    # --- exécution ---------------------------------------------------------

    def run(self):
        self.setup()
        d = self.deck
        for i, step in enumerate(d.get("steps", []), start=1):
            # garde : YAML 1.1 convertit les clés non cotées 'on/yes/off/no'
            # en booléens -> erreur claire plutôt qu'un crash JSON ultérieur
            for k in step:
                if isinstance(k, bool):
                    raise DeckError(
                        f"étape {i} : clé YAML booléenne (on/yes/off/no…) — "
                        f"écrire 'on_materials:' ou coter la clé ('on': …)")
            t0 = time.perf_counter()
            try:
                self._exec(step)
            except DeckError as e:
                raise DeckError(f"étape {i} : {e}") from None
            except KeyError as e:
                raise DeckError(f"étape {i} : paramètre manquant {e}") from None
            self.timings.append((i, step.get("op", "?"),
                                 time.perf_counter() - t0))
        self.outputs()
        self.report_deck()

    def _exec(self, step: dict):
        op = step.get("op")
        where = f"(op {op})"
        g = self.g

        if op == "substrate":
            t = g.voxels(self._thickness(step, "thickness_nm"))
            get_step(self.backend, "substrate")(g, self._mid(step["material"]), t)
            self._log(f"substrate {g.materials.name(self.mat[step['material']])} ({t} vx)")

        elif op == "deposit":
            on = step.get("on_materials", step.get("on"))
            if isinstance(on, bool):
                raise DeckError("clé 'on:' interprétée comme booléen YAML — "
                                "utiliser 'on_materials:'")
            mode = step.get("mode", "conformal")
            t = g.voxels(self._thickness(step, "thickness_nm"))
            get_step(self.backend, "deposit", mode)(
                g, self._mid(step["material"]), t,
                on_materials=self._mids(on) if on else None)
            self._log(f"deposit {mode} {step['material']} ({t} vx)")

        elif op == "etch":
            mode = step.get("mode", "directional")
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            materials = self._mids(step.get("materials")) or None
            until = self._mid(step["until"]) if "until" in step else None
            get_step(self.backend, "etch", mode)(
                g, t, materials=materials, until_material=until,
                mask=self._mask(step.get("mask"), where))
            self._log(f"etch {mode} {t} vx"
                      + (f" sur {tuple(step['materials'])}" if step.get("materials") else ""))

        elif op == "litho":
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            materials = self._mids(step.get("materials")) or None
            until = self._mid(step["until"]) if "until" in step else None
            mask = self._mask(step["mask"], where)
            get_step(self.backend, "etch", "directional")(
                g, t, materials=materials, until_material=until, mask=mask)
            self._log(f"litho (zone ouverte = {mask.mean() * 100:.0f} % de la surface)")

        elif op == "fill":
            get_step(self.backend, "fill")(g, self._mid(step["material"]),
                                           max_z=int(step["to_plane_z"]))
            self._log(f"fill {step['material']} (tranchées sous z={int(step['to_plane_z'])})")

        elif op == "cmp":
            target = int(step["target_z"])
            stop = self._mid(step["stop_on"]) if "stop_on" in step else None
            removed = get_step(self.backend, "cmp")(g, target, stop_on=stop)
            self._log(f"CMP -> plan z={target} ({removed} vx retirés)")

        else:
            raise DeckError(f"op inconnue : {op!r}")

    # --- sorties -----------------------------------------------------------

    def outputs(self):
        from .export import export_stl, export_vtk, render_views
        self.outdir.mkdir(parents=True, exist_ok=True)
        for out in self.deck.get("outputs", []):
            kind, file = out.get("type"), out.get("file", "out")
            path = str(self.outdir / file)
            if kind == "views":
                render_views(self.g, filename=path,
                             title=f"voxelfab — {self.deck_path or 'deck'}")
            elif kind == "vtk":
                export_vtk(self.g, filename=path)
            elif kind == "stl":
                export_stl(self.g, prefix=path[:-4] if path.endswith(".stl")
                           else path)
            elif kind == "report":
                self.report_path = path
            else:
                raise DeckError(f"output inconnu : {kind!r}")

    def report_deck(self):
        """Rapport JSON : métrologie virtuelle + provenance."""
        met = Metrology(self.g)
        prov = {
            "deck": str(self.deck_path) if self.deck_path else "<inline>",
            "deck_sha256": (hashlib.sha256(
                json.dumps(self.deck, default=repr).encode()).hexdigest()),
            "grid": [self.g.nx, self.g.ny, self.g.nz],
            "pitch_nm": self.g.pitch,
            "voxelfab_version": __version__,
            "backend": self.backend,
            "steps": [{"n": n, "op": op, "s": round(dt, 3)}
                      for n, op, dt in self.timings],
        }
        rep = {"provenance": prov,
               "volume_fractions": met.report()["volume_fractions"],
               "structure_sha256": hashlib.sha256(
                   self.g.data.tobytes()).hexdigest()}
        path = getattr(self, "report_path", None) \
            or str(self.outdir / "report.json")
        Path(path).write_text(json.dumps(rep, indent=2))
        print(f"[report] {path}")
        print(f"[report] structure_sha256 = {rep['structure_sha256'][:16]}…")
        print(f"[report] volumes = {rep['volume_fractions']}")


def run_deck(path: Path, outdir: Path, backend: str = "cpu"):
    """Charge un deck YAML et l'exécute."""
    from .deck import load_deck
    deck = load_deck(path)
    DeckRunner(deck, deck_path=path, outdir=outdir, backend=backend).run()
