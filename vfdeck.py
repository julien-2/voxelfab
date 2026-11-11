"""
vfdeck.py — Runner de process decks YAML pour le moteur voxel.

C'est la graine du produit 'kernel-first' : le deck est le fichier que
l'utilisateur édite, le runner l'exécute sur le moteur voxelfab et produit
la structure + les rapports. Ce fichier devient deck.py + run.py + cli.py
au découpage en package (cf. architecture v1).

Dépendances : numpy, scipy, matplotlib, pyyaml (+ voxelfab.py du canvas
précédent, à jour) ; optionnelles : gdstk, shapely (layout GDS/OAS),
pyvista (VTK), numba.

Usage :
    python vfdeck.py demo                       # écrit + exécute demo_deck.yaml
    python vfdeck.py run deck.yaml [-o outdir]  # exécute un deck
    python vfdeck.py info deck.yaml             # valide + résume le deck

Format de deck (v1) :
    version: 1
    grid:      {pitch_nm: 2.0, nx: 120, ny: 60, nz: 64}
    materials: {<nom>: {color: '#rrggbb'}, ...}
    layout:    (optionnel) {file, cell, bias_nm, layers: {<nom>: [L, D]}}
    steps:     liste ordonnée d'opérations (voir _OPS)
    outputs:   liste de {type: views|vtk|stl|report, file: ...}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from voxelfab import (VoxelGrid, Process, Metrology, mask_rectangle,
                      masks_from_layout, layout_info, grid_size_for,
                      render_views, export_vtk, export_stl)


class DeckError(Exception):
    """Erreur de deck, avec le numéro de l'étape fautive."""


# --------------------------------------------------------------------------
# Validation + exécution des étapes
# --------------------------------------------------------------------------

class DeckRunner:
    def __init__(self, deck: dict, deck_path: Path | None = None,
                 outdir: Path = Path(".")):
        self.deck = deck
        self.deck_path = deck_path
        self.outdir = outdir
        self.timings: list[tuple] = []
        self.mat: dict = {}          # nom -> id
        self.layer_masks: dict = {}  # nom logique -> mask (nx, ny)

    # --- helpers -----------------------------------------------------------

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
            m = np.zeros((g.nx, g.ny), dtype=bool)
            for a, b in spec["bands"]:
                m[int(a * g.nx):int(b * g.nx), :] = True
            return m
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
        self.p = Process(self.g)

    # --- exécution -----------------------------------------------------------

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
        if op == "substrate":
            self.p.substrate(self._mid(step["material"]),
                             thickness_nm=self._thickness(step, "thickness_nm"))
        elif op == "deposit":
            on = step.get("on_materials", step.get("on"))
            if isinstance(on, bool):
                raise DeckError("clé 'on:' interprétée comme booléen YAML — "
                                "utiliser 'on_materials:'")
            self.p.deposit(self._mid(step["material"]),
                           thickness_nm=self._thickness(step, "thickness_nm"),
                           mode=step.get("mode", "conformal"),
                           on_materials=self._mids(on) if on else None)
        elif op == "etch":
            self.p.etch(thickness_nm=self._thickness(step, "depth_nm",
                                                     "thickness_nm"),
                        mode=step.get("mode", "directional"),
                        materials=self._mids(step.get("materials")) or None,
                        until_material=self._mid(step["until"]) if "until" in step else None,
                        mask=self._mask(step.get("mask"), where))
        elif op == "litho":
            self.p.litho(self._mask(step["mask"], where),
                         thickness_nm=self._thickness(step, "depth_nm",
                                                     "thickness_nm"),
                         materials=self._mids(step.get("materials")) or None,
                         until_material=self._mid(step["until"]) if "until" in step else None)
        elif op == "fill":
            self.p.fill_trench(self._mid(step["material"]),
                               max_z=int(step["to_plane_z"]))
        elif op == "cmp":
            self.p.cmp(target_z_voxels=int(step["target_z"]),
                       stop_on=self._mid(step["stop_on"]) if "stop_on" in step else None)
        else:
            raise DeckError(f"op inconnue : {op!r}")

    # --- sorties ---------------------------------------------------------------

    def outputs(self):
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
            "voxelfab_version": "0.1",
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DEMO_DECK = """\
version: 1
grid:      {pitch_nm: 2.0, nx: 120, ny: 60, nz: 64}
materials:
  Si:   {color: '#4d79ff'}
  SiO2: {color: '#c9c9c9'}
  SiN:  {color: '#8bc34a'}
  poly: {color: '#e64a19'}
  W:    {color: '#5d4037'}
steps:
  - {op: substrate, material: Si, thickness_nm: 40}
  - {op: etch, depth_nm: 12, mask: {bands: [[0.0, 0.18], [0.82, 1.0]]}}
  - {op: fill, material: SiO2, to_plane_z: 19}
  - {op: cmp, target_z: 19, stop_on: Si}
  - {op: deposit, material: SiO2, thickness_nm: 4}
  - {op: deposit, material: poly, thickness_nm: 36}
  - {op: etch, depth_nm: 60, materials: [poly],
     mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
  - {op: etch, depth_nm: 4, mode: isotropic, materials: [SiO2],
     mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
  - {op: deposit, material: SiN, thickness_nm: 8}
  - {op: etch, depth_nm: 8, mode: directional, materials: [SiN]}
  - {op: deposit, material: Si, thickness_nm: 12, mode: directional, on_materials: [Si]}
  - {op: deposit, material: SiO2, thickness_nm: 50}
  - {op: cmp, target_z: 52}
  - {op: etch, depth_nm: 60, until: Si,
     mask: {rect: [0.18, 0.30, 0.30, 0.70]}}
  - {op: etch, depth_nm: 60, until: Si,
     mask: {rect: [0.70, 0.82, 0.30, 0.70]}}
  - {op: deposit, material: W, thickness_nm: 60, mode: directional}
  - {op: cmp, target_z: 52}
outputs:
  - {type: views, file: mos3d.png}
  - {type: report, file: report.json}
"""


def cmd_demo(args):
    path = Path("demo_deck.yaml")
    path.write_text(DEMO_DECK)
    print(f"[demo] deck écrit : {path}")
    run_deck(path, Path(args.outdir or "."))


def run_deck(path: Path, outdir: Path):
    deck = yaml.safe_load(path.read_text())
    if not isinstance(deck, dict):
        raise DeckError("deck vide ou mal formé")
    DeckRunner(deck, deck_path=path, outdir=outdir).run()


def cmd_run(args):
    run_deck(Path(args.deck), Path(args.outdir or Path(args.deck).parent))


def cmd_info(args):
    deck = yaml.safe_load(Path(args.deck).read_text())
    steps = deck.get("steps", [])
    print(f"deck     : {args.deck}")
    print(f"version  : {deck.get('version')}")
    print(f"grid     : {deck.get('grid')}")
    print(f"matériaux: {', '.join(deck.get('materials', {}))}")
    if "layout" in deck:
        print(f"layout   : {deck['layout'].get('file')} "
              f"couches={list(deck['layout'].get('layers', {}))}")
    print(f"étapes   : {len(steps)}")
    for i, s in enumerate(steps, 1):
        print(f"  {i:2d}. {s.get('op'):10s} "
              f"{ {k: v for k, v in s.items() if k != 'op'} }")
    print(f"sorties  : {[o.get('type') for o in deck.get('outputs', [])]}")


def main():
    ap = argparse.ArgumentParser(prog="vfdeck",
                                 description="runner de process decks voxel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="exécuter un deck")
    p_run.add_argument("deck")
    p_run.add_argument("-o", "--outdir")
    p_run.set_defaults(fn=cmd_run)
    p_info = sub.add_parser("info", help="valider et résumer un deck")
    p_info.add_argument("deck")
    p_info.set_defaults(fn=cmd_info)
    p_demo = sub.add_parser("demo", help="écrire + exécuter le deck de démo")
    p_demo.add_argument("-o", "--outdir")
    p_demo.set_defaults(fn=cmd_demo)
    args = ap.parse_args()
    try:
        args.fn(args)
    except DeckError as e:
        print(f"ERREUR de deck : {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
