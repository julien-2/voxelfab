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
from .layout import grid_size_for, layout_info, layer_masks_from_layout
from .masks import mask_bands, mask_rectangle
from .metro import Metrology
from .ops import get_step, resolve_backend


class DeckRunner:
    """Exécute un deck (dict issu du YAML) et produit structure + rapport."""

    def __init__(self, deck: dict, deck_path: Path | None = None,
                 outdir: Path = Path("."), backend: str = "cpu"):
        self.deck = deck
        self.deck_path = deck_path
        self.outdir = outdir
        # 'auto' -> 'gpu' si taichi présent, sinon 'cpu' ; le backend effectif
        # est celui-ci (report.json / HDF5) pour la reproductibilité.
        self.backend = resolve_backend(backend)
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

    def _crystal_step_params(self, step: dict, default_rates: dict):
        """Validation partagée des étapes cristallographiques (epitaxy
        crystal, etch_crystal) : orientation + facet_rates."""
        ori = str(step.get("orientation", "100"))
        if ori not in ("100", "110", "111"):
            raise DeckError(f"orientation inconnue : {ori!r} "
                            "(100, 110 ou 111)")
        rates = step.get("facet_rates", default_rates)
        if not isinstance(rates, dict) or not rates:
            raise DeckError("'facet_rates' doit être un dict "
                            "non vide {famille: taux}")
        clean = {}
        for k, v in rates.items():
            ks = str(k)
            if ks not in ("100", "110", "111"):
                raise DeckError(f"famille de facettes inconnue : "
                                f"{k!r} (100, 110 ou 111)")
            if isinstance(v, bool) or \
                    not isinstance(v, (int, float)) or v <= 0:
                raise DeckError(f"taux invalide pour {ks} : {v!r}")
            clean[ks] = float(v)
        return ori, clean

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
        # nz_max optionnel : auto-grow de nz si une op déborde (issue #11)
        nz_max = int(gr["nz_max"]) if gr.get("nz_max") else None

        # layout optionnel : la grille est déduite de la bbox
        lay = d.get("layout")
        if lay:
            bbox, layers_found = layout_info(lay["file"], lay.get("cell"))
            print(f"[layout] {lay['file']} bbox={bbox} µm couches={layers_found}")
            nx, ny = grid_size_for(bbox, pitch)
            self.g = VoxelGrid(nx, ny, nz, pitch=pitch, nz_max=nz_max)
            # layers: {nom: [layer, datatype]} (v1) ou
            #         {nom: {ld: ..., polarity: dark|clear, bias_nm: ...}}
            # -> merge multi-datatypes, polarité clear/dark, biais par couche
            try:
                self.layer_masks = layer_masks_from_layout(
                    lay["file"], self.g, lay.get("layers", {}),
                    cell_name=lay.get("cell"), bbox=bbox,
                    bias_nm=float(lay.get("bias_nm", 0.0)))
            except ValueError as e:
                raise DeckError(f"layout: {e}") from None
        else:
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise DeckError("grid: nx/ny/nz requis (ou un layout)")
            self.g = VoxelGrid(nx, ny, nz, pitch=pitch, nz_max=nz_max)

        # matériaux
        for name, m in d.get("materials", {}).items():
            self.mat[name] = self.g.add_material(name, m.get("color", "#888888"))

        # substrat multicouche déclaratif (issue #11) : exécuté AVANT les
        # étapes, ordre bottom-first (premier élément = fond, z = 0).
        # Ex. SOI : [{material: Si, thickness_nm: 500},   # handle
        #           {material: SiO2, thickness_nm: 145},  # BOX
        #           {material: Si, thickness_nm: 40}]      # top Si
        sub = d.get("substrate")
        if sub:
            if not isinstance(sub, list) or not sub:
                raise DeckError("'substrate:' doit être une liste non vide "
                                "de {material, thickness_nm}")
            z = 0
            for i, layer in enumerate(sub, 1):
                mat = layer.get("material") if isinstance(layer, dict) else None
                if mat not in self.mat:
                    raise DeckError(f"substrate couche {i} : matériau {mat!r} "
                                    f"inconnu (materials: {sorted(self.mat)})")
                th = float(layer.get("thickness_nm", 0.0))
                t = self.g.voxels(th)
                if t <= 0:
                    raise DeckError(f"substrate couche {i} ({mat}) : "
                                    "thickness_nm > 0 requis")
                if not self.g.ensure_nz(z + t):
                    raise DeckError(
                        f"substrate : l'épaisseur totale ({(z + t) * self.g.pitch:.0f} nm) "
                        f"dépasse nz*pitch ({self.g.nz * self.g.pitch:.0f} nm, "
                        f"nz_max={self.g.nz_max}) — augmentez nz ou nz_max")
                self.g.data[:, :, z:z + t] = self.mat[mat]
                self._log(f"substrate {mat} ({t} vx, z {z}..{z + t - 1})")
                z += t

    # --- exécution ---------------------------------------------------------

    def run(self):
        self.setup()
        d = self.deck
        # Mode device-résident (backend gpu) : la grille reste sur le device
        # pendant toutes les étapes — un seul H2D en entrée, un seul D2H en
        # sortie (juste avant les exports), soit ~2×N octets de PCIe en moins
        # par étape. Voir ops/gpu.py (bind_resident/release_resident).
        resident = self.backend == "gpu"
        if resident and self.g.nz_max is not None:
            print("[grid] WARNING : nz_max (auto-grow) non supporté en backend "
                  "gpu — champs Taichi à shape fixe ; les dépôts au-delà de "
                  f"nz={self.g.nz} seront tronqués")
        # historique du procédé dans UN seul fichier (issue #9) : état initial
        # (avant bind_resident, grid.data est encore côté hôte)
        hw = None
        hcfg = d.get("history")
        if hcfg:
            from .history import HistoryWriter
            if not isinstance(hcfg, dict):
                raise DeckError(f"'history:' doit être un dict, pas {hcfg!r}")
            mode = str(hcfg.get("mode", "delta"))
            every = int(hcfg.get("every", 1) or 1)
            hfile = Path(self.outdir) / str(hcfg.get("file", "history.h5"))
            hfile.parent.mkdir(parents=True, exist_ok=True)
            if resident:
                print(f"[history] backend gpu résident : un D2H par étape "
                      f"enregistrée (every={every})")
            try:
                hw = HistoryWriter(hfile, self.g, self.provenance(),
                                   mode=mode, every=every)
            except ValueError as e:
                raise DeckError(f"history: {e}") from None
            hw.record(0, "init", {}, self.g)
            print(f"[history] {hfile} (mode={mode}, every={every})")
        if resident:
            from .ops.gpu import bind_resident, release_resident
            bind_resident(self.g)
        try:
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
                self._maybe_checkpoint(i, step)
                if hw is not None and i % hw.every == 0:
                    if resident:
                        from .ops.gpu import sync_to_host
                        sync_to_host(self.g)   # D2H temporaire pour l'historique
                    params = {k: v for k, v in step.items()
                              if k not in ("op", "checkpoint")}
                    sha = hw.record(i, step.get("op", "?"), params, self.g)
                    print(f"[history] étape {i} ({step.get('op')}) "
                          f"sha={sha[:12]}…")
        finally:
            if resident:
                release_resident(self.g)
            if hw is not None:
                hw.close()
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

        elif op == "etch_selective":
            mode = step.get("mode", "directional")
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            rates = step.get("rates")
            if not isinstance(rates, dict) or not rates:
                raise DeckError("'rates: {matériau: taux}' requis "
                                "(taux relatif, référence = 1.0)")
            rates_by_id = {}
            for k, v in rates.items():
                if isinstance(k, bool):
                    raise DeckError("clé YAML booléenne dans 'rates' — "
                                    "coter le nom du matériau ('on': …)")
                if float(v) <= 0:
                    raise DeckError(f"taux <= 0 pour {k!r} : {v!r}")
                rates_by_id[self._mid(k)] = float(v)
            until = self._mid(step["until"]) if "until" in step else None
            get_step(self.backend, "etch_selective", mode)(
                g, t, rates_by_id, until_material=until,
                mask=self._mask(step.get("mask"), where))
            self._log(f"etch_selective {mode} {t} vx "
                      f"({', '.join(f'{k}×{v:g}' for k, v in rates.items())})")

        elif op == "grow":
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            frm = self._mid(step["from"])
            mat = self._mid(step["material"])
            exp = float(step.get("expansion", 1.0))
            if exp <= 0:
                raise DeckError(f"'expansion' doit être > 0 : {exp!r}")
            into = self._mids(step.get("into")) or None
            get_step(self.backend, "grow")(
                g, t, frm, mat, expansion=exp, into=into,
                mask=self._mask(step.get("mask"), where))
            self._log(f"grow {step['material']} <- {step['from']} "
                      f"({t} vx consommés, expansion {exp:g})")

        elif op == "epitaxy":
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            mat = self._mid(step["material"])
            seed = self._mid(step["seed"])
            mode = step.get("mode", "seed")
            mask = self._mask(step.get("mask"), where)
            if mode == "seed":
                get_step(self.backend, "epitaxy", "seed")(
                    g, t, mat, seed, mask=mask)
                self._log(f"epitaxy seed {step['material']} {t} vx "
                          f"(graine {step['seed']})")
            elif mode == "crystal":
                ori, clean = self._crystal_step_params(
                    step, {"100": 1.0, "110": 0.6, "111": 0.25})
                get_step(self.backend, "epitaxy", "crystal")(
                    g, t, mat, orientation=ori, facet_rates=clean,
                    seed=seed, mask=mask)
                self._log(f"epitaxy crystal {step['material']} {t} vx "
                          f"({ori}, {clean})")
            else:
                raise DeckError(f"mode epitaxy inconnu : {mode!r}")

        elif op == "etch_crystal":
            t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
            mat = self._mid(step["material"])
            ori, clean = self._crystal_step_params(
                step, {"100": 1.0, "110": 1.6, "111": 0.1})
            get_step(self.backend, "etch_crystal")(
                g, t, mat, orientation=ori, facet_rates=clean,
                mask=self._mask(step.get("mask"), where))
            self._log(f"etch_crystal {step['material']} {t} vx "
                      f"({ori}, {clean})")

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

    # --- checkpoints entre étapes (issue #12) --------------------------------

    CKPT_FORMATS = ("png", "h5", "vtk", "stl")

    def _maybe_checkpoint(self, i: int, step: dict):
        """Sauvegarde l'état de la grille après l'étape i si demandé :
        - au niveau étape : `checkpoint: true` (formats par défaut), ou
          `checkpoint: [png, h5]`, ou `checkpoint: {png: a.png, h5: b.h5}` ;
        - globalement : `checkpoints: {every: 5, dir: ckpt, formats: [...]}`
          (toutes les `every` étapes).
        Nommage auto : {outdir}/{dir}/step_{i:04d}_{op}.{ext}.
        """
        gck = self.deck.get("checkpoints") or {}
        if not isinstance(gck, dict):
            raise DeckError(f"'checkpoints:' doit être un dict, pas {gck!r}")
        every = int(gck.get("every", 0) or 0)
        gdir = str(gck.get("dir", "ckpt"))
        gformats = list(gck.get("formats", ["png", "h5"]))
        specs: list = []
        sck = step.get("checkpoint")
        if sck:
            if sck is True:
                specs += [(f, None) for f in gformats]
            elif isinstance(sck, (list, tuple)):
                specs += [(f, None) for f in sck]
            elif isinstance(sck, dict):
                specs += list(sck.items())
            else:
                raise DeckError(f"étape {i} : 'checkpoint' invalide {sck!r} "
                                "(true | [formats] | {format: fichier})")
        if every and i % every == 0:
            specs += [(f, None) for f in gformats]
        if not specs:
            return
        for f, _ in specs:
            if str(f).lower() not in self.CKPT_FORMATS:
                raise DeckError(f"étape {i} : format de checkpoint inconnu "
                                f"{f!r} (png | h5 | vtk | stl)")
        # mode device-résident GPU : D2H temporaire (le résident continue)
        if getattr(self.g, "_gpu_resident", False):
            from .ops.gpu import sync_to_host
            sync_to_host(self.g)
        base = Path(self.outdir)
        auto_dir = base / gdir
        auto_dir.mkdir(parents=True, exist_ok=True)
        op = str(step.get("op", "step"))
        for fmt, name in specs:
            fmt = str(fmt).lower()
            path = (base / str(name)) if name else \
                auto_dir / f"step_{i:04d}_{op}.{fmt}"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._export_checkpoint(fmt, path, i, step)
            self._log(f"checkpoint étape {i} ({op}) -> {path.name} [{fmt}]")
            print(f"[checkpoint] étape {i} ({op}) -> {path} [{fmt}]")

    def _export_checkpoint(self, fmt: str, path: Path, i: int, step: dict):
        from .export import export_h5, export_stl, export_vtk, render_views
        g = self.g
        if fmt == "png":
            render_views(g, filename=str(path),
                         title=f"étape {i} — {step.get('op', '?')}")
        elif fmt == "h5":
            prov = self.provenance()
            export_h5(g, filename=str(path),
                      provenance={**{k: v for k, v in prov.items()
                                     if k != "steps"},
                                  "checkpoint_step": i})
        elif fmt == "vtk":
            export_vtk(g, filename=str(path))
        elif fmt == "stl":
            export_stl(g, prefix=str(path)[:-4] if str(path).endswith(".stl")
                       else str(path))

    def provenance(self) -> dict:
        """Bloc de provenance : tout ce qu'il faut pour rejouer un run."""
        return {
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

    def outputs(self):
        from .export import export_h5, export_stl, export_vtk, render_views
        self.outdir.mkdir(parents=True, exist_ok=True)
        for out in self.deck.get("outputs", []):
            kind, file = out.get("type"), out.get("file", "out")
            path = str(self.outdir / file)
            if kind == "views":
                render_views(self.g, filename=path,
                             title=f"voxelfab — {self.deck_path or 'deck'}")
            elif kind == "h5":
                # sous-ensemble déterministe : pas de timings dans le fichier
                export_h5(self.g, filename=path,
                          provenance={k: v for k, v in self.provenance().items()
                                      if k != "steps"})
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
        prov = self.provenance()
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
