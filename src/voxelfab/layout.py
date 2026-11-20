"""Import GDSII / OASIS : masques lithographiques depuis un fichier de layout.

Dépendances optionnelles : gdstk (lecture) + shapely >= 2 (rastérisation
vectorisée aux centres de voxels). Importées paresseusement : le coeur du
moteur fonctionne sans elles.
"""

from __future__ import annotations

import operator
from functools import reduce

import numpy as np

from .grid import VoxelGrid


def _read_layout(filename: str, cell_name: str | None = None):
    """Lit un fichier GDSII ou OASIS avec gdstk et retourne la cellule cible
    aplatie (les références SREF/AREF sont résolues). Les coordonnées sont
    en µm (unité par défaut de gdstk)."""
    try:
        import gdstk
    except ImportError:
        raise SystemExit("Import GDS/OASIS : pip install gdstk "
                         "(ou uv pip install 'voxelfab[layout]')")
    if filename.lower().endswith((".gds", ".gds.gz")):
        lib = gdstk.read_gds(filename)
    elif filename.lower().endswith((".oas", ".oasis")):
        lib = gdstk.read_oas(filename)
    else:
        raise ValueError("format non reconnu : .gds ou .oas attendu")
    if cell_name is None:
        # gdstk >= 1.0 : top_level() ; avant : top_cells()
        tops = (lib.top_level() if hasattr(lib, "top_level")
                else lib.top_cells())
        cell = tops[0]
    else:
        cell = {c.name: c for c in lib.cells}[cell_name]
    cell.flatten()
    return cell


def layout_info(filename: str, cell_name: str | None = None):
    """Bounding box ((x0, y0), (x1, y1)) en µm de la cellule cible,
    et liste des couches (layer, datatype) présentes."""
    cell = _read_layout(filename, cell_name)
    layers = sorted({(p.layer, p.datatype) for p in cell.polygons})
    return cell.bounding_box(), layers


def grid_size_for(bbox, pitch_nm: float) -> tuple:
    """Dimensions (nx, ny) pour couvrir `bbox` (µm) au pas donné (nm)."""
    pitch_um = pitch_nm * 1e-3
    (x0, y0), (x1, y1) = bbox
    return (max(int(np.ceil((x1 - x0) / pitch_um)), 1),
            max(int(np.ceil((y1 - y0) / pitch_um)), 1))


def masks_from_layout(filename: str, grid: VoxelGrid,
                      layers: tuple | None = None,
                      bbox: tuple | None = None,
                      cell_name: str | None = None,
                      bias_nm: float = 0.0,
                      evenodd: bool = False) -> dict:
    """
    Rastérise les couches d'un layout en masques booléens (nx, ny), prêts
    pour l'étape litho. Dépendances optionnelles : gdstk + shapely (>= 2).

    filename  : .gds / .gds.gz / .oas
    grid      : grille cible ; les CENTRES des voxels échantillonnent le
                layout (alignement < 1/2 voxel garanti par construction)
    layers    : liste de (layer, datatype) à importer ; None = toutes
    bbox      : région d'intérêt ((x0, y0), (x1, y1)) en µm ;
                None = bounding box de la cellule
    bias_nm   : biais de masque : élargit (+) / rétrécit (-) les polygones
                (compensation de la gravure latérale), via shapely.buffer.
                Accepte un float (toutes les couches) OU un dict
                {(layer, datatype): float} pour un biais par couche.
    evenodd   : True = règle even-odd (les polygones imbriqués deviennent
                des trous, convention GDS pour les openings) ; False =
                union simple. Chaque couche retournée est alors le XOR
                de tous ses polygones.

    Retour : {(layer, datatype): bool (nx, ny)} + une clé spéciale
    "bbox" donnant la bbox effectivement échantillonnée (µm).
    """
    try:
        import shapely
    except ImportError:
        raise SystemExit("Import GDS/OASIS : pip install shapely "
                         "(ou uv pip install 'voxelfab[layout]')")
    cell = _read_layout(filename, cell_name)
    if bbox is None:
        bbox = cell.bounding_box()
    (x0, y0), (x1, y1) = bbox

    # centres des voxels en coordonnées layout (µm) -> (nx, ny)
    pitch_um = grid.pitch * 1e-3
    xs = x0 + (np.arange(grid.nx) + 0.5) * pitch_um
    ys = y0 + (np.arange(grid.ny) + 0.5) * pitch_um

    # collecte des polygones par couche (paths convertis), hors ROI exclus
    polys = list(cell.get_polygons())
    for path in cell.get_paths():                    # FlexPath/RobustPath
        try:
            polys.extend(path.to_polygons())
        except Exception:                            # API variable : on ignore
            print("  [gds] ATTENTION : path non converti dans le masque")
    wanted = set(layers) if layers is not None else None
    by_layer: dict = {}
    for poly in polys:
        key = (poly.layer, poly.datatype)
        if wanted is not None and key not in wanted:
            continue
        (px0, py0), (px1, py1) = poly.bounding_box()
        if px1 < x0 or px0 > x1 or py1 < y0 or py0 > y1:
            continue
        by_layer.setdefault(key, []).append(np.asarray(poly.points))

    X, Y = np.meshgrid(xs, ys, indexing="ij")        # (nx, ny)
    masks: dict = {"bbox": bbox}
    for key, pts in by_layer.items():
        geoms = [shapely.Polygon(p) for p in pts]
        if evenodd:
            geom = reduce(operator.xor, geoms)       # XOR = even-odd
        else:
            geom = shapely.union_all(geoms)
        b = bias_nm.get(key, 0.0) if isinstance(bias_nm, dict) else bias_nm
        if b:
            geom = geom.buffer(b * 1e-3)
        shapely.prepare(geom)
        m = shapely.contains_xy(geom, X.ravel(), Y.ravel())
        m = m.reshape(grid.nx, grid.ny)
        if not m.any() and wanted is not None:
            print(f"  [gds] ATTENTION couche {key} : vide à ce pas "
                  f"({grid.pitch} nm/voxel) — raffiner la grille ?")
        masks[key] = m
    for key in (wanted or set()) - set(by_layer):
        print(f"  [gds] ATTENTION couche {key} absente du layout")
    return masks


def _norm_ld(ld) -> list:
    """Normalise le champ `ld` d'une spec de couche en liste de (layer,
    datatype) :
      [10, 0]            -> [(10, 0)]
      [10, [0, 4]]       -> [(10, 0), (10, 4)]   (merge de datatypes)
      [[10, 0], [11, 2]] -> [(10, 0), (11, 2)]   (merge de couches)
    """
    if isinstance(ld, (list, tuple)) and ld and isinstance(ld[0], (list, tuple)):
        pairs = list(ld)                                 # [[l, dt], ...]
    elif isinstance(ld, (list, tuple)) and len(ld) == 2:
        layer, dt = ld
        pairs = [(layer, d) for d in dt] if isinstance(dt, (list, tuple)) \
            else [(layer, dt)]
    else:
        raise ValueError(f"ld invalide : {ld!r} (attendu [layer, datatype], "
                         "[layer, [datatypes]] ou [[layer, datatype], ...])")
    try:
        out = [(int(l), int(d)) for l, d in pairs]
    except (TypeError, ValueError):
        raise ValueError(f"ld invalide : {ld!r} (layer/datatype entiers "
                         "attendus)") from None
    if not out:
        raise ValueError("ld invalide : aucune paire (layer, datatype)")
    return out


def parse_layer_spec(name, spec, default_bias: float = 0.0) -> dict:
    """Normalise une spec de couche du bloc `layout: layers:` (rétrocompatible).

      gate: [10, 0]                                        (format v1)
      gate: [10, [0, 4]]                                   (merge datatypes)
      gate: {ld: [10, [0, 4]], polarity: clear, bias_nm: -5}

    Retour : {"ld": [(layer, datatype), ...],
              "polarity": "dark" | "clear",
              "bias_nm": float}
    - polarity "dark" (défaut) : le masque imprime le motif tel que dessiné ;
      "clear" : c'est le complémentaire qui est imprimé dans la résine.
    - bias_nm local ; par défaut le biais global du bloc layout.
    """
    if isinstance(spec, dict):
        if "ld" not in spec:
            raise ValueError(f"couche {name!r} : champ 'ld' requis dans {spec!r}")
        ld = _norm_ld(spec["ld"])
        polarity = str(spec.get("polarity", "dark")).lower()
        if polarity not in ("dark", "clear"):
            raise ValueError(f"couche {name!r} : polarity {polarity!r} invalide "
                             "(dark | clear)")
        bias = float(spec.get("bias_nm", default_bias))
    else:
        ld = _norm_ld(spec)
        polarity = "dark"
        bias = float(default_bias)
    return {"ld": ld, "polarity": polarity, "bias_nm": bias}


def layer_masks_from_layout(filename: str, grid: VoxelGrid, layers: dict,
                            cell_name: str | None = None,
                            bbox: tuple | None = None,
                            bias_nm: float = 0.0,
                            evenodd: bool = False) -> dict:
    """Masques nommés depuis un layout, avec polarité et biais par couche.

    layers : {nom_logique: spec} — spec = [layer, datatype] (v1) ou
             {ld: ..., polarity: dark|clear, bias_nm: ...} (cf. parse_layer_spec)
    bias_nm : biais global (défaut des couches qui n'en déclarent pas).

    Retour : {nom_logique: bool (nx, ny)}. Le merge multi-datatypes/multi-
    couches est l'union des masques ; la polarité s'applique APRÈS le merge
    ("clear" -> complémentaire du motif dessiné).
    """
    specs = {name: parse_layer_spec(name, spec, bias_nm)
             for name, spec in layers.items()}
    wanted = sorted({ld for s in specs.values() for ld in s["ld"]})
    bias_map: dict = {}
    for name, s in specs.items():
        for ld in s["ld"]:
            if ld in bias_map and bias_map[ld] != s["bias_nm"]:
                raise ValueError(
                    f"conflit de bias sur la couche {ld} : {bias_map[ld]} nm "
                    f"vs {s['bias_nm']} nm (couche {name!r}) — un seul bias "
                    "par (layer, datatype)")
            bias_map[ld] = s["bias_nm"]
    masks = masks_from_layout(filename, grid, layers=tuple(wanted),
                              bbox=bbox, cell_name=cell_name,
                              bias_nm=bias_map, evenodd=evenodd)
    out: dict = {}
    for name, s in specs.items():
        m = np.zeros((grid.nx, grid.ny), dtype=bool)
        for ld in s["ld"]:
            if ld in masks:
                m |= masks[ld]
        if s["polarity"] == "clear":
            m = ~m
        out[name] = m
    return out
