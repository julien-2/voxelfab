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
        cell = lib.top_cells()[0]
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
                (compensation de la gravure latérale), via shapely.buffer
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
        if bias_nm:
            geom = geom.buffer(bias_nm * 1e-3)
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
