"""
voxelfab.py — Mini-moteur voxel de simulation de procédé (type SEMulator3D).

Version 2 : noyau vectorisé, calibré pour des grilles de 1 à 100 millions de
voxels (grille uint8 dense ~10-100 Mo). Aucune équation physique lourde :
uniquement de la morphologie mathématique sur des tableaux NumPy.

Choix de performance appliqués :
  1. Deposit conformal / gravure isotrope = dilation / érosion booléennes
     (plus de distance_transform_edt -> plus de temporaire float64 de 8 octets
     par voxel). Petite épaisseur -> élément structurant boule euclidienne ;
     grande épaisseur -> itérations de dilation 6-voisine (métrique L1,
     octaèdre), en O(N) par itération.
  2. CMP / deposit directionnel : plus aucune boucle Python par colonne ;
     tout est broadcast sur le champ de hauteur (height map).
  3. Hygiène mémoire : tout reste uint8/bool (1 octet/voxel), sélections de
     matériaux par table de correspondance (LUT) à 256 entrées au lieu de
     np.isin, tampons réutilisés entre passes.
  4. Gravure directionnelle : noyau Numba @njit(parallel=True) qui parcourt
     chaque colonne en séquentiel (sémantique exacte : sauts d'air, arrêt sur
     matériau) mais en parallèle sur tous les coeurs. Retour vectorisé NumPy
     (dépouillement par couches) si Numba n'est pas installé.

Dépendances : numpy, scipy, matplotlib ; optionnel : numba, pyvista (--3d)
    pip install numpy scipy matplotlib numba
    python voxelfab.py            # flow CMOS simplifié (coupe 2D)
    python voxelfab.py --mos3d    # flow MOSFET 3D paramétrable + vues 3D
    python voxelfab.py --bench 256  # micro-benchmark sur grille 256^3
"""

from __future__ import annotations

import operator
import sys
from functools import reduce

import numpy as np
from scipy import ndimage

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# --------------------------------------------------------------------------
# Matériaux
# --------------------------------------------------------------------------

class Materials:
    """Registre des matériaux : id voxel -> (nom, couleur). 0 = air."""
    AIR = 0

    def __init__(self):
        self._names = {0: ("air", "#000000")}

    def register(self, name: str, color: str) -> int:
        mid = len(self._names)
        self._names[mid] = (name, color)
        return mid

    def name(self, mid: int) -> str:
        return self._names[mid][0]

    def color(self, mid: int) -> str:
        return self._names[mid][1]

    @property
    def ids(self):
        return list(self._names.keys())


# --------------------------------------------------------------------------
# Grille voxel
# --------------------------------------------------------------------------

class VoxelGrid:
    """
    Grille 3D multi-matériaux, dtype uint8 (1 octet/voxel).
    Axes : grid[x, y, z], z vertical (z = nz-1 = surface).
    `pitch` = nm par voxel, pour exprimer les étapes en nm.
    """

    def __init__(self, nx: int, ny: int, nz: int, pitch: float = 1.0):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.pitch = pitch
        self.data = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.materials = Materials()
        # index de colonnes, alloués une seule fois et réutilisés
        self._xs = np.arange(nx)[:, None]   # (nx, 1)
        self._ys = np.arange(ny)[None, :]   # (1, ny)
        self._zs = np.arange(nz)[None, None, :]  # (1, 1, nz)

    # --- helpers -----------------------------------------------------------

    def add_material(self, name: str, color: str) -> int:
        return self.materials.register(name, color)

    def voxels(self, thickness_nm: float) -> int:
        """Épaisseur nm -> nombre de voxels (>= 1 si > 0)."""
        n = int(round(thickness_nm / self.pitch))
        return max(n, 1) if thickness_nm > 0 else 0

    def solid(self, out: np.ndarray | None = None) -> np.ndarray:
        """Masque booléen des voxels non-air (tampon `out` réutilisable)."""
        if out is None:
            out = np.empty(self.data.shape, dtype=bool)
        np.not_equal(self.data, 0, out=out)
        return out

    def select(self, materials: tuple | None = None) -> np.ndarray:
        """Masque booléen des voxels appartenant aux matériaux donnés.
        LUT 256 entrées : O(N) sans tri ni boucle, contrairement à np.isin."""
        if materials is None:
            return self.solid()
        lut = np.zeros(256, dtype=bool)
        lut[np.fromiter(materials, dtype=np.int64, count=len(materials))] = True
        return lut[self.data]

    def height_map(self) -> np.ndarray:
        """Altitude du plus haut voxel solide par colonne (nx, ny) ; -1 si vide.
        Entièrement vectorisé : un seul argmax sur l'axe z retourné."""
        solid = self.data != 0
        has = solid.any(axis=2)
        h = np.full((self.nx, self.ny), -1, dtype=np.int32)
        first_from_top = np.argmax(solid[:, :, ::-1], axis=2)
        h[has] = (self.nz - 1) - first_from_top[has]
        return h

    def volume_fractions(self) -> dict:
        total = self.nx * self.ny * self.nz
        return {
            self.materials.name(mid): round(float((self.data == mid).sum()) / total, 4)
            for mid in self.materials.ids if mid != Materials.AIR
        }

    # --- morphologie interne --------------------------------------------------

    @staticmethod
    def _ball(radius: int) -> np.ndarray:
        """Élément structurant boule euclidienne (métrique L2 exacte).
        Éléments mis en cache : recréer une boule à chaque étape coûte cher."""
        r = np.arange(-radius, radius + 1)
        d = np.hypot(np.hypot(r[:, None, None], r[None, :, None]),
                     r[None, None, :])
        return d <= radius + 0.5

    _ball_cache: dict = {}

    def ball(self, radius: int) -> np.ndarray:
        if radius not in self._ball_cache:
            self._ball_cache[radius] = self._ball(radius)
        return self._ball_cache[radius]

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Dilation booléenne par une 'sphère' de rayon `radius` voxels.
        rayon <= 4 : boule euclidienne (exact, |SE| reste petit).
        rayon >  4 : t itérations de la croix 6-voisine -> boule L1
        (octaèdre), en O(N) par itération au lieu de O(N*|SE|)."""
        if radius <= 4:
            return ndimage.binary_dilation(mask, structure=self.ball(radius))
        return ndimage.binary_dilation(mask, iterations=radius)

    def _erode(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 4:
            return ndimage.binary_erosion(mask, structure=self.ball(radius),
                                          border_value=1)
        return ndimage.binary_erosion(mask, iterations=radius, border_value=1)


# --------------------------------------------------------------------------
# Masques lithographiques (champs 2D sur (x, y))
# --------------------------------------------------------------------------

def mask_line(nx: int, ny: int, x0: float, x1: float) -> np.ndarray:
    m = np.zeros((nx, ny), dtype=bool)
    m[int(x0 * nx):int(x1 * nx), :] = True
    return m


def mask_rectangle(nx: int, ny: int, x0: float, x1: float,
                   y0: float, y1: float) -> np.ndarray:
    m = np.zeros((nx, ny), dtype=bool)
    m[int(x0 * nx):int(x1 * nx), int(y0 * ny):int(y1 * ny)] = True
    return m


def invert(mask: np.ndarray) -> np.ndarray:
    return ~mask


# --------------------------------------------------------------------------
# Import GDS/OASIS : masques lithographiques depuis un fichier de layout
# --------------------------------------------------------------------------

def _read_layout(filename: str, cell_name: str | None = None):
    """Lit un fichier GDSII ou OASIS avec gdstk et retourne la cellule cible
    aplatie (les références SREF/AREF sont résolues). Les coordonnées sont
    en µm (unité par défaut de gdstk)."""
    try:
        import gdstk
    except ImportError:
        raise SystemExit("Import GDS/OASIS : pip install gdstk")
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
    pour Process.litho(). Dépendances optionnelles : gdstk + shapely (>= 2).

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
        raise SystemExit("Import GDS/OASIS : pip install shapely")
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


# --------------------------------------------------------------------------
# Noyau Numba : gravure directionnelle (sémantique séquentielle par colonne,
# parallélisée sur tous les coeurs). -- étape 4 --
# --------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _etch_directional_kernel(data, open_mask, t, stop_on, allowed):
        """
        Pour chaque colonne ouverte du masque :
          - part du sommet solide, descend en sautant les voxels d'air ;
          - retire jusqu'à t voxels de matériau autorisé ;
          - s'arrête sur un matériau d'arrêt ou non autorisé.
        data       : uint8 (nx, ny, nz), modifié sur place
        open_mask  : bool (nx, ny)
        allowed    : bool (256,) LUT des matériaux gravables
        stop_on    : int (-1 = aucun)
        """
        nx, ny, nz = data.shape
        for i in prange(nx * ny):
            x = i // ny
            y = i % ny
            if not open_mask[x, y]:
                continue
            z = -1
            for zz in range(nz - 1, -1, -1):        # sommet solide
                if data[x, y, zz] != 0:
                    z = zz
                    break
            dug = 0
            while z >= 0 and dug < t:
                mid = data[x, y, z]
                if mid == 0:                        # air : on traverse
                    z -= 1
                    continue
                if stop_on >= 0 and mid == stop_on:  # matériau d'arrêt
                    break
                if not allowed[mid]:                # matériau non gravable
                    break
                data[x, y, z] = 0
                dug += 1
                z -= 1


def _etch_directional_fallback(g, t, stop_on, materials, open_mask):
    """Retour vectorisé (sans Numba) : on dépouille une couche de voxels
    par itération ; t est petit, chaque passe est un O(N) vectorisé."""
    allowed = _allowed_lut(materials)
    open_ = open_mask if open_mask is not None else np.ones((g.nx, g.ny), bool)
    for _ in range(t):
        h = g.height_map()
        xs, ys = np.nonzero((h >= 0) & open_)
        if xs.size == 0:
            break
        z = h[xs, ys]
        mid = g.data[xs, ys, z]
        rem = allowed[mid]
        if stop_on is not None:
            rem &= mid != stop_on
        if not rem.any():
            break
        g.data[xs[rem], ys[rem], z[rem]] = 0


def _allowed_lut(materials: tuple | None) -> np.ndarray:
    """LUT booléenne (256,) : True partout si materials est None."""
    lut = np.ones(256, dtype=bool)
    if materials is not None:
        lut = np.zeros(256, dtype=bool)
        lut[np.fromiter(materials, dtype=np.int64, count=len(materials))] = True
    return lut


# --------------------------------------------------------------------------
# Étapes de procédé (process steps)
# --------------------------------------------------------------------------

class Process:
    def __init__(self, grid: VoxelGrid):
        self.g = grid
        self.log: list[str] = []

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"  [flow] {msg}")

    # --- initialisation ----------------------------------------------------

    def substrate(self, material: int, thickness_nm: float):
        t = self.g.voxels(thickness_nm)
        self.g.data[:, :, :t] = material
        self._log(f"substrate {self.g.materials.name(material)} ({t} vx)")

    # --- dépôt ---------------------------------------------------------------

    def deposit(self, material: int, thickness_nm: float, mode: str = "conformal",
                on_materials: tuple | None = None):
        """
        conformal  : couche d'épaisseur constante sur toute surface exposée
                     (CVD) = dilation(solid, t) privé du solide. Aucun EDT,
                     donc aucun temporaire float64.
        directional: dépôt vertical au-dessus des surfaces exposées (PVD),
                     vectorisé par broadcast du champ de hauteur.
        on_materials : (directional) dépôt sélectif : uniquement sur les
                     colonnes dont le sommet est l'un de ces matériaux.
        """
        t = self.g.voxels(thickness_nm)
        if t == 0:
            return
        g = self.g

        if mode == "conformal":
            solid = g.solid()
            shell = g._dilate(solid, t) & ~solid
            g.data[shell] = material

        elif mode == "directional":
            h = g.height_map()
            active = h >= 0
            if on_materials is not None:
                # LUT indexée par le matériau du sommet : pas de masque 3D
                top_mat = g.data[g._xs, g._ys, np.clip(h, 0, g.nz - 1)]
                active &= _allowed_lut(on_materials)[top_mat]
            z0 = np.clip(h + 1, 0, g.nz)[..., None]
            z1 = np.clip(h + 1 + t, 0, g.nz)[..., None]
            shell = active[..., None] & (g._zs >= z0) & (g._zs < z1)
            g.data[shell] = material
        else:
            raise ValueError(f"mode de dépôt inconnu : {mode}")

        self._log(f"deposit {mode} {g.materials.name(material)} ({t} vx)")

    # --- gravure ---------------------------------------------------------------

    def etch(self, thickness_nm: float, mode: str = "isotropic",
             materials: tuple | None = None, until_material: int | None = None,
             mask: np.ndarray | None = None):
        """
        mode isotropic : retrait uniforme dans toutes les directions
                         = érosion(solid, t) ; sélectivité via LUT.
        mode directional: gravure verticale (RIE) ; noyau Numba si disponible,
                         sinon dépouillement vectorisé.
        until_material  : matériau d'arrêt (gravure verticale uniquement).
        mask            : masque 2D (x, y) ; on ne grave que où mask = True.
        """
        t = self.g.voxels(thickness_nm)
        if t == 0:
            return
        g = self.g

        if mode == "isotropic":
            solid = g.solid()
            kept = g._erode(solid, t)
            remove = solid & ~kept
            if materials is not None:
                remove &= g.select(materials)
            if mask is not None:
                remove &= mask[:, :, None]
            g.data[remove] = Materials.AIR

        elif mode == "directional":
            open_mask = mask if mask is not None else np.ones((g.nx, g.ny), bool)
            allowed = _allowed_lut(materials)
            stop = -1 if until_material is None else until_material
            if HAS_NUMBA:
                _etch_directional_kernel(g.data, open_mask, t, stop, allowed)
            else:
                _etch_directional_fallback(g, t, stop if stop >= 0 else None,
                                           materials, mask)
        else:
            raise ValueError(f"mode de gravure inconnu : {mode}")

        self._log(f"etch {mode} {t} vx"
                 + (f" sur {tuple(g.materials.name(m) for m in materials)}"
                    if materials else ""))

    # --- lithographie ------------------------------------------------------------

    def litho(self, mask: np.ndarray, thickness_nm: float,
              materials: tuple | None = None,
              until_material: int | None = None):
        """Lithographie + transfert : gravure verticale à travers le masque."""
        self.etch(thickness_nm, mode="directional",
                  materials=materials, until_material=until_material, mask=mask)
        self._log(f"litho (zone ouverte = {mask.mean() * 100:.0f} % de la surface)")

    # --- CMP -----------------------------------------------------------------------

    def cmp(self, target_z_voxels: int | None = None,
            stop_on: int | None = None):
        """
        Planarisation CMP, entièrement vectorisée (champ de hauteur) :
        abrase tout voxel solide au-dessus du plan cible ; `stop_on` n'est
        jamais traversé (première occurrence depuis la surface, par argmax
        sur l'axe z retourné — un seul passage vectorisé).
        """
        g = self.g
        target = target_z_voxels
        solid = g.solid()
        remove = solid & (g._zs > target)
        if stop_on is not None:
            st = (g.data == stop_on)                       # (nx, ny, nz)
            has = st.any(axis=2)                           # (nx, ny)
            zstop = (g.nz - 1) - np.argmax(st[:, :, ::-1], axis=2)
            remove &= ~has[..., None] | (g._zs > zstop[..., None])
        g.data[remove] = Materials.AIR
        self._log(f"CMP -> plan z={target} ({int(remove.sum())} vx retirés)")

    # --- remplissage de tranchée -------------------------------------------------------

    def fill_trench(self, material: int, max_z: int):
        """
        Remplit les tranchées : volumes d'air connectés à la surface et
        situés sous le plan z <= max_z (surface du wafer avant gravure).
        ndimage.label alloue un int32 temporaire (4 octets/voxel) : c'est la
        seule étape coûteuse en mémoire ; acceptable car appelée rarement.
        """
        g = self.g
        air = ~g.solid()
        lbl, _ = ndimage.label(air)
        top_labels = set(np.unique(lbl[:, :, -1]))
        top_labels.discard(0)
        below = g._zs <= max_z
        shell = np.isin(lbl, np.fromiter(top_labels, dtype=lbl.dtype,
                                         count=len(top_labels))) & below
        g.data[shell] = material
        self._log(f"fill {g.materials.name(material)} (tranchées sous z={max_z})")


# --------------------------------------------------------------------------
# Métrologie virtuelle
# --------------------------------------------------------------------------

class Metrology:
    def __init__(self, grid: VoxelGrid):
        self.g = grid

    def thickness_at(self, x: int, y: int, material: int) -> int:
        return int((self.g.data[x, y, :] == material).sum())

    def top_material(self, x: int, y: int) -> str:
        h = self.g.height_map()
        return self.g.materials.name(self.g.data[x, y, h[x, y]])

    def cd(self, y: int, material: int) -> int:
        """Plus grande largeur continue du matériau dans la coupe y."""
        m = (self.g.data[:, y, :] == material).any(axis=1)
        best = cur = 0
        for v in m:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    def report(self) -> dict:
        return {"volume_fractions": self.g.volume_fractions()}


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------

def plot_cross_section(grid: VoxelGrid, y: int, title: str,
                       filename: str = "flow_cmos.png"):
    if not HAS_MPL:
        print("matplotlib absent : pas de figure.")
        return
    from matplotlib.colors import to_rgb
    sec = grid.data[:, y, :].T                      # (nz, nx)
    rgb = np.zeros((*sec.shape, 3))
    for mid in grid.materials.ids:
        rgb[sec == mid] = to_rgb(grid.materials.color(mid))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(rgb, origin="lower", aspect="equal",
              extent=[0, grid.nx, 0, grid.nz])
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("z (voxels)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"  [plot] coupe sauvegardée : {filename}")
    plt.show()


# --------------------------------------------------------------------------
# Rendu 3D rapide (projections orthogonales) + exports optionnels
# --------------------------------------------------------------------------

def render_views(grid: VoxelGrid, filename: str = "mos3d.png",
                 title: str = "MOSFET 3D"):
    """Rendu 3D économique : projection du dessus (matériau de surface,
    via le champ de hauteur) + coupes x-z et y-z. O(N) vectorisé,
    contrairement à Axes3D.voxels qui est inutilisable au-delà de ~50k voxels."""
    if not HAS_MPL:
        print("matplotlib absent : pas de figure.")
        return
    from matplotlib.colors import to_rgb
    rgb = np.zeros((256, 3))
    for mid in grid.materials.ids:
        rgb[mid] = to_rgb(grid.materials.color(mid))

    # vue de dessus : couleur du matériau au sommet de chaque colonne
    h = grid.height_map()
    top = grid.data[grid._xs, grid._ys, np.clip(h, 0, grid.nz - 1)]
    top_view = rgb[top].transpose(1, 0, 2)          # (ny, nx, 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].imshow(top_view, origin="lower",
                   extent=[0, grid.nx, 0, grid.ny])
    axes[0].set_title("vue de dessus (surface)")
    axes[0].set_xlabel("x (vx)"); axes[0].set_ylabel("y (vx)")

    sec = grid.data[:, grid.ny // 2, :].T            # coupe x-z
    axes[1].imshow(rgb[sec], origin="lower", aspect="equal",
                   extent=[0, grid.nx, 0, grid.nz])
    axes[1].set_title(f"coupe x-z (y={grid.ny // 2})")
    axes[1].set_xlabel("x (vx)"); axes[1].set_ylabel("z (vx)")

    sec = grid.data[grid.nx // 2, :, :].T            # coupe y-z
    axes[2].imshow(rgb[sec], origin="lower", aspect="equal",
                   extent=[0, grid.ny, 0, grid.nz])
    axes[2].set_title(f"coupe y-z (x={grid.nx // 2})")
    axes[2].set_xlabel("y (vx)"); axes[2].set_ylabel("z (vx)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"  [plot] vues 3D sauvegardées : {filename}")
    plt.show()


def export_vtk(grid: VoxelGrid, filename: str = "mos3d.vtk"):
    """Export .vtk pour ParaView/PyVista (si pyvista est installé)."""
    try:
        import pyvista as pv
    except ImportError:
        print("  [export] pyvista absent : export VTK ignoré")
        return
    img = pv.ImageData(dimensions=(grid.nx + 1, grid.ny + 1, grid.nz + 1))
    img.cell_data["material"] = grid.data.flatten(order="F").astype(np.int32)
    img.save(filename)
    print(f"  [export] {filename} ({grid.nx * grid.ny * grid.nz} cellules)")


def export_stl(grid: VoxelGrid, prefix: str = "mos3d"):
    """Export STL par material (marching cubes, si scikit-image installé)."""
    try:
        from skimage import measure
        from stl import mesh as stl_mesh
    except ImportError:
        print("  [export] scikit-image/numpy-stl absent : export STL ignoré")
        return
    for mid in grid.materials.ids:
        if mid == Materials.AIR:
            continue
        vol = grid.data == mid
        if not vol.any():
            continue
        v, f, _, _ = measure.marching_cubes(vol.astype(np.uint8), level=0.5)
        m = stl_mesh.Mesh(np.zeros(f.shape[0], dtype=stl_mesh.Mesh.dtype))
        m.vectors = v[f]
        out = f"{prefix}_{grid.materials.name(mid).split()[0]}.stl"
        m.save(out)
        print(f"  [export] {out} ({f.shape[0]} faces)")




# --------------------------------------------------------------------------
# Micro-benchmark (--bench)
# --------------------------------------------------------------------------

def bench(n: int = 256):
    """Chronomètre chaque type d'étape sur une grille n x n x n.
    n = 256 -> 16,7 M voxels ; n = 430 -> ~80 M voxels."""
    import time
    g = VoxelGrid(n, n, n, pitch=1.0)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    SiN = g.add_material("SiN", "#8bc34a")
    p = Process(g)
    p.substrate(SiO2, thickness_nm=n // 2)
    print(f"\n=== bench {g.nx}x{g.ny}x{g.nz} = "
          f"{g.nx * g.ny * g.nz / 1e6:.1f} M voxels (numba: {HAS_NUMBA}) ===")

    def timed(label, fn):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        print(f"  {label:<28} {dt * 1e3:8.0f} ms")

    timed("deposit conformal t=5", lambda: p.deposit(SiN, 5))
    timed("deposit conformal t=20", lambda: p.deposit(SiN, 20))
    timed("deposit directional t=10", lambda: p.deposit(SiN, 10, "directional"))
    timed("etch isotropic t=5", lambda: p.etch(5, "isotropic"))
    timed("etch isotropic t=20", lambda: p.etch(20, "isotropic"))
    mask = np.zeros((g.nx, g.ny), bool)
    mask[::4, :] = True
    timed("etch directional t=30", lambda: p.etch(30, "directional", mask=mask))
    timed("cmp", lambda: p.cmp(target_z_voxels=n - 8, stop_on=SiO2))
    timed("height_map", g.height_map)


# --------------------------------------------------------------------------
# Flow CMOS ultra-simplifié (démo)
# --------------------------------------------------------------------------

def demo_cmos():
    print("=== voxelfab : flow CMOS simplifié ===")
    pitch = 2.0  # nm/voxel
    g = VoxelGrid(nx=100, ny=40, nz=60, pitch=pitch)
    p = Process(g)
    m = g.materials

    Si = g.add_material("Si (substrat)", "#4d79ff")
    SiO2 = g.add_material("SiO2 (oxyde)", "#c9c9c9")
    SiN = g.add_material("Si3N4 (nitrure)", "#8bc34a")
    poly = g.add_material("Poly-Si (grille)", "#e64a19")
    W = g.add_material("W (métal)", "#5d4037")

    # 1. Substrat
    p.substrate(Si, thickness_nm=20 * pitch)

    # 2. STI : gravure de tranchées, remplissage oxyde, CMP
    print("\n-- STI --")
    p.litho(mask_line(g.nx, g.ny, 0.05, 0.20), thickness_nm=12)
    p.litho(mask_line(g.nx, g.ny, 0.80, 0.95), thickness_nm=12)
    p.fill_trench(SiO2, max_z=19)
    p.cmp(target_z_voxels=19, stop_on=Si)

    # 3. Empilement grille : oxyde fin + poly-Si
    print("\n-- Gate stack --")
    p.deposit(SiO2, thickness_nm=2 * pitch, mode="conformal")
    p.deposit(poly, thickness_nm=18 * pitch, mode="conformal")

    # 4. Litho grille : gravure du poly (arrêt sur l'oxyde), puis retrait
    #    de l'oxyde exposé hors grille -> Si exposé pour l'épitaxie S/D
    print("\n-- Gate patterning --")
    gate_mask = mask_line(g.nx, g.ny, 0.35, 0.50)
    p.litho(invert(gate_mask), thickness_nm=30 * pitch, materials=(poly,))
    p.etch(thickness_nm=2 * pitch, mode="isotropic", materials=(SiO2,),
           mask=invert(gate_mask))

    # 5. Spacers : dépôt conformal nitrure + RIE directionnel de même
    #    épaisseur : les surfaces horizontales sont dénudées, seules les
    #    parois verticales de grille gardent leur nitrure.
    print("\n-- Spacers --")
    p.deposit(SiN, thickness_nm=8, mode="conformal")
    p.etch(thickness_nm=8, mode="directional", materials=(SiN,))

    # 6. Épitaxie S/D : dépôt directionnel sélectif au Si exposé
    print("\n-- Source/Drain (épi simplifiée) --")
    p.deposit(Si, thickness_nm=12, mode="directional", on_materials=(Si,))

    # 7. ILD : oxyde épais + CMP
    print("\n-- ILD + CMP --")
    p.deposit(SiO2, thickness_nm=25 * pitch, mode="conformal")
    p.cmp(target_z_voxels=48)

    # 8. Contact : ouverture gravée jusqu'à la grille (matériau d'arrêt)
    print("\n-- Contact --")
    p.litho(mask_line(g.nx, g.ny, 0.40, 0.45), thickness_nm=25 * pitch,
            until_material=poly)
    p.deposit(W, thickness_nm=25 * pitch, mode="directional")
    p.cmp(target_z_voxels=48)

    # 9. Métrologie virtuelle
    print("\n-- Métrologie virtuelle --")
    met = Metrology(g)
    print(f"  CD grille (coupe y=20) : {met.cd(20, poly) * pitch:.1f} nm")
    print(f"  Épaisseur oxyde grille  : "
          f"{met.thickness_at(45, 20, SiO2) * pitch:.1f} nm")
    print(f"  Largeur spacer nitrure  : {met.cd(20, SiN) * pitch:.1f} nm")
    print(f"  Volume des matériaux   : {met.report()['volume_fractions']}")

    # 10. Visualisation
    plot_cross_section(g, y=20, title="Flow CMOS simplifié — coupe (x, z)")


# --------------------------------------------------------------------------
# Flow MOSFET 3D paramétrable (--mos3d)
# --------------------------------------------------------------------------

def mos3d_flow(cfg: dict | None = None) -> VoxelGrid:
    """
    Transistor MOS planaire entièrement 3D : STI, grille en ruban le long
    de y, spacers, épitaxie S/D, contacts W rectangulaires. Toutes les
    dimensions sont des fractions de grille ou des nm : modifiez `cfg`
    pour expérimenter (c'est le 'process input' du flow).

    cfg : voir MOS3D_CONFIG ci-dessous ; une valeur None -> défaut.
    """
    c = dict(MOS3D_CONFIG)
    if cfg:
        c.update(cfg)

    # Masques : soit depuis un layout GDS/OASIS, soit géométriques
    gds = c.get("gds_file")
    if gds:
        print(f"-- Layout : {gds} --")
        bbox, lay = layout_info(gds, c.get("gds_cell"))
        print(f"  bbox = {bbox} µm ; couches = {lay}")
        c["nx"], c["ny"] = grid_size_for(bbox, c["pitch"])
        g = VoxelGrid(c["nx"], c["ny"], c["nz"], pitch=c["pitch"])
        layers = tuple(c["gds_layers"].values())
        masks = masks_from_layout(gds, g, layers=layers, bbox=bbox,
                                 cell_name=c.get("gds_cell"),
                                 bias_nm=c.get("mask_bias_nm", 0.0))
    else:
        g = VoxelGrid(c["nx"], c["ny"], c["nz"], pitch=c["pitch"])
        masks = None

    p = Process(g)
    Si   = g.add_material("Si (substrat)",     "#4d79ff")
    SiO2 = g.add_material("SiO2 (oxyde)",     "#c9c9c9")
    SiN  = g.add_material("Si3N4 (nitrure)",  "#8bc34a")
    poly = g.add_material("Poly-Si (grille)", "#e64a19")
    W    = g.add_material("W (métal)",        "#5d4037")

    z_plane = g.voxels(c["substrate_nm"]) - 1  # plan de surface du substrat

    # 1. Substrat
    p.substrate(Si, thickness_nm=c["substrate_nm"])

    # 2. STI : tranchées d'isolation de part et d'autre de la zone active
    print("\n-- STI --")
    if masks:
        p.litho(masks[c["gds_layers"]["sti"]], thickness_nm=c["trench_nm"])
    else:
        for fx0, fx1 in c["sti_bands"]:
            p.litho(mask_line(g.nx, g.ny, fx0, fx1), thickness_nm=c["trench_nm"])
    p.fill_trench(SiO2, max_z=z_plane)
    p.cmp(target_z_voxels=z_plane, stop_on=Si)

    # 3. Empilement grille
    print("\n-- Gate stack --")
    p.deposit(SiO2, thickness_nm=c["gate_ox_nm"], mode="conformal")
    p.deposit(poly, thickness_nm=c["poly_nm"], mode="conformal")

    # 4. Patterning : poly gravé hors grille (arrêt sur l'oxyde), puis
    #    retrait de l'oxyde exposé -> Si affleurant pour l'épitaxie
    print("\n-- Gate patterning --")
    if masks:
        gate_mask = masks[c["gds_layers"]["gate"]]
    else:
        gate_mask = mask_rectangle(g.nx, g.ny, *c["gate"])
    outside = invert(gate_mask)
    p.litho(outside, thickness_nm=c["gate_ox_nm"] + c["poly_nm"] + 10 * c["pitch"],
            materials=(poly,))
    p.etch(thickness_nm=c["gate_ox_nm"], mode="isotropic",
           materials=(SiO2,), mask=outside)

    # 5. Spacers : dépôt conformal + RIE directionnel de même épaisseur
    #    (les surfaces horizontales sont dénudées, les parois de grille
    #    conservent leur nitrure)
    print("\n-- Spacers --")
    p.deposit(SiN, thickness_nm=c["spacer_dep_nm"], mode="conformal")
    p.etch(thickness_nm=c["spacer_dep_nm"], mode="directional",
           materials=(SiN,))

    # 6. Épitaxie S/D sélective
    print("\n-- Source/Drain --")
    p.deposit(Si, thickness_nm=c["epi_nm"], mode="directional",
              on_materials=(Si,))

    # 7. ILD + CMP
    print("\n-- ILD + CMP --")
    p.deposit(SiO2, thickness_nm=c["ild_nm"], mode="conformal")
    p.cmp(target_z_voxels=c["ild_top_z"])

    # 8. Contacts S/D (rectangles 3D) + bouchons W
    print("\n-- Contacts --")
    if masks:
        for key in ("contact_s", "contact_d"):
            p.litho(masks[c["gds_layers"][key]],
                    thickness_nm=c["contact_nm"], until_material=Si)
    else:
        for fx0, fx1, fy0, fy1 in c["contacts"]:
            p.litho(mask_rectangle(g.nx, g.ny, fx0, fx1, fy0, fy1),
                    thickness_nm=c["contact_nm"], until_material=Si)
    p.deposit(W, thickness_nm=c["plug_nm"], mode="directional")
    p.cmp(target_z_voxels=c["ild_top_z"])

    return g


MOS3D_CONFIG = {
    # grille
    "nx": 120, "ny": 60, "nz": 64, "pitch": 2.0,   # nm/voxel
    # matériaux déposés
    "substrate_nm": 40.0,
    "trench_nm": 12.0,            # profondeur STI
    "gate_ox_nm": 4.0, "poly_nm": 36.0,
    # masques (fractions de la grille)
    "sti_bands": [(0.00, 0.18), (0.82, 1.00)],    # bandes ouvertes en x
    "gate": (0.34, 0.52, 0.08, 0.92),             # fx0, fx1, fy0, fy1
    "spacer_dep_nm": 8.0,                          # RIE directionnel = dépôt
    "epi_nm": 12.0,
    "ild_nm": 50.0, "ild_top_z": 52,              # plan de CMP
    "contacts": [(0.18, 0.30, 0.30, 0.70),        # source
                 (0.70, 0.82, 0.30, 0.70)],       # drain
    "contact_nm": 60.0, "plug_nm": 60.0,
    # entrée layout : gds_file -> les masques viennent du layout,
    # la grille (nx, ny) est déduite de la bbox du layout et du pitch
    "gds_file": None,          # ex. "inv.gds" ; None = masques géométriques
    "gds_cell": None,          # cellule top ; None = première top-cell
    "gds_layers": {"sti": (1, 0), "gate": (10, 0),
                   "contact_s": (20, 0), "contact_d": (21, 0)},
    "mask_bias_nm": 0.0,       # biais de masque (nm), + = élargit
}


def _mat_id(g: VoxelGrid, name_start: str) -> int:
    for mid in g.materials.ids:
        if g.materials.name(mid).startswith(name_start):
            return mid
    return -1


def demo_mos_3d(cfg: dict | None = None):
    print("=== voxelfab : MOSFET 3D ===")
    g = mos3d_flow(cfg)

    print("\n-- Métrologie virtuelle --")
    met = Metrology(g)
    poly = _mat_id(g, "Poly")
    print(f"  CD grille (coupe y={g.ny // 2}) : "
          f"{met.cd(g.ny // 2, poly) * g.pitch:.1f} nm")
    print(f"  Volume des matériaux         : {met.report()['volume_fractions']}")

    render_views(g, filename="mos3d.png",
                 title="MOSFET 3D — vue de dessus + coupes x-z / y-z")
    export_vtk(g)
    export_stl(g)



if __name__ == "__main__":
    if "--bench" in sys.argv:
        i = sys.argv.index("--bench")
        bench(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 256)
    elif "--mos3d" in sys.argv:
        # accepte un layout en argument : --mos3d [layout.gds|layout.oas]
        rest = sys.argv[sys.argv.index("--mos3d") + 1:]
        gds = next((a for a in rest
                    if a.lower().endswith((".gds", ".gds.gz", ".oas"))), None)
        demo_mos_3d({"gds_file": gds} if gds else None)
    else:
        demo_cmos()
