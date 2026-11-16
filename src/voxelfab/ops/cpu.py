"""Backend CPU (référence) : NumPy/SciPy vectorisé, Numba optionnel.

CONTRAT (ops/base.py) : les épaisseurs sont en VOXELS ; la conversion
nm -> voxels est faite par l'orchestrateur (run.py). Toute implémentation
(cpu aujourd'hui, gpu demain) doit produire une grille identique octet
pour octet pour les mêmes entrées.

Choix de performance :
  1. deposit conformal / etch isotropic = dilation / érosion booléennes
     (pas de distance_transform_edt -> pas de temporaire float64).
     rayon <= 4 : boule euclidienne exacte ; au-delà : itérations de la
     croix 6-voisine (boule L1), O(N) par itération ; grow réutilise la
     dilation pour la consommation, puis un front de croissance à budget
     par colonne (croix 6-voisine).
  2. cmp / deposit directional : tout est broadcast sur le champ de
     hauteur, aucune boucle Python par colonne.
  3. Hygiène mémoire : uint8/bool partout (1 octet/voxel), sélections de
     matériaux par LUT 256 entrées au lieu de np.isin.
  4. etch directional / etch_selective directional : noyaux Numba
     @njit(parallel=True) — parcours séquentiel par colonne (sémantique
     exacte), parallèle sur les colonnes ; retour vectorisé NumPy si
     Numba est absent.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..grid import Materials
from .base import register

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# --------------------------------------------------------------------------
# Morphologie
# --------------------------------------------------------------------------

_BALL_CACHE: dict = {}


def ball(radius: int) -> np.ndarray:
    """Élément structurant boule euclidienne (métrique L2 exacte), en cache."""
    if radius not in _BALL_CACHE:
        r = np.arange(-radius, radius + 1)
        d = np.hypot(np.hypot(r[:, None, None], r[None, :, None]),
                     r[None, None, :])
        _BALL_CACHE[radius] = d <= radius + 0.5
    return _BALL_CACHE[radius]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 4:
        return ndimage.binary_dilation(mask, structure=ball(radius))
    return ndimage.binary_dilation(mask, iterations=radius)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 4:
        return ndimage.binary_erosion(mask, structure=ball(radius),
                                      border_value=1)
    return ndimage.binary_erosion(mask, iterations=radius, border_value=1)


def _allowed_lut(materials) -> np.ndarray:
    """LUT booléenne (256,) : True partout si materials est None."""
    lut = np.ones(256, dtype=bool)
    if materials is not None:
        lut = np.zeros(256, dtype=bool)
        lut[np.fromiter(materials, dtype=np.int64, count=len(materials))] = True
    return lut


def _rates_lut(rates: dict) -> np.ndarray:
    """LUT float64 (256,) des taux relatifs ; 0.0 = matériau non gravé."""
    lut = np.zeros(256, dtype=np.float64)
    for mid, rate in rates.items():
        lut[mid] = float(rate)
    return lut


# --------------------------------------------------------------------------
# Noyaux Numba : gravures directionnelles (sémantique séquentielle par
# colonne, parallélisée sur tous les coeurs)
# --------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _etch_directional_kernel(data, open_mask, t, stop_on, allowed):
        """
        Pour chaque colonne ouverte du masque :
          - part du sommet solide, descend en sautant les voxels d'air ;
          - retire jusqu'à t voxels de matériau autorisé ;
          - s'arrête sur un matériau d'arrêt ou non autorisé.
        data      : uint8 (nx, ny, nz), modifié sur place
        open_mask : bool (nx, ny)
        allowed   : bool (256,) LUT des matériaux gravables
        stop_on   : int (-1 = aucun)
        """
        nx, ny, nz = data.shape
        for i in prange(nx * ny):
            x = i // ny
            y = i % ny
            if not open_mask[x, y]:
                continue
            z = -1
            for zz in range(nz - 1, -1, -1):         # sommet solide
                if data[x, y, zz] != 0:
                    z = zz
                    break
            dug = 0
            while z >= 0 and dug < t:
                mid = data[x, y, z]
                if mid == 0:                          # air : on traverse
                    z -= 1
                    continue
                if stop_on >= 0 and mid == stop_on:   # matériau d'arrêt
                    break
                if not allowed[mid]:                  # non gravable
                    break
                data[x, y, z] = 0
                dug += 1
                z -= 1

    @njit(parallel=True, cache=True)
    def _etch_selective_kernel(data, open_mask, budget, stop_on, rates):
        """
        Gravure directionnelle à sélectivité différentielle.
        Pour chaque colonne ouverte du masque, budget de temps B = budget :
          - part du sommet solide, descend en sautant les voxels d'air ;
          - retirer un voxel du matériau m coûte 1/rates[m] ;
          - s'arrête sur un matériau d'arrêt, absent de rates (taux 0),
            ou quand le budget ne suffit plus pour le voxel courant.
        data    : uint8 (nx, ny, nz), modifié sur place
        rates   : float64 (256,) taux relatifs (0 = non gravé)
        """
        nx, ny, nz = data.shape
        for i in prange(nx * ny):
            x = i // ny
            y = i % ny
            if not open_mask[x, y]:
                continue
            z = -1
            for zz in range(nz - 1, -1, -1):         # sommet solide
                if data[x, y, zz] != 0:
                    z = zz
                    break
            b = budget
            while z >= 0 and b > 0.0:
                mid = data[x, y, z]
                if mid == 0:                          # air : on traverse
                    z -= 1
                    continue
                if stop_on >= 0 and mid == stop_on:   # matériau d'arrêt
                    break
                r = rates[mid]
                if r <= 0.0:                          # absent de rates
                    break
                cost = 1.0 / r
                if b < cost:                          # budget insuffisant
                    break
                data[x, y, z] = 0
                b -= cost
                z -= 1


def _etch_directional_fallback(grid, t, stop_on, materials, open_mask):
    """Retour vectorisé (sans Numba) : on dépouille une couche de voxels
    par itération ; t est petit, chaque passe est un O(N) vectorisé."""
    allowed = _allowed_lut(materials)
    open_ = open_mask if open_mask is not None else np.ones(
        (grid.nx, grid.ny), bool)
    for _ in range(t):
        h = grid.height_map()
        xs, ys = np.nonzero((h >= 0) & open_)
        if xs.size == 0:
            break
        z = h[xs, ys]
        mid = grid.data[xs, ys, z]
        rem = allowed[mid]
        if stop_on is not None:
            rem &= mid != stop_on
        if not rem.any():
            break
        grid.data[xs[rem], ys[rem], z[rem]] = 0


def _etch_selective_fallback(grid, budget, rates_lut, open_mask,
                             until_material):
    """Retour vectorisé (sans Numba) de la gravure sélective : dépouille
    une couche par passe, avec un budget de temps par colonne."""
    b = np.full((grid.nx, grid.ny), float(budget))
    open_ = open_mask if open_mask is not None else np.ones(
        (grid.nx, grid.ny), bool)
    while True:
        h = grid.height_map()
        xs, ys = np.nonzero((h >= 0) & open_ & (b > 0.0))
        if xs.size == 0:
            break
        z = h[xs, ys]
        mid = grid.data[xs, ys, z]
        rate = rates_lut[mid]
        cost = np.where(rate > 0, 1.0 / np.where(rate > 0, rate, 1.0),
                        np.inf)
        rem = (rate > 0) & (b[xs, ys] >= cost)
        if until_material is not None:
            rem &= mid != until_material
        if not rem.any():
            break
        grid.data[xs[rem], ys[rem], z[rem]] = 0
        b[xs[rem], ys[rem]] -= cost[rem]


# --------------------------------------------------------------------------
# Étapes enregistrées (unités : VOXELS)
# --------------------------------------------------------------------------

@register("cpu", "substrate")
def substrate(grid, material, t):
    grid.data[:, :, :t] = material


@register("cpu", "deposit", "conformal")
def deposit_conformal(grid, material, t, on_materials=None):
    if t == 0:
        return
    solid = grid.solid()
    shell = _dilate(solid, t) & ~solid
    grid.data[shell] = material


@register("cpu", "deposit", "directional")
def deposit_directional(grid, material, t, on_materials=None):
    if t == 0:
        return
    h = grid.height_map()
    active = h >= 0
    if on_materials is not None:
        # LUT indexée par le matériau du sommet : pas de masque 3D
        top_mat = grid.data[grid._xs, grid._ys, np.clip(h, 0, grid.nz - 1)]
        active &= _allowed_lut(on_materials)[top_mat]
    z0 = np.clip(h + 1, 0, grid.nz)[..., None]
    z1 = np.clip(h + 1 + t, 0, grid.nz)[..., None]
    shell = active[..., None] & (grid._zs >= z0) & (grid._zs < z1)
    grid.data[shell] = material


@register("cpu", "etch", "isotropic")
def etch_isotropic(grid, t, materials=None, until_material=None, mask=None):
    if t == 0:
        return
    solid = grid.solid()
    kept = _erode(solid, t)
    remove = solid & ~kept
    if materials is not None:
        remove &= grid.select(materials)
    if mask is not None:
        remove &= mask[:, :, None]
    grid.data[remove] = Materials.AIR


@register("cpu", "etch", "directional")
def etch_directional(grid, t, materials=None, until_material=None, mask=None):
    if t == 0:
        return
    open_mask = mask if mask is not None else np.ones((grid.nx, grid.ny), bool)
    allowed = _allowed_lut(materials)
    stop = -1 if until_material is None else until_material
    if HAS_NUMBA:
        _etch_directional_kernel(grid.data, open_mask, t, stop, allowed)
    else:
        _etch_directional_fallback(grid, t, stop if stop >= 0 else None,
                                   materials, mask)


@register("cpu", "etch_selective", "directional")
def etch_selective_directional(grid, depth, rates, until_material=None,
                               mask=None):
    if depth <= 0 or not rates:
        return
    rates_lut = _rates_lut(rates)
    open_mask = mask if mask is not None else np.ones((grid.nx, grid.ny), bool)
    stop = -1 if until_material is None else until_material
    if HAS_NUMBA:
        _etch_selective_kernel(grid.data, open_mask, float(depth), stop,
                               rates_lut)
    else:
        _etch_selective_fallback(grid, float(depth), rates_lut, open_mask,
                                 until_material)


@register("cpu", "etch_selective", "isotropic")
def etch_selective_isotropic(grid, depth, rates, until_material=None,
                             mask=None):
    """Chaque matériau m est érodé de round(depth * rate_m) ; les matériaux
    absents de rates sont intacts. until_material est ignoré (cf. contrat)."""
    if depth <= 0 or not rates:
        return
    solid = grid.solid()
    keep = ~grid.select(tuple(rates.keys()))   # hors rates : intact
    for mid, rate in rates.items():
        r = int(round(depth * float(rate)))
        m = grid.data == mid
        if r <= 0:
            keep |= m
        else:
            keep |= _erode(solid, r) & m
    remove = solid & ~keep
    if mask is not None:
        remove &= mask[:, :, None]
    grid.data[remove] = Materials.AIR


def _dilate6(m: np.ndarray) -> np.ndarray:
    """Dilation 6-voisine (croix), par slices — pas de temporaire scipy."""
    out = np.zeros_like(m)
    out[:, :-1, :] |= m[:, 1:, :]
    out[:, 1:, :] |= m[:, :-1, :]
    out[:-1, :, :] |= m[1:, :, :]
    out[1:, :, :] |= m[:-1, :, :]
    out[:, :, :-1] |= m[:, :, 1:]
    out[:, :, 1:] |= m[:, :, :-1]
    return out


@register("cpu", "grow")
def grow(grid, depth, from_id, product_id, expansion=1.0, into=None,
         mask=None):
    """Croissance par conversion (oxydation, siliciuration) — cf. contrat
    ops/base.py. Tout vectorisé : consommation par dilation, puis front de
    croissance par itérations de la croix, budget proportionnel par
    colonne (le cumul par colonne alloue un int32 temporaire, comme fill)."""
    if depth <= 0 or expansion <= 0:
        return
    data = grid.data
    react = data == Materials.AIR
    if into:
        react |= grid.select(tuple(into))
    if mask is not None:
        react &= mask[:, :, None]
    # consommation : voxels de `from` à distance <= depth du côté réactif
    # (minceur épuisée : tout est consommé, le budget suivra au prorata)
    consume = _dilate(react, depth) & (data == from_id)
    if mask is not None:
        consume &= mask[:, :, None]
    if not consume.any():
        return
    data[consume] = product_id
    d2 = int(round(depth * (float(expansion) - 1.0)))
    if d2 <= 0:
        return
    # budget de croissance proportionnel par colonne, propagé au voisinage
    # (maximum filter) pour oxyder aussi les parois latérales
    n = consume.sum(axis=2, dtype=np.int64)
    b = np.rint(n * (float(expansion) - 1.0)).astype(np.int64)
    b = ndimage.maximum_filter(b, size=2 * d2 + 1)
    grown = np.zeros(data.shape, dtype=bool)
    cnt = np.zeros((grid.nx, grid.ny), dtype=np.int64)
    front = consume
    for _ in range(d2):
        nxt = _dilate6(front) & react & ~consume & ~grown
        if not nxt.any():
            break
        ranks = np.cumsum(nxt, axis=2, dtype=np.int32)
        keep = nxt & (ranks + cnt[:, :, None] <= b[:, :, None])
        if not keep.any():
            break
        grown |= keep
        cnt += keep.sum(axis=2, dtype=np.int64)
        front = grown | consume
        if bool((cnt >= b).all()):
            break
    if grown.any():
        data[grown] = product_id


# --------------------------------------------------------------------------
# Épitaxie : SEG conforme + croissance facettée (carte de hauteur)
# --------------------------------------------------------------------------

# axes grille (x, y, z) exprimés dans la base cristalline (entiers, à
# normaliser) — orthogonaux deux à deux par construction
_ORIENTATIONS = {
    "100": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "110": ((1, -1, 0), (0, 0, 1), (1, 1, 0)),
    "111": ((1, 1, -2), (1, -1, 0), (1, 1, 1)),
}

# normales de plans par famille {hkl} (cristal, à signes près : la
# déduplication par (|nx|, |ny|, nz) en repère grille les complète)
_FACET_NORMALS = {
    "100": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "110": ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
    "111": ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)),
}


def _min_scan(m: np.ndarray, s: float, axis: int) -> np.ndarray:
    """res[i] = min_j (m[j] + s·|i−j|) le long de `axis` (deux passages
    accumulate, O(N))."""
    n = m.shape[axis]
    shape = [1] * m.ndim
    shape[axis] = n
    idx = np.arange(n, dtype=np.float64).reshape(shape)
    fwd = np.minimum.accumulate(m - s * idx, axis=axis) + s * idx
    acc = np.minimum.accumulate(np.flip(m, axis=axis) - s * idx, axis=axis)
    bwd = np.flip(acc, axis=axis) + s * ((n - 1) - idx)
    return np.minimum(fwd, bwd)


def _min_pyramid(a: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """min sur (dx, dy) de a[x+dx, y+dy] + sx·|dx| + sy·|dy|
    (a : +inf hors ancres) — séparable par construction."""
    return _min_scan(_min_scan(a, sx, 0), sy, 1)


@register("cpu", "epitaxy", "seed")
def epitaxy_seed(grid, depth, material, seed, mask=None):
    """SEG conforme : flood géodésique 6-voisin depuis la graine à travers
    l'air (croissance sur toutes les faces exposées de la graine, bloquée
    par les matériaux inertes) — cf. contrat ops/base.py."""
    if depth <= 0:
        return
    data = grid.data
    air = data == Materials.AIR
    if mask is not None:
        air &= mask[:, :, None]
    reach = np.zeros(data.shape, dtype=bool)
    front = _dilate6(data == seed) & air
    for _ in range(int(depth)):
        if not front.any():
            break
        reach |= front
        front = _dilate6(reach) & air & ~reach
    if reach.any():
        data[reach] = material


@register("cpu", "epitaxy", "crystal")
def epitaxy_crystal(grid, depth, material, orientation="100",
                    facet_rates=None, seed=None, mask=None):
    """Épitaxie facettée (construction SEG : plans lents partant des bords
    de la graine exposée) — cf. contrat ops/base.py. Carte de hauteur,
    tout vectorisé (min-filtres pyramidaux séparables, aucun temporaire
    3D float64)."""
    if depth <= 0:
        return
    data = grid.data
    nx, ny, nz = data.shape
    t = float(depth)
    rates = dict(facet_rates or {"100": 1.0, "110": 0.6, "111": 0.25})
    vmax = max(rates.values())
    Gx, Gy, Gz = _ORIENTATIONS[orientation]
    # plans distincts (|nx|, |ny|, nz) par famille, en repère grille
    planes = []  # (sx, sy, 1/nz, taux)
    for fam, v in rates.items():
        seen = set()
        for n in _FACET_NORMALS[fam]:
            ng = np.array([np.dot(n, Gx) / np.dot(Gx, Gx),
                           np.dot(n, Gy) / np.dot(Gy, Gy),
                           np.dot(n, Gz) / np.dot(Gz, Gz)])
            ng /= np.linalg.norm(ng)
            if ng[2] <= 0.05:      # facette quasi verticale : hors carte
                continue           # de hauteur (documenté)
            key = (round(abs(ng[0]), 6), round(abs(ng[1]), 6),
                   round(ng[2], 6))
            if key in seen:
                continue
            seen.add(key)
            planes.append((key[0] / key[2], key[1] / key[2], 1.0 / key[2],
                           float(v) / vmax))
    # surface de la graine exposée à l'air, par colonne (top seed voxel)
    zidx = np.arange(nz, dtype=np.int64)
    zseed = np.where(data == seed, zidx[None, None, :],
                     np.int64(-1)).max(axis=2)
    if not (zseed >= 0).any():
        return
    top_air = np.take_along_axis(
        data == Materials.AIR, np.minimum(zseed + 1, nz - 1)[..., None],
        axis=2)[..., 0]
    surf = (zseed >= 0) & top_air
    # ancres : colonnes de graine exposée au bord (voisin sans surface de
    # graine, ou de hauteur différente)
    zsr = np.where(surf, zseed.astype(np.float64), np.inf)
    # bords de la graine exposée, PAR AXE : un voisin hors grille n'est pas
    # un bord (le wafer continue) -> on réplique ; un voisin sans surface
    # de graine ou de hauteur différente (>= 1/2 voxel) en est un.
    edge_x = np.zeros((nx, ny), dtype=bool)
    edge_y = np.zeros((nx, ny), dtype=bool)
    for axis in (0, 1):
        for sgn in (-1, 1):
            nb = zsr.copy()
            if axis == 0 and sgn == 1:
                nb[1:, :] = zsr[:-1, :]
            elif axis == 0:
                nb[:-1, :] = zsr[1:, :]
            elif sgn == 1:
                nb[:, 1:] = zsr[:, :-1]
            else:
                nb[:, :-1] = zsr[:, 1:]
            diff = np.where(np.isfinite(nb) & np.isfinite(zsr),
                            np.abs(nb - zsr), np.inf)
            bad = (~np.isfinite(nb)) | (diff >= 0.5)
            if axis == 0:
                edge_x |= surf & bad
            else:
                edge_y |= surf & bad
    # surface résultante : min(terme vertical, plans de facettes)
    znew = np.full((nx, ny), np.inf)
    for sx, sy, inv_nz, v in planes:
        if sx == 0.0 and sy == 0.0:
            # plan parallèle au wafer : croissance verticale par colonne
            cand = np.where(surf, zseed + v * t * inv_nz, np.inf)
        else:
            # un plan de facette contient la ligne d'arête sur laquelle il
            # repose : il ne peut s'ancre que sur les bords PERPENDICULAIRES
            # à sa montée (sx > 0 -> bord en x ; sy > 0 -> bord en y ;
            # les deux -> coins seulement). Sinon un plan (0, sy) ancré sur
            # un bord en x couvrirait toute la rangée à basse hauteur.
            anch_ok = surf
            if sx > 0.0:
                anch_ok = anch_ok & edge_x
            if sy > 0.0:
                anch_ok = anch_ok & edge_y
            anch = np.where(anch_ok, zseed + v * t * inv_nz, np.inf)
            cand = _min_pyramid(anch, sx, sy)
        znew = np.minimum(znew, cand)
    znew = np.where(surf, znew, np.inf)
    # remplissage : air au-dessus de la graine, jusqu'à znew, sans
    # traverser un matériau (surplomb -> on s'arrête)
    zfill = np.where(np.isfinite(znew),
                     np.rint(np.minimum(znew, nz - 1)),
                     np.int64(-1)).astype(np.int64)
    zi = zidx[None, None, :]
    blocked = (data != Materials.AIR) & (zi > zseed[..., None])
    zblock = np.where(blocked, zi, np.int64(nz)).min(axis=2)
    fill = ((data == Materials.AIR) & (zseed >= 0)[..., None]
            & (zi > zseed[..., None]) & (zi <= zfill[..., None])
            & (zi < zblock[..., None]))
    if mask is not None:
        fill &= mask[:, :, None]
    if fill.any():
        data[fill] = material


@register("cpu", "fill")
def fill_trench(grid, material, max_z):
    """
    Remplit les tranchées : volumes d'air connectés à la surface et situés
    sous le plan z <= max_z. ndimage.label alloue un int32 temporaire
    (4 octets/voxel) : seule étape coûteuse en mémoire, appelée rarement.
    """
    air = ~grid.solid()
    lbl, _ = ndimage.label(air)
    top_labels = set(np.unique(lbl[:, :, -1]))
    top_labels.discard(0)
    below = grid._zs <= max_z
    shell = np.isin(lbl, np.fromiter(top_labels, dtype=lbl.dtype,
                                     count=len(top_labels))) & below
    grid.data[shell] = material


@register("cpu", "cmp")
def cmp(grid, target_z_voxels, stop_on=None):
    """
    Planarisation CMP, entièrement vectorisée (champ de hauteur).
    Retourne le nombre de voxels retirés.
    """
    solid = grid.solid()
    remove = solid & (grid._zs > target_z_voxels)
    if stop_on is not None:
        st = (grid.data == stop_on)                    # (nx, ny, nz)
        has = st.any(axis=2)                           # (nx, ny)
        zstop = (grid.nz - 1) - np.argmax(st[:, :, ::-1], axis=2)
        remove &= ~has[..., None] | (grid._zs > zstop[..., None])
    grid.data[remove] = Materials.AIR
    return int(remove.sum())
