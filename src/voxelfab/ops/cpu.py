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
