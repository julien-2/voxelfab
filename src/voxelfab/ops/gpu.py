"""Backend GPU (Taichi) — v0.2.

CONTRAT (ops/base.py) : même sémantique que le backend CPU, grille identique
octet pour octet pour les mêmes entrées (vérifié par tests/test_equiv_gpu.py).

Design (cf. docs/ARCHITECTURE.md §5 « Taichi design notes ») :
  - un `_GpuCtx` par shape de grille, mis en cache : les champs `ti.field`
    et les noyaux (closures sur ces champs — Taichi capture les champs au
    moment de la compilation du noyau) ;
  - chaque étape synchronise hôte -> device en entrée et device -> hôte en
    sortie : `grid.data` (NumPy) reste la source de vérité côté hôte, les
    sorties (HDF5, métrologie, vues) ne changent pas ;
  - déterminisme : réductions min/max uniquement (atomics entiers pour les
    compteurs), arithmétique f64 effectuée dans le même ordre que NumPy,
    aucun atomics flottant -> hash identique au CPU.

Mémoire : ~6 tampons u8 3D (d + 5 travail) + champs 2D => ~600 Mo à 10^8
voxels, OK sur un GPU 6 Go.

NOTE : pas de `from __future__ import annotations` ici — les annotations des
noyaux Taichi doivent être de vrais objets de types (ti.f64, ...), pas des
chaînes (PEP 563).
"""

import os

import numpy as np

from ..grid import Materials
from .base import register
from .cpu import _allowed_lut, _crystal_planes, _rates_lut

try:

    import taichi as ti
    HAS_TAICHI = True

except ImportError:                                    # extra [gpu] absente

    ti = None
    HAS_TAICHI = False


# --------------------------------------------------------------------------
# Helpers hôte
# --------------------------------------------------------------------------

_BALL_OFF: dict = {}


def _ball_offsets(r: int) -> np.ndarray:
    """Offsets (n, 3) int32 de l'élément structurant, identiques au CPU :
    r <= 4 : boule euclidienne exacte (dx²+dy²+dz² <= r²+r, équivalent à
    d <= r+0.5 en f64) ; r > 4 : itérations de la croix 6-voisine = boule L1
    (|dx|+|dy|+|dz| <= r)."""
    if r not in _BALL_OFF:
        if r <= 4:
            offs = [(dx, dy, dz)
                    for dx in range(-r, r + 1)
                    for dy in range(-r, r + 1)
                    for dz in range(-r, r + 1)
                    if dx * dx + dy * dy + dz * dz <= r * r + r]
        else:
            offs = [(dx, dy, dz)
                    for dx in range(-r, r + 1)
                    for dy in range(-r, r + 1)
                    for dz in range(-r, r + 1)
                    if abs(dx) + abs(dy) + abs(dz) <= r]
        _BALL_OFF[r] = np.ascontiguousarray(np.array(offs, dtype=np.int32))
    return _BALL_OFF[r]


def _m2(mask, nx: int, ny: int) -> np.ndarray:
    """Masque 2D hôte -> ndarray u8 (nx, ny) ; None = tout ouvert."""
    if mask is None:
        return np.ones((nx, ny), dtype=np.uint8)
    return np.ascontiguousarray(mask, dtype=np.uint8)


def _lut_u8(lut_bool: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(lut_bool, dtype=np.uint8)


# --------------------------------------------------------------------------
# Taichi : init + fonctions device
# --------------------------------------------------------------------------

_INIT = False


def _ensure_init():
    global _INIT
    if _INIT:
        return
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f64, default_ip=ti.i32,
                offline_cache=True)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32,
                offline_cache=True)
    _INIT = True


_CUDA_DEVICES = None


def _cuda_device_count() -> int:
    """Nombre de devices CUDA détectés via cuInit + cuDeviceGetCount
    (nvcuda.dll / libcuda.so) — sonde ctypes SANS effet de bord sur le
    runtime Taichi (contrairement à ti.init(arch=ti.cuda))."""
    import ctypes
    for lib in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            cuda = ctypes.CDLL(lib)
            if cuda.cuInit(0) != 0:
                continue
            n = ctypes.c_int(0)
            if cuda.cuDeviceGetCount(ctypes.byref(n)) == 0:
                return n.value
        except Exception:
            continue
    return 0


def gpu_available() -> bool:
    """True si taichi est installé ET qu'un device CUDA est détecté.
    C'est le critère de ``ops.resolve_backend('auto')``."""
    global _CUDA_DEVICES
    if not HAS_TAICHI:
        return False
    if _CUDA_DEVICES is None:
        _CUDA_DEVICES = _cuda_device_count()
    return _CUDA_DEVICES > 0


if HAS_TAICHI:

    INF = float("inf")

    @ti.func
    def _rint(x: ti.f64) -> ti.i32:
        """np.rint (half-to-even) en f64, puis cast entier — identique à
        np.rint(...).astype(int64) côté CPU."""
        f = ti.floor(x)
        frac = x - f
        r = f
        if frac > 0.5:
            r = f + 1.0
        elif frac == 0.5:
            if ti.floor(f * 0.5) * 2.0 != f:      # f impair -> pair voisin
                r = f + 1.0
        return ti.cast(r, ti.i32)

    @ti.func
    def _refl(i: ti.i32, n: ti.i32) -> ti.i32:
        """Indice réfléchi façon scipy 'reflect' (half-sample symmetric) :
        j < 0 -> -j-1 ; j >= n -> 2n-1-j, replié jusqu'à [0, n)."""
        j = i
        if n == 1:
            j = 0
        else:
            while j < 0:
                j = -j - 1
            while j >= n:
                j = 2 * n - 1 - j
        return j

    # ----------------------------------------------------------------------
    # Contexte GPU d'une grille : champs + noyaux (closures sur les champs)
    # ----------------------------------------------------------------------

    class _GpuCtx:
        def __init__(self, shape):
            nx, ny, nz = shape
            self.shape = shape

            d = ti.field(ti.u8, shape=shape)          # données grille
            w1 = ti.field(ti.u8, shape=shape)         # tampons 3D
            w2 = ti.field(ti.u8, shape=shape)
            w3 = ti.field(ti.u8, shape=shape)
            w4 = ti.field(ti.u8, shape=shape)
            w5 = ti.field(ti.u8, shape=shape)
            lbl = ti.field(ti.i32, shape=shape)       # union-find (fill): label 3D
            topflag = ti.field(ti.u8, shape=(nx * ny * nz,))  # labels touchant le top
            h = ti.field(ti.i32, shape=(nx, ny))      # 2D i32 (hauteurs...)
            h2 = ti.field(ti.i32, shape=(nx, ny))     # 2D i32 (budgets...)
            s2 = ti.field(ti.u8, shape=(nx, ny))      # surface exposée
            ex2 = ti.field(ti.u8, shape=(nx, ny))     # bords en x
            ey2 = ti.field(ti.u8, shape=(nx, ny))     # bords en y
            a = ti.field(ti.f64, shape=(nx, ny))      # 2D f64 travail
            ab = ti.field(ti.f64, shape=(nx, ny))
            ac = ti.field(ti.f64, shape=(nx, ny))     # accumulateur plans
            ad = ti.field(ti.f64, shape=(nx, ny))
            flag = ti.field(ti.i32, shape=())         # compteur / changed

            self.d, self.flag = d, flag

            # --- noyau : substrat -----------------------------------------
            @ti.kernel
            def k_substrate(t: ti.i32, mat: ti.u8):
                for i, j, k in d:
                    if k < t:
                        d[i, j, k] = mat

            # --- morphologie boule ----------------------------------------
            @ti.kernel
            def k_solid_w1():
                for i, j, k in w1:
                    w1[i, j, k] = ti.cast(d[i, j, k] != 0, ti.u8)

            @ti.kernel
            def k_ball12(off: ti.types.ndarray(ti.i32, ndim=2),
                         n_off: ti.i32, erode: ti.i32):
                """w2 = dilation(w1) si erode==0 (hors grille = 0), sinon
                w2 = érosion(w1) avec border_value=1 (hors grille = solide)."""
                for i, j, k in w2:
                    if erode == 0:
                        hit = 0
                        for q in range(n_off):
                            ii = i + off[q, 0]
                            jj = j + off[q, 1]
                            kk = k + off[q, 2]
                            if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                                if w1[ii, jj, kk] != 0:
                                    hit = 1
                        w2[i, j, k] = ti.cast(hit, ti.u8)
                    else:
                        ok = 1
                        for q in range(n_off):
                            ii = i + off[q, 0]
                            jj = j + off[q, 1]
                            kk = k + off[q, 2]
                            if 0 <= ii < nx and 0 <= jj < ny \
                                    and 0 <= kk < nz:
                                if w1[ii, jj, kk] == 0:
                                    ok = 0
                        w2[i, j, k] = ti.cast(ok, ti.u8)

            @ti.kernel
            def k_conformal_set(mat: ti.u8):
                """dépôt conforme : coquille = dilate(solide) & ~solide."""
                for i, j, k in d:
                    if w2[i, j, k] != 0 and d[i, j, k] == 0:
                        d[i, j, k] = mat

            # --- croix 6-voisine itérée (ping-pong w1 <-> w2) -----------------
            # r > 4 : comme binary_dilation/erosion(iterations=r) côté CPU,
            # r passes O(N) au lieu d'une boucle sur toute la boule L1.
            @ti.kernel
            def k_cd12():
                for i, j, k in w2:
                    hit = w1[i, j, k]
                    if i > 0 and w1[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w1[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w1[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w1[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w1[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w1[i, j, k + 1] != 0:
                        hit = 1
                    w2[i, j, k] = ti.cast(hit != 0, ti.u8)

            @ti.kernel
            def k_cd21():
                for i, j, k in w1:
                    hit = w2[i, j, k]
                    if i > 0 and w2[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w2[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w2[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w2[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w2[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w2[i, j, k + 1] != 0:
                        hit = 1
                    w1[i, j, k] = ti.cast(hit != 0, ti.u8)

            @ti.kernel
            def k_ce12():
                """érosion croix, hors grille = solide (border_value=1)."""
                for i, j, k in w2:
                    ok = w1[i, j, k] != 0
                    if ok and i > 0 and w1[i - 1, j, k] == 0:
                        ok = False
                    if ok and i < nx - 1 and w1[i + 1, j, k] == 0:
                        ok = False
                    if ok and j > 0 and w1[i, j - 1, k] == 0:
                        ok = False
                    if ok and j < ny - 1 and w1[i, j + 1, k] == 0:
                        ok = False
                    if ok and k > 0 and w1[i, j, k - 1] == 0:
                        ok = False
                    if ok and k < nz - 1 and w1[i, j, k + 1] == 0:
                        ok = False
                    w2[i, j, k] = ti.cast(ok, ti.u8)

            @ti.kernel
            def k_ce21():
                for i, j, k in w1:
                    ok = w2[i, j, k] != 0
                    if ok and i > 0 and w2[i - 1, j, k] == 0:
                        ok = False
                    if ok and i < nx - 1 and w2[i + 1, j, k] == 0:
                        ok = False
                    if ok and j > 0 and w2[i, j - 1, k] == 0:
                        ok = False
                    if ok and j < ny - 1 and w2[i, j + 1, k] == 0:
                        ok = False
                    if ok and k > 0 and w2[i, j, k - 1] == 0:
                        ok = False
                    if ok and k < nz - 1 and w2[i, j, k + 1] == 0:
                        ok = False
                    w1[i, j, k] = ti.cast(ok, ti.u8)

            @ti.kernel
            def k_copy12():
                for i, j, k in w2:
                    w2[i, j, k] = w1[i, j, k]

            @ti.kernel
            def k_copy15():
                for i, j, k in w5:
                    w5[i, j, k] = w1[i, j, k]

            @ti.kernel
            def k_copy51():
                for i, j, k in w1:
                    w1[i, j, k] = w5[i, j, k]

            # --- champ de hauteur + dépôt directionnel ---------------------
            @ti.kernel
            def k_height():
                for i, j in h:
                    m = -1
                    for kk in range(nz):
                        k = nz - 1 - kk
                        if d[i, j, k] != 0:
                            m = k
                            break
                    h[i, j] = m

            @ti.kernel
            def k_dep_dir(t: ti.i32, mat: ti.u8,
                          lut: ti.types.ndarray(ti.u8, ndim=1)):
                for i, j in h:
                    hh = h[i, j]
                    if hh >= 0 and lut[d[i, j, hh]] != 0:
                        z1 = hh + 1 + t
                        if z1 > nz:
                            z1 = nz
                        for k in range(hh + 1, z1):
                            d[i, j, k] = mat

            # --- gravures ---------------------------------------------------
            @ti.kernel
            def k_etch_iso_final(lut: ti.types.ndarray(ti.u8, ndim=1),
                                 m2: ti.types.ndarray(ti.u8, ndim=2)):
                """retire = solide & ~érodé & matériaux autorisés & masque."""
                for i, j, k in d:
                    if d[i, j, k] != 0 and w2[i, j, k] == 0 \
                            and lut[d[i, j, k]] != 0 and m2[i, j] != 0:
                        d[i, j, k] = Materials.AIR

            @ti.kernel
            def k_etch_dir(t: ti.i32, stop: ti.i32,
                           open2: ti.types.ndarray(ti.u8, ndim=2),
                           allowed: ti.types.ndarray(ti.u8, ndim=1)):
                """RIE verticale : parcours séquentiel par colonne (un thread
                par colonne), identique au noyau Numba CPU."""
                for i, j in ti.ndrange(nx, ny):
                    if open2[i, j] != 0:
                        z = -1
                        for zz in range(nz):
                            zc = nz - 1 - zz
                            if d[i, j, zc] != 0:
                                z = zc
                                break
                        dug = 0
                        while z >= 0 and dug < t:
                            mid = d[i, j, z]
                            if mid == 0:
                                z -= 1
                            else:
                                if stop >= 0 and mid == stop:
                                    break
                                if allowed[mid] == 0:
                                    break
                                d[i, j, z] = Materials.AIR
                                dug += 1
                                z -= 1

            @ti.kernel
            def k_etch_sel_dir(budget: ti.f64, stop: ti.i32,
                               open2: ti.types.ndarray(ti.u8, ndim=2),
                               rates: ti.types.ndarray(ti.f64, ndim=1)):
                """Gravure sélective directionnelle : budget de temps f64 par
                colonne, mêmes opérations f64 dans le même ordre que Numba."""
                for i, j in ti.ndrange(nx, ny):
                    if open2[i, j] != 0:
                        z = -1
                        for zz in range(nz):
                            zc = nz - 1 - zz
                            if d[i, j, zc] != 0:
                                z = zc
                                break
                        b = budget
                        while z >= 0 and b > 0.0:
                            mid = d[i, j, z]
                            if mid == 0:
                                z -= 1
                            else:
                                if stop >= 0 and mid == stop:
                                    break
                                r = rates[mid]
                                if r <= 0.0:
                                    break
                                cost = 1.0 / r
                                if b < cost:
                                    break
                                d[i, j, z] = Materials.AIR
                                b -= cost
                                z -= 1

            # --- etch_selective isotrope ------------------------------------
            @ti.kernel
            def k_sel_iso_keep_init(lut_not: ti.types.ndarray(ti.u8, ndim=1)):
                for i, j, k in w3:
                    w3[i, j, k] = lut_not[d[i, j, k]]

            @ti.kernel
            def k_or_and_w3(mid: ti.u8):
                """w3 |= érodé(w1) & (d == mid) — w2 = érosion courante."""
                for i, j, k in w3:
                    if w2[i, j, k] != 0 and d[i, j, k] == mid:
                        w3[i, j, k] = 1

            @ti.kernel
            def k_sel_iso_final(m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j, k in d:
                    if d[i, j, k] != 0 and w3[i, j, k] == 0 \
                            and m2[i, j] != 0:
                        d[i, j, k] = Materials.AIR

            # --- grow (conversion) ------------------------------------------
            @ti.kernel
            def k_grow_react(into_lut: ti.types.ndarray(ti.u8, ndim=1),
                             m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j, k in w1:
                    r = 0
                    if d[i, j, k] == Materials.AIR or into_lut[d[i, j, k]] != 0:
                        r = 1
                    if m2[i, j] == 0:
                        r = 0
                    w1[i, j, k] = ti.cast(r, ti.u8)

            @ti.kernel
            def k_grow_consume(from_id: ti.u8,
                               m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j, k in w2:
                    if w2[i, j, k] != 0 and d[i, j, k] != from_id:
                        w2[i, j, k] = 0
                    if m2[i, j] == 0:
                        w2[i, j, k] = 0

            @ti.kernel
            def k_set_w2(mat: ti.u8):
                for i, j, k in d:
                    if w2[i, j, k] != 0:
                        d[i, j, k] = mat

            @ti.kernel
            def k_set_w3(mat: ti.u8):
                for i, j, k in d:
                    if w3[i, j, k] != 0:
                        d[i, j, k] = mat

            @ti.kernel
            def k_set_w4(mat: ti.u8):
                for i, j, k in d:
                    if w4[i, j, k] != 0:
                        d[i, j, k] = mat

            @ti.kernel
            def k_col_count():
                """h = nombre de voxels consommés par colonne."""
                for i, j in h:
                    c = 0
                    for k in range(nz):
                        if w2[i, j, k] != 0:
                            c += 1
                    h[i, j] = c

            @ti.kernel
            def k_col_budget(s: ti.f64):
                """h2 = rint(h * s) — budget de croissance par colonne."""
                for i, j in h2:
                    h2[i, j] = _rint(ti.f64(h[i, j]) * s)

            @ti.kernel
            def k_maxf_x(w: ti.i32):
                """max filter 1D le long de x, bord 'reflect' : h2 -> h."""
                for i, j in h:
                    m = h2[i, j]
                    for dx in range(-w, w + 1):
                        v = h2[_refl(i + dx, nx), j]
                        if v > m:
                            m = v
                    h[i, j] = m

            @ti.kernel
            def k_maxf_y(w: ti.i32):
                """max filter 1D le long de y, bord 'reflect' : h -> h2."""
                for i, j in h2:
                    m = h[i, j]
                    for dy in range(-w, w + 1):
                        v = h[i, _refl(j + dy, ny)]
                        if v > m:
                            m = v
                    h2[i, j] = m

            @ti.kernel
            def k_zero_grow():
                for i, j, k in w3:
                    w3[i, j, k] = 0
                for i, j, k in w4:
                    w4[i, j, k] = 0
                for i, j in h:
                    h[i, j] = 0

            @ti.kernel
            def k_copy_w2_w4():
                for i, j, k in w4:
                    w4[i, j, k] = w2[i, j, k]

            @ti.kernel
            def k_grow_nxt():
                """w5 = dilate6(w4) & react & ~consume & ~grown."""
                for i, j, k in w5:
                    hit = 0
                    if i > 0 and w4[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w4[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w4[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w4[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w4[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w4[i, j, k + 1] != 0:
                        hit = 1
                    if hit != 0 and w1[i, j, k] != 0 and w2[i, j, k] == 0 \
                            and w3[i, j, k] == 0:
                        w5[i, j, k] = 1
                    else:
                        w5[i, j, k] = 0

            @ti.kernel
            def k_grow_col():
                """Par colonne : rang croissant en z, budget fixe pendant le
                tour — keep = nxt & (rang + cnt <= b) ; puis cnt += keep."""
                for i, j in h:
                    c0 = h[i, j]
                    b = h2[i, j]
                    r = 0
                    kept = 0
                    for k in range(nz):
                        if w5[i, j, k] != 0:
                            r += 1
                            if r + c0 <= b:
                                w3[i, j, k] = 1
                                kept += 1
                    h[i, j] = c0 + kept

            @ti.kernel
            def k_front():
                for i, j, k in w4:
                    w4[i, j, k] = ti.cast(w3[i, j, k] != 0 or w2[i, j, k] != 0,
                                          ti.u8)

            # --- epitaxy seed (BFS géodésique L1 à travers l'air) -----------
            @ti.kernel
            def k_seed_air(m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j, k in w1:
                    w1[i, j, k] = ti.cast(d[i, j, k] == 0 and m2[i, j] != 0,
                                          ti.u8)

            @ti.kernel
            def k_eq_w2(mid: ti.u8):
                for i, j, k in w2:
                    w2[i, j, k] = ti.cast(d[i, j, k] == mid, ti.u8)

            @ti.kernel
            def k_dil6_and_w2_w1_w3():
                for i, j, k in w3:
                    hit = 0
                    if i > 0 and w2[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w2[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w2[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w2[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w2[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w2[i, j, k + 1] != 0:
                        hit = 1
                    w3[i, j, k] = ti.cast(hit != 0 and w1[i, j, k] != 0, ti.u8)

            @ti.kernel
            def k_or_w4_w3():
                for i, j, k in w4:
                    if w3[i, j, k] != 0:
                        w4[i, j, k] = 1

            @ti.kernel
            def k_dil6_and_not_w4_w1_w3():
                """w3 = dilate6(w4) & air & ~reach."""
                for i, j, k in w3:
                    hit = 0
                    if i > 0 and w4[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w4[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w4[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w4[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w4[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w4[i, j, k + 1] != 0:
                        hit = 1
                    w3[i, j, k] = ti.cast(
                        hit != 0 and w1[i, j, k] != 0 and w4[i, j, k] == 0,
                        ti.u8)

            @ti.kernel
            def k_seed_pass_ab():
                """Une passe BFS seed en UN noyau : w3 = w4 | (dil6(w4) & air
                & ~w4) — fusion de or_w4_w3 + dil6_and_not_w4_w1_w3 (x2 moins
                de lancements, bit-identique)."""
                for i, j, k in w3:
                    old = w4[i, j, k]
                    hit = 0
                    if i > 0 and w4[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w4[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w4[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w4[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w4[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w4[i, j, k + 1] != 0:
                        hit = 1
                    nw = old
                    if hit != 0 and w1[i, j, k] != 0 and old == 0:
                        nw = 1
                    w3[i, j, k] = ti.cast(nw, ti.u8)

            @ti.kernel
            def k_seed_pass_ba():
                """Passe retour : w4 = w3 | (dil6(w3) & air & ~w3)."""
                for i, j, k in w4:
                    old = w3[i, j, k]
                    hit = 0
                    if i > 0 and w3[i - 1, j, k] != 0:
                        hit = 1
                    if i < nx - 1 and w3[i + 1, j, k] != 0:
                        hit = 1
                    if j > 0 and w3[i, j - 1, k] != 0:
                        hit = 1
                    if j < ny - 1 and w3[i, j + 1, k] != 0:
                        hit = 1
                    if k > 0 and w3[i, j, k - 1] != 0:
                        hit = 1
                    if k < nz - 1 and w3[i, j, k + 1] != 0:
                        hit = 1
                    nw = old
                    if hit != 0 and w1[i, j, k] != 0 and old == 0:
                        nw = 1
                    w4[i, j, k] = ti.cast(nw, ti.u8)

            @ti.kernel
            def k_copy_w3_w4():
                for i, j, k in w4:
                    w4[i, j, k] = w3[i, j, k]

            # --- surface exposée + bords (epitaxy crystal / etch_crystal) ---
            @ti.kernel
            def k_exposed(mid: ti.u8):
                """Passe 1 : ztop, surface exposée (s2), zsr f64 (a)."""
                for i, j in h:
                    zt = -1
                    for kk in range(nz):
                        k = nz - 1 - kk
                        if d[i, j, k] == mid:
                            zt = k
                            break
                    h[i, j] = zt
                    zc = zt + 1
                    if zc > nz - 1:
                        zc = nz - 1
                    sf = 0
                    if zt >= 0 and d[i, j, zc] == Materials.AIR:
                        sf = 1
                    s2[i, j] = ti.cast(sf, ti.u8)
                    if sf:
                        a[i, j] = ti.f64(zt)
                    else:
                        a[i, j] = INF

            @ti.kernel
            def k_edges():
                """Passe 2 : bords en x/y depuis s2/a (noyau séparé : pas de
                course entre threads sur les champs 2D)."""
                for i, j in ex2:
                    sf = s2[i, j] != 0
                    badx = 0
                    if sf:
                        if i > 0:
                            if s2[i - 1, j] == 0:
                                badx = 1
                            else:
                                dd = a[i - 1, j] - a[i, j]
                                if dd < 0.0:
                                    dd = -dd
                                if dd >= 0.5:
                                    badx = 1
                        if i < nx - 1 and badx == 0:
                            if s2[i + 1, j] == 0:
                                badx = 1
                            else:
                                dd = a[i + 1, j] - a[i, j]
                                if dd < 0.0:
                                    dd = -dd
                                if dd >= 0.5:
                                    badx = 1
                    ex2[i, j] = ti.cast(badx, ti.u8)
                    bady = 0
                    if sf:
                        if j > 0:
                            if s2[i, j - 1] == 0:
                                bady = 1
                            else:
                                dd = a[i, j - 1] - a[i, j]
                                if dd < 0.0:
                                    dd = -dd
                                if dd >= 0.5:
                                    bady = 1
                        if j < ny - 1 and bady == 0:
                            if s2[i, j + 1] == 0:
                                bady = 1
                            else:
                                dd = a[i, j + 1] - a[i, j]
                                if dd < 0.0:
                                    dd = -dd
                                if dd >= 0.5:
                                    bady = 1
                    ey2[i, j] = ti.cast(bady, ti.u8)

            # --- plans de facettes : min-pyramide séparable ------------------
            @ti.kernel
            def k_init_ac_inf():
                for i, j in ac:
                    ac[i, j] = INF

            @ti.kernel
            def k_par_into(val: ti.f64):
                """Plan parallèle au wafer : ac = min(ac, surf ? zseed+val)."""
                for i, j in ac:
                    if s2[i, j] != 0:
                        c = ti.f64(h[i, j]) + val
                        if c < ac[i, j]:
                            ac[i, j] = c

            @ti.kernel
            def k_anch(sx: ti.f64, sy: ti.f64, val: ti.f64):
                """ab = ancres (epitaxy : hauteur = zseed + val) : surf & bords
                perpendiculaires à la montée."""
                for i, j in ab:
                    ok = s2[i, j] != 0
                    if sx > 0.0 and ex2[i, j] == 0:
                        ok = False
                    if sy > 0.0 and ey2[i, j] == 0:
                        ok = False
                    if ok:
                        ab[i, j] = ti.f64(h[i, j]) + val
                    else:
                        ab[i, j] = INF

            @ti.kernel
            def k_par_etch(val: ti.f64):
                """Plan parallèle au wafer (etch_crystal : profondeur = val,
                sans zseed) : ac = min(ac, surf ? val : inf)."""
                for i, j in ac:
                    if s2[i, j] != 0:
                        if val < ac[i, j]:
                            ac[i, j] = val

            @ti.kernel
            def k_anch_etch(sx: ti.f64, sy: ti.f64, val: ti.f64):
                """ab = ancres (etch_crystal : profondeur = val, sans zseed)."""
                for i, j in ab:
                    ok = s2[i, j] != 0
                    if sx > 0.0 and ex2[i, j] == 0:
                        ok = False
                    if sy > 0.0 and ey2[i, j] == 0:
                        ok = False
                    if ok:
                        ab[i, j] = val
                    else:
                        ab[i, j] = INF

            @ti.kernel
            def k_minscan_x(s: ti.f64):
                """a = min_j ab[j] + s*|i-j| le long de x (2 passages, même
                ordre d'opérations f64 que np.minimum.accumulate)."""
                for j in range(ny):
                    acc = INF
                    for i in range(nx):
                        t = ab[i, j] - s * ti.f64(i)
                        if t < acc:
                            acc = t
                        a[i, j] = acc + s * ti.f64(i)
                    acc = INF
                    for ii in range(nx):
                        i = nx - 1 - ii
                        t = ab[i, j] - s * ti.f64(nx - 1 - i)
                        if t < acc:
                            acc = t
                        b = acc + s * ti.f64(nx - 1 - i)
                        if b < a[i, j]:
                            a[i, j] = b

            @ti.kernel
            def k_minscan_y(s: ti.f64):
                """ad = min_j a[i, j] + s*|j-j'| le long de y."""
                for i in range(nx):
                    acc = INF
                    for j in range(ny):
                        t = a[i, j] - s * ti.f64(j)
                        if t < acc:
                            acc = t
                        ad[i, j] = acc + s * ti.f64(j)
                    acc = INF
                    for jj in range(ny):
                        j = ny - 1 - jj
                        t = a[i, j] - s * ti.f64(ny - 1 - j)
                        if t < acc:
                            acc = t
                        b = acc + s * ti.f64(ny - 1 - j)
                        if b < ad[i, j]:
                            ad[i, j] = b

            @ti.kernel
            def k_min_into_ac():
                for i, j in ac:
                    if ad[i, j] < ac[i, j]:
                        ac[i, j] = ad[i, j]

            @ti.kernel
            def k_surf_only_ac():
                for i, j in ac:
                    if s2[i, j] == 0:
                        ac[i, j] = INF

            @ti.kernel
            def k_zfill():
                """h2 = rint(min(ac, nz-1)) si fini, sinon -1 (epitaxy)."""
                for i, j in h2:
                    v = ac[i, j]
                    if v < INF:
                        m = v
                        if m > ti.cast(nz - 1, ti.f64):
                            m = ti.cast(nz - 1, ti.f64)
                        h2[i, j] = _rint(m)
                    else:
                        h2[i, j] = -1

            @ti.kernel
            def k_nmap():
                """h2 = surf ? rint(min(ac, nz)) : 0 (etch_crystal)."""
                for i, j in h2:
                    if s2[i, j] != 0:
                        m = ac[i, j]
                        if m > ti.cast(nz, ti.f64):
                            m = ti.cast(nz, ti.f64)
                        h2[i, j] = _rint(m)
                    else:
                        h2[i, j] = 0

            @ti.kernel
            def k_epi_fill(mat: ti.u8, m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j in h:
                    zs = h[i, j]
                    if zs >= 0:
                        zblock = nz
                        for k in range(zs + 1, nz):
                            if d[i, j, k] != Materials.AIR:
                                zblock = k
                                break
                        zf = h2[i, j]
                        k2 = zf
                        if k2 > zblock - 1:
                            k2 = zblock - 1
                        for k in range(zs + 1, k2 + 1):
                            if d[i, j, k] == Materials.AIR and m2[i, j] != 0:
                                d[i, j, k] = mat

            @ti.kernel
            def k_etch_crystal_fill(mat: ti.u8,
                                    m2: ti.types.ndarray(ti.u8, ndim=2)):
                for i, j in h:
                    zt = h[i, j]
                    if zt >= 0:
                        n = h2[i, j]
                        zstop = -1
                        for k in range(zt + 1):
                            if d[i, j, k] != mat:
                                zstop = k
                        zlo = zstop
                        if zt - n > zlo:
                            zlo = zt - n
                        for k in range(zlo + 1, zt + 1):
                            if d[i, j, k] == mat and m2[i, j] != 0:
                                d[i, j, k] = Materials.AIR

            # --- fill (tranchées) : inondation depuis la face z = nz-1 ------
            # --- fill par union-find (issue #3) : O(N), passes ~log --------
            @ti.kernel
            def k_uf_init():
                for i, j, k in lbl:
                    if d[i, j, k] == 0:
                        lbl[i, j, k] = i + nx * (j + ny * k)
                    else:
                        lbl[i, j, k] = -1

            @ti.kernel
            def k_uf_topflag_clear():
                for v in topflag:
                    topflag[v] = 0

            @ti.kernel
            def k_uf_mark_top():
                for i, j in h:
                    l = lbl[i, j, nz - 1]
                    if l >= 0:
                        topflag[l] = 1

            @ti.kernel
            def k_uf_pass():
                """Union par atomic_min : propage le min de label aux 6
                voisins ; flag si un label a baissé (convergence)."""
                for i, j, k in lbl:
                    l = lbl[i, j, k]
                    if l >= 0:
                        m = l
                        if i > 0 and lbl[i - 1, j, k] >= 0 and lbl[i - 1, j, k] < m:
                            m = lbl[i - 1, j, k]
                        if i < nx - 1 and lbl[i + 1, j, k] >= 0 and lbl[i + 1, j, k] < m:
                            m = lbl[i + 1, j, k]
                        if j > 0 and lbl[i, j - 1, k] >= 0 and lbl[i, j - 1, k] < m:
                            m = lbl[i, j - 1, k]
                        if j < ny - 1 and lbl[i, j + 1, k] >= 0 and lbl[i, j + 1, k] < m:
                            m = lbl[i, j + 1, k]
                        if k > 0 and lbl[i, j, k - 1] >= 0 and lbl[i, j, k - 1] < m:
                            m = lbl[i, j, k - 1]
                        if k < nz - 1 and lbl[i, j, k + 1] >= 0 and lbl[i, j, k + 1] < m:
                            m = lbl[i, j, k + 1]
                        if m < l:
                            ti.atomic_or(flag[None], 1)
                            ti.atomic_min(lbl[i, j, k], m)
                        if i < nx - 1 and lbl[i + 1, j, k] >= 0:
                            ti.atomic_min(lbl[i + 1, j, k], m)
                        if j < ny - 1 and lbl[i, j + 1, k] >= 0:
                            ti.atomic_min(lbl[i, j + 1, k], m)
                        if k < nz - 1 and lbl[i, j, k + 1] >= 0:
                            ti.atomic_min(lbl[i, j, k + 1], m)

            @ti.kernel
            def k_uf_compress():
                """Path compression : lbl[v] = lbl[lbl[v]] (accélère la
                convergence ; ne change pas le min final de la composante,
                donc bit-identique)."""
                for i, j, k in lbl:
                    l = lbl[i, j, k]
                    if l >= 0:
                        li = l % nx
                        lj = (l // nx) % ny
                        lk = l // (nx * ny)
                        r = lbl[li, lj, lk]
                        if r >= 0 and r < l:
                            lbl[i, j, k] = r

            @ti.kernel
            def k_fill_uf_set(mat: ti.u8, max_z: ti.i32):
                for i, j, k in d:
                    l = lbl[i, j, k]
                    if l >= 0 and k <= max_z and topflag[l] != 0:
                        d[i, j, k] = mat

            @ti.kernel
            def k_clear_flag():
                flag[None] = 0

            # --- CMP ----------------------------------------------------------
            @ti.kernel
            def k_cmp(target: ti.i32, stop: ti.i32):
                for i, j in h:
                    has = 0
                    zstop = -1
                    if stop >= 0:
                        for kk in range(nz):
                            k = nz - 1 - kk
                            if d[i, j, k] == stop:
                                zstop = k
                                has = 1
                                break
                    for k in range(nz):
                        if d[i, j, k] != 0 and k > target:
                            if has == 0 or k > zstop:
                                d[i, j, k] = Materials.AIR
                                ti.atomic_add(flag[None], 1)

            # --- registre des noyaux -----------------------------------------
            self.k = {
                "substrate": k_substrate,
                "solid_w1": k_solid_w1,
                "ball12": k_ball12,
                "cd12": k_cd12,
                "cd21": k_cd21,
                "ce12": k_ce12,
                "ce21": k_ce21,
                "copy12": k_copy12,
                "copy15": k_copy15,
                "copy51": k_copy51,
                "conformal_set": k_conformal_set,
                "height": k_height,
                "dep_dir": k_dep_dir,
                "etch_iso_final": k_etch_iso_final,
                "etch_dir": k_etch_dir,
                "etch_sel_dir": k_etch_sel_dir,
                "sel_iso_keep_init": k_sel_iso_keep_init,
                "or_and_w3": k_or_and_w3,
                "sel_iso_final": k_sel_iso_final,
                "grow_react": k_grow_react,
                "grow_consume": k_grow_consume,
                "set_w2": k_set_w2,
                "set_w3": k_set_w3,
                "set_w4": k_set_w4,
                "col_count": k_col_count,
                "col_budget": k_col_budget,
                "maxf_x": k_maxf_x,
                "maxf_y": k_maxf_y,
                "zero_grow": k_zero_grow,
                "copy_w2_w4": k_copy_w2_w4,
                "grow_nxt": k_grow_nxt,
                "grow_col": k_grow_col,
                "front": k_front,
                "seed_air": k_seed_air,
                "eq_w2": k_eq_w2,
                "dil6_and_w2_w1_w3": k_dil6_and_w2_w1_w3,
                "or_w4_w3": k_or_w4_w3,
                "dil6_and_not_w4_w1_w3": k_dil6_and_not_w4_w1_w3,
                "seed_pass_ab": k_seed_pass_ab,
                "seed_pass_ba": k_seed_pass_ba,
                "copy_w3_w4": k_copy_w3_w4,
                "exposed": k_exposed,
                "edges": k_edges,
                "init_ac_inf": k_init_ac_inf,
                "par_into": k_par_into,
                "par_etch": k_par_etch,
                "anch": k_anch,
                "anch_etch": k_anch_etch,
                "minscan_x": k_minscan_x,
                "minscan_y": k_minscan_y,
                "min_into_ac": k_min_into_ac,
                "surf_only_ac": k_surf_only_ac,
                "zfill": k_zfill,
                "nmap": k_nmap,
                "epi_fill": k_epi_fill,
                "etch_crystal_fill": k_etch_crystal_fill,
                "clear_flag": k_clear_flag,
                "uf_init": k_uf_init,
                "uf_topflag_clear": k_uf_topflag_clear,
                "uf_mark_top": k_uf_mark_top,
                "uf_pass": k_uf_pass,
                "uf_compress": k_uf_compress,
                "fill_uf_set": k_fill_uf_set,
                "cmp": k_cmp,
            }

    _CTX_CACHE: dict = {}
    _ALLOCATED = 0          # octets VRAM alloués (comptés côté hôte)
    _BUDGET = None          # budget VRAM (calculé une fois)
    _FH_CACHE: dict = {}    # (nx, ny) -> hauteur de champ zslab (fixée une fois)
    _TILE_CACHE: dict = {}  # nz -> (tx, ty) taille de tuile xy (fixée une fois)

    # --- budget VRAM : tiling au-delà (cf. ARCHITECTURE.md §5) --------------

    def _vram_free_bytes():
        """VRAM libre via cuInit + cuMemGetInfo (nvcuda.dll / libcuda.so)."""
        import ctypes
        for lib in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
            try:
                cuda = ctypes.CDLL(lib)
                cuda.cuInit(0)
                free = ctypes.c_size_t()
                total = ctypes.c_size_t()
                if cuda.cuMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
                    return free.value
            except Exception:
                continue
        return None

    def _budget() -> int:
        """Budget VRAM utilisable : env VOXELFAB_GPU_VRAM_BYTES, sinon 85 %
        de la VRAM libre mesurée, sinon 4 Go par défaut."""
        global _BUDGET
        if _BUDGET is None:
            env = os.environ.get("VOXELFAB_GPU_VRAM_BYTES")
            if env:
                _BUDGET = int(env)
            else:
                free = _vram_free_bytes()
                _BUDGET = int(free * 0.85) if free else 4 * 1024 ** 3
        return _BUDGET

    def _ctx_bytes(shape) -> int:
        """Octets VRAM d'un _GpuCtx : 6 tampons u8 3D + lbl i32 3D +
        topflag u8 1D (union-find) + champs 2D (~51 o/col)."""
        nx, ny, nz = shape
        return 11 * nx * ny * nz + 51 * nx * ny

    def _fits(shape) -> bool:
        """True si un ctx complet de cette shape tient dans le budget VRAM.
        VOXELFAB_FORCE_TILED=1 force le mode tuilé (tests)."""
        if os.environ.get("VOXELFAB_FORCE_TILED"):
            return False
        return _ALLOCATED + _ctx_bytes(shape) <= _budget()

    def _ctx(shape) -> _GpuCtx:
        global _ALLOCATED
        ctx = _CTX_CACHE.get(shape)
        if ctx is None:
            _ensure_init()
            _ALLOCATED += _ctx_bytes(shape)
            ctx = _GpuCtx(shape)
            _CTX_CACHE[shape] = ctx
        return ctx

    class _View:
        """Vue tuile d'une grille : seules les dims sont lues par les ops."""
        def __init__(self, nx, ny, nz):
            self.nx, self.ny, self.nz = nx, ny, nz
            self.data = None
            self._gpu_resident = False

    def _tile(mode, halo=None):
        """Marque une étape comme tuilable : mode 'zslab' (stencil, halo exact
        = rayon) ou 'xy' (colonnes indépendantes, pas de halo, z complet).
        halo(*args) -> rayon du stencil pour les ops zslab."""
        def deco(w):
            w._tile = mode
            w._halo = halo
            return w
        return deco

    def _run_tiled(fn, wrapper, grid, args, kwargs, mode):
        """Exécution tuilée d'une étape, la grille restant côté hôte (grid.data
        = backing store). Halo exact pour les stencils : un voxel à distance
        > r hors de la tuile ne peut pas influencer l'intérieur."""
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        if mode == "zslab":
            r = wrapper._halo(*args, **kwargs)
            # hors grille : border_value du CPU (dilate=0, erode=1=solide)
            pad = 1 if fn.__name__ in ("etch_isotropic",
                                       "etch_selective_isotropic") else 0
            # hauteur de champ calculée UNE FOIS par (nx, ny) : sinon chaque op
            # aurait sa propre shape de ctx -> recompile de tous les noyaux
            # (les noyaux en closure sont compilés par instance de ctx).
            if (nx, ny) not in _FH_CACHE:
                per_z = 6 * nx * ny + 51 * nx * ny
                _FH_CACHE[(nx, ny)] = min(nz, max(1, int(_budget() // per_z)))
            fh = _FH_CACHE[(nx, ny)]
            if 2 * r + 1 > fh:                  # halo seul dépasse : shape dédiée
                fh = min(nz, 2 * r + 1)
            h = max(1, fh - 2 * r)
            sctx = _ctx((nx, ny, fh))
            # snapshot d'entrée : les halos des tuiles doivent être lus dans
            # l'état AVANT l'étape (sinon les tuiles déjà écrites contaminent
            # le halo des suivantes -> double dépôt/érosion aux frontières)
            src = grid.data.copy() if r > 0 else grid.data
            for z0 in range(0, nz, h):
                z1 = min(z0 + h, nz)
                buf = np.full((nx, ny, fh), pad, np.uint8)
                lo, hi = max(0, z0 - r), min(nz, z1 + r)
                buf[:, :, r + lo - z0: r + hi - z0] = src[:, :, lo:hi]
                sctx.d.from_numpy(buf)
                a = args
                if fn.__name__ == "substrate" and len(args) >= 2:
                    a = (args[0], args[1] - z0 + r)   # seuil global -> local
                    if a[1] <= 0:
                        continue
                fn(sctx, _View(nx, ny, z1 - z0), *a, **kwargs)
                out = sctx.d.to_numpy()
                grid.data[:, :, z0:z1] = out[:, :, r: r + (z1 - z0)]
            return None
        # mode "xy" : tuiles (x, y) à z complet — colonnes indépendantes
        # taille de tuile calculée UNE FOIS par nz (même raison que zslab)
        if nz not in _TILE_CACHE:
            per_col = 6 * nz + 51
            t = max(1, int((_budget() / per_col) ** 0.5))
            _TILE_CACHE[nz] = (min(nx, t), min(ny, t))
        tx, ty = _TILE_CACHE[nz]
        tctx = _ctx((tx, ty, nz))
        total = 0
        for x0 in range(0, nx, tx):
            for y0 in range(0, ny, ty):
                x1, y1 = min(x0 + tx, nx), min(y0 + ty, ny)
                w, h = x1 - x0, y1 - y0
                buf = np.zeros((tx, ty, nz), np.uint8)
                buf[:w, :h, :] = grid.data[x0:x1, y0:y1, :]
                tctx.d.from_numpy(buf)
                kw = dict(kwargs)
                m = kw.get("mask")
                if m is not None:
                    mt = np.zeros((tx, ty), bool)
                    mt[:w, :h] = m[x0:x1, y0:y1]
                    kw["mask"] = mt
                r = fn(tctx, _View(tx, ty, nz), *args, **kw)
                if fn.__name__ == "cmp":
                    total += r
                out = tctx.d.to_numpy()
                grid.data[x0:x1, y0:y1, :] = out[:w, :h, :]
        return total if fn.__name__ == "cmp" else None

    def _morph(ctx, r: int, erode: bool):
        """w2 = morphisme de w1 par la boule de rayon r, exactement comme le
        CPU (w1 est préservé) : r <= 4 -> élément structurant boule exacte
        (un noyau) ; r > 4 -> r itérations de la croix 6-voisine
        (ping-pong w1 <-> w2, w1 sauvegardé dans w5)."""
        if r <= 4:
            off = _ball_offsets(r)
            ctx.k["ball12"](off, off.shape[0], 1 if erode else 0)
            return
        ctx.k["copy15"]()                       # sauvegarde de w1
        for i in range(r):
            if erode:
                ctx.k["ce12"]() if i % 2 == 0 else ctx.k["ce21"]()
            else:
                ctx.k["cd12"]() if i % 2 == 0 else ctx.k["cd21"]()
        if r % 2 == 0:
            ctx.k["copy12"]()                   # résultat (dans w1) -> w2
        ctx.k["copy51"]()                       # restaure w1

    def _gpu_op(fn):
        """Décorateur : dispatch selon la taille de grille.

        - grille qui tient en VRAM : ctx complet. En mode device-résident
          (bind_resident pose grid._gpu_resident), les syncs hôte↔device sont
          sautés entre les étapes (un seul D2H final via release_resident) ;
          sinon sync par étape. Dans les deux cas ti.sync()/to_numpy assurent
          la fin réelle de l'étape (lancements Taichi asynchrones).
        - grille trop grosse (hors budget VRAM) : mode tuilé avec halos exacts
          (_run_tiled), grid.data reste le backing store côté hôte. Les étapes
          non tuilables (propagation globale : grow, epitaxy seed/crystal,
          etch_crystal, fill) lèvent une erreur claire."""
        def wrapper(grid, *args, **kwargs):
            shape = (grid.nx, grid.ny, grid.nz)
            resident = getattr(grid, "_gpu_resident", False)
            if resident or _fits(shape):
                ctx = _ctx(shape)
                if not resident:
                    ctx.d.from_numpy(grid.data)
                r = fn(ctx, grid, *args, **kwargs)
                if resident:
                    ti.sync()   # lancements Taichi asynchrones : on attend la
                                # fin réelle de l'étape (chrono honnête)
                else:
                    grid.data = ctx.d.to_numpy()
                return r
            mode = getattr(wrapper, "_tile", None)
            if mode is None:
                raise RuntimeError(
                    f"op {fn.__name__!r} non tuilable : la grille {shape} = "
                    f"{shape[0] * shape[1] * shape[2] / 1e6:.0f} M vx dépasse "
                    f"le budget VRAM (~{_budget() / 2**30:.1f} Go) et cette op "
                    "a besoin de la grille entière sur le device (propagation "
                    "globale).")
            return _run_tiled(fn, wrapper, grid, args, kwargs, mode)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    def bind_resident(grid):
        """Mode device-résident si la grille tient en VRAM : copie grid.data
        sur le device UNE fois ; les étapes chaînent sur le device (supprime
        ~2×N octets de PCIe par étape). Sinon : mode tuilé (grid.data reste
        côté hôte, chaque étape traite des tuiles avec halos exacts)."""
        if not HAS_TAICHI:
            raise RuntimeError("bind_resident : taichi non installé "
                               "(extra [gpu])")
        shape = (grid.nx, grid.ny, grid.nz)
        if _fits(shape):
            ctx = _ctx(shape)
            ctx.d.from_numpy(grid.data)
            grid._gpu_resident = True
        else:
            grid._gpu_resident = False
            grid._gpu_tiled = True

    def release_resident(grid):
        """Termine le mode device-résident : unique D2H, grid.data à jour
        (à appeler avant toute lecture hôte : exports, métrologie, vues).
        En mode tuilé, grid.data est déjà à jour côté hôte : no-op."""
        if getattr(grid, "_gpu_resident", False):
            ctx = _ctx((grid.nx, grid.ny, grid.nz))
            grid.data = ctx.d.to_numpy()
            grid._gpu_resident = False

    def sync_to_host(grid):
        """D2H temporaire SANS quitter le mode device-résident (checkpoints
        entre étapes) : grid.data est rapatrié depuis le device, mais
        _gpu_resident reste True — les étapes suivantes continuent sur le
        device. En mode tuilé / non-résident : no-op (grid.data déjà hôte)."""

        if getattr(grid, "_gpu_resident", False):
            ctx = _ctx((grid.nx, grid.ny, grid.nz))
            ti.sync()
            grid.data = ctx.d.to_numpy()

    # ----------------------------------------------------------------------
    # Étapes enregistrées (unités : VOXELS, mêmes signatures que cpu.py)
    # ----------------------------------------------------------------------

    @register("gpu", "substrate")
    @_tile("zslab", lambda material, t: 0)
    @_gpu_op
    def substrate(ctx, grid, material, t):
        ctx.k["substrate"](t, material)

    @register("gpu", "deposit", "conformal")
    @_tile("zslab", lambda material, t, on_materials=None: t)
    @_gpu_op
    def deposit_conformal(ctx, grid, material, t, on_materials=None):
        if t == 0:
            return
        ctx.k["solid_w1"]()
        _morph(ctx, t, False)
        ctx.k["conformal_set"](material)

    @register("gpu", "deposit", "directional")
    @_tile("xy")
    @_gpu_op
    def deposit_directional(ctx, grid, material, t, on_materials=None):
        if t == 0:
            return
        ctx.k["height"]()
        lut = _lut_u8(_allowed_lut(on_materials))
        ctx.k["dep_dir"](t, material, lut)

    @register("gpu", "etch", "isotropic")
    @_tile("zslab",
           lambda t, materials=None, until_material=None, mask=None: t)
    @_gpu_op
    def etch_isotropic(ctx, grid, t, materials=None, until_material=None,
                       mask=None):
        if t == 0:
            return
        ctx.k["solid_w1"]()
        _morph(ctx, t, True)
        ctx.k["etch_iso_final"](_lut_u8(_allowed_lut(materials)),
                                _m2(mask, grid.nx, grid.ny))

    @register("gpu", "etch", "directional")
    @_tile("xy")
    @_gpu_op
    def etch_directional(ctx, grid, t, materials=None, until_material=None,
                         mask=None):
        if t == 0:
            return
        open_mask = _m2(mask, grid.nx, grid.ny)
        allowed = _lut_u8(_allowed_lut(materials))
        stop = -1 if until_material is None else until_material
        ctx.k["etch_dir"](t, stop, open_mask, allowed)

    @register("gpu", "etch_selective", "directional")
    @_tile("xy")
    @_gpu_op
    def etch_selective_directional(ctx, grid, depth, rates,
                                   until_material=None, mask=None):
        if depth <= 0 or not rates:
            return
        open_mask = _m2(mask, grid.nx, grid.ny)
        stop = -1 if until_material is None else until_material
        ctx.k["etch_sel_dir"](float(depth), stop, open_mask,
                              np.ascontiguousarray(_rates_lut(rates)))

    @register("gpu", "etch_selective", "isotropic")
    @_tile("zslab",
           lambda depth, rates, until_material=None, mask=None:
           max(int(round(depth * float(v))) for v in rates.values()))
    @_gpu_op
    def etch_selective_isotropic(ctx, grid, depth, rates,
                                 until_material=None, mask=None):
        if depth <= 0 or not rates:
            return
        ctx.k["solid_w1"]()
        lut_not = np.ones(256, dtype=bool)
        lut_not[np.fromiter(rates.keys(), dtype=np.int64,
                            count=len(rates))] = False
        ctx.k["sel_iso_keep_init"](_lut_u8(lut_not))
        for mid, rate in rates.items():
            r = int(round(depth * float(rate)))
            if r <= 0:
                # keep |= (d == mid) : aucun noyau, on marque directement
                ctx.k["eq_w2"](mid)
                ctx.k["or_and_w3"](mid)      # w3 |= w2 & (d==mid) ; w2 = eq
                continue
            _morph(ctx, r, True)                     # w2 = erode(solid, r)
            ctx.k["or_and_w3"](mid)                  # w3 |= w2 & (d==mid)
        ctx.k["sel_iso_final"](_m2(mask, grid.nx, grid.ny))

    @register("gpu", "grow")
    @_gpu_op
    def grow(ctx, grid, depth, from_id, product_id, expansion=1.0, into=None,
             mask=None):
        if depth <= 0 or expansion <= 0:
            return
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        m2 = _m2(mask, nx, ny)
        into_lut = np.zeros(256, dtype=np.uint8)
        if into:
            into_lut[np.fromiter(into, dtype=np.int64, count=len(into))] = 1
        ctx.k["grow_react"](into_lut, m2)                 # w1 = réactif
        _morph(ctx, depth, False)                         # w2 = dilate(w1)
        ctx.k["grow_consume"](from_id, m2)                # w2 = consommation
        ctx.k["set_w2"](product_id)                       # data[consume] = prod
        d2 = int(round(depth * (float(expansion) - 1.0)))
        if d2 <= 0:
            return
        ctx.k["col_count"]()                              # h = conso/colonne
        ctx.k["col_budget"](float(expansion) - 1.0)       # h2 = rint(h*s)
        ctx.k["maxf_x"](d2)                               # filtre max en x
        ctx.k["maxf_y"](d2)                               # filtre max en y
        # h2 = budget filtré (maximum_filter size=2*d2+1, séparable, 'reflect')
        ctx.k["zero_grow"]()                              # w3=0, w4=0, h(cnt)=0
        ctx.k["copy_w2_w4"]()                             # front = consume
        for _ in range(d2):
            ctx.k["grow_nxt"]()                           # w5
            ctx.k["grow_col"]()                           # w3, h vs h2
            ctx.k["front"]()                              # w4 = w3 | w2
        ctx.k["set_w3"](product_id)                       # data[grown] = prod

    @register("gpu", "epitaxy", "seed")
    @_gpu_op
    def epitaxy_seed(ctx, grid, depth, material, seed, mask=None):
        if depth <= 0:
            return
        m2 = _m2(mask, grid.nx, grid.ny)
        ctx.k["zero_grow"]()                              # w3=0, w4(reach)=0
        ctx.k["seed_air"](m2)                             # w1 = air & masque
        ctx.k["eq_w2"](seed)                              # w2 = graine
        ctx.k["dil6_and_w2_w1_w3"]()                      # w3 = front dist 1
        ctx.k["copy_w3_w4"]()                             # w4 = reach (dist <= 1)
        # passes BFS en UN noyau ping-pong (x2 moins de lancements,
        # bit-identique) : après p passes, reach = dist <= p+1
        depth = int(depth)
        for p in range(depth - 1):
            ctx.k["seed_pass_ab" if p % 2 == 0 else "seed_pass_ba"]()
        if depth > 1 and depth % 2 == 0:
            ctx.k["copy_w3_w4"]()                         # reach final -> w4
        ctx.k["set_w4"](material)

    @register("gpu", "epitaxy", "crystal")
    @_gpu_op
    def epitaxy_crystal(ctx, grid, depth, material, orientation="100",
                        facet_rates=None, seed=None, mask=None):
        if depth <= 0:
            return
        t = float(depth)
        rates = facet_rates or {"100": 1.0, "110": 0.6, "111": 0.25}
        planes = _crystal_planes(orientation, rates)
        ctx.k["exposed"](seed)
        ctx.k["edges"]()
        ctx.k["init_ac_inf"]()
        for sx, sy, inv_nz, v in planes:
            val = v * t * inv_nz
            if sx == 0.0 and sy == 0.0:
                ctx.k["par_into"](val)
            else:
                ctx.k["anch"](sx, sy, val)
                ctx.k["minscan_x"](sx)
                ctx.k["minscan_y"](sy)
                ctx.k["min_into_ac"]()
        ctx.k["surf_only_ac"]()
        ctx.k["zfill"]()
        ctx.k["epi_fill"](material, _m2(mask, grid.nx, grid.ny))

    @register("gpu", "etch_crystal")
    @_gpu_op
    def etch_crystal(ctx, grid, depth, material, orientation="100",
                     facet_rates=None, mask=None):
        if depth <= 0:
            return
        t = float(depth)
        rates = facet_rates or {"100": 1.0, "110": 1.6, "111": 0.1}
        planes = _crystal_planes(orientation, rates)
        vpar = max(v for sx, sy, _, v in planes if sx == 0.0 and sy == 0.0)
        planes = [(sx, sy, inv_nz, v / vpar) for sx, sy, inv_nz, v in planes]
        ctx.k["exposed"](material)
        ctx.k["edges"]()
        ctx.k["init_ac_inf"]()
        for sx, sy, inv_nz, v in planes:
            val = v * t * inv_nz
            if sx == 0.0 and sy == 0.0:
                ctx.k["par_etch"](val)
            else:
                ctx.k["anch_etch"](sx, sy, val)
                ctx.k["minscan_x"](sx)
                ctx.k["minscan_y"](sy)
                ctx.k["min_into_ac"]()
        ctx.k["nmap"]()
        ctx.k["etch_crystal_fill"](material, _m2(mask, grid.nx, grid.ny))

    @register("gpu", "fill")
    @_gpu_op
    def fill_trench(ctx, grid, material, max_z):
        # union-find sur l'air (issue #3) : label final = min d'index par
        # composante 6-connexe -> bit-identique vs ndimage.label (CPU).
        # Convergence en ~log(passes) grâce à la compression de chemins.
        ctx.k["uf_init"]()
        ctx.k["uf_topflag_clear"]()
        max_passes = grid.nx + grid.ny + grid.nz      # borne sûre (diamètre)
        for p in range(max_passes):
            ctx.k["clear_flag"]()
            ctx.k["uf_pass"]()
            ctx.k["uf_compress"]()
            if (p + 1) % 8 == 0 or p == max_passes - 1:
                if int(ctx.flag.to_numpy()) == 0:
                    break
        # après convergence, lbl[v] = min de sa composante : on marque les
        # racines des composantes qui touchent le plan top
        ctx.k["uf_mark_top"]()
        ctx.k["fill_uf_set"](material, max_z)

    @register("gpu", "cmp")
    @_tile("xy")
    @_gpu_op
    def cmp(ctx, grid, target_z_voxels, stop_on=None):
        ctx.k["clear_flag"]()
        stop = -1 if stop_on is None else stop_on
        ctx.k["cmp"](target_z_voxels, stop)
        return int(ctx.flag.to_numpy())
