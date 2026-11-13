"""Grille voxel multi-matériaux : données pures, sans dépendance backend.

Les implémentations d'étapes vivent dans voxelfab.ops.<backend> ; cette
module ne contient que la structure de données et la géométrie dérivée
(champ de hauteur, sélections de matériaux).
"""

from __future__ import annotations

import numpy as np


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


class VoxelGrid:
    """
    Grille 3D multi-matériaux, dtype uint8 (1 octet/voxel).

    Axes : grid[x, y, z], z vertical (z = nz-1 = surface).
    `pitch` = nm par voxel ; les étapes parlent en nm, les ops en voxels.
    """

    def __init__(self, nx: int, ny: int, nz: int, pitch: float = 1.0):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.pitch = pitch
        self.data = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.materials = Materials()
        # index de colonnes, alloués une seule fois et réutilisés
        self._xs = np.arange(nx)[:, None]        # (nx, 1)
        self._ys = np.arange(ny)[None, :]        # (1, ny)
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

    def copy(self) -> "VoxelGrid":
        g = VoxelGrid(self.nx, self.ny, self.nz, self.pitch)
        g.data = self.data.copy()
        g.materials = self.materials
        return g
