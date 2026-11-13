"""Métrologie virtuelle : mesures sur la grille voxel finale."""

from __future__ import annotations

from .grid import VoxelGrid


class Metrology:
    def __init__(self, grid: VoxelGrid):
        self.g = grid

    def thickness_at(self, x: int, y: int, material: int) -> int:
        """Épaisseur verticale (en voxels) d'un matériau dans une colonne."""
        return int((self.g.data[x, y, :] == material).sum())

    def top_material(self, x: int, y: int) -> str:
        """Nom du matériau au sommet de la colonne (x, y)."""
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
