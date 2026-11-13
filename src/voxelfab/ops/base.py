"""Registre des étapes de procédé + SÉMANTIQUE DE RÉFÉRENCE.

CONTRAT : toute implémentation (cpu aujourd'hui, gpu demain) doit produire
une grille voxel identique octet pour octet pour les mêmes entrées.
C'est ce que vérifient tests/test_golden.py (référence commitée) et les
tests d'équivalence inter-backends.

Signatures uniformes (les épaisseurs sont en VOXELS ici ; la conversion
nm -> voxels est faite par Process) :

    substrate(grid, material, thickness_nm)
        Remplit le fond de la grille (z < épaisseur) avec `material`.

    deposit[mode](grid, material, thickness_nm, on_materials=None)
        conformal   : couche d'épaisseur constante sur toute surface
                      exposée (CVD) = voxels d'air à distance euclidienne
                      <= t du solide.
        directional : dépôt vertical au-dessus des surfaces exposées
                      (PVD, pas de recouvrement latéral) ;
                      on_materials : sélectif au matériau du sommet.

    etch[mode](grid, thickness_nm, materials=None, until_material=None,
               mask=None)
        isotropic   : retrait uniforme dans toutes les directions depuis
                      toute surface exposée à l'air (érosion euclidienne
                      de rayon t) ; sélectivité via `materials`, masque 2D
                      via `mask`.
        directional : RIE vertical ; depuis le sommet solide de chaque
                      colonne ouverte du masque, descend en sautant les
                      voxels d'air, retire jusqu'à t voxels de matériau
                      autorisé, s'arrête sur un matériau d'arrêt
                      (until_material) ou non autorisé.

    fill(grid, material, max_z)
        Remplit les volumes d'air connectés à la surface et situés sous
        z <= max_z (surface du wafer avant gravure).

    cmp(grid, target_z_voxels, stop_on=None) -> int
        Abrase tout voxel solide au-dessus du plan cible ; stop_on n'est
        jamais traversé (première occurrence depuis la surface).
        Retourne le nombre de voxels retirés.
"""

from __future__ import annotations

# backend -> {(op, mode): fonction}
REGISTRIES: dict = {}


def register(backend: str, op: str, mode: str | None = None):
    """Décore une implémentation d'étape pour un backend donné."""
    def deco(fn):
        REGISTRIES.setdefault(backend, {})[(op, mode)] = fn
        return fn
    return deco


def get_step(backend: str, op: str, mode: str | None = None):
    try:
        return REGISTRIES[backend][(op, mode)]
    except KeyError:
        raise KeyError(f"étape inconnue pour le backend {backend!r} : "
                       f"({op!r}, mode={mode!r})") from None


def available_backends() -> list:
    return sorted(REGISTRIES.keys())
