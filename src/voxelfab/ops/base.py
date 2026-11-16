"""Registre des étapes de procédé + SÉMANTIQUE DE RÉFÉRENCE.

CONTRAT : toute implémentation (cpu aujourd'hui, gpu demain) doit produire
une grille voxel identique octet pour octet pour les mêmes entrées.
C'est ce que vérifient tests/test_golden.py (référence commitée) et les
tests d'équivalence inter-backends.

Signatures uniformes (les épaisseurs sont en VOXELS ici ; la conversion
nm -> voxels est faite par l'orchestrateur) :

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

    etch_selective[mode](grid, depth_voxels, rates, until_material=None,
                         mask=None)
        Gravure à sélectivité différentielle. `rates` associe un id de
        matériau à sa vitesse RELATIVE (le matériau de référence a un
        taux de 1.0 et `depth_voxels` est la profondeur gravie DANS ce
        matériau de référence). Les matériaux absents de `rates` ne sont
        jamais gravés (ils bloquent, comme un matériau d'arrêt).
        directional : même parcours par colonne que etch directional,
                      mais avec un budget de temps B = depth_voxels :
                      retirer un voxel du matériau m coûte 1/rate_m ;
                      arrêt sur until_material, sur un matériau absent
                      de rates (taux 0), ou quand le budget ne suffit
                      plus pour le voxel courant.
        isotropic   : chaque matériau m est érodé du rayon
                      round(depth_voxels * rate_m) (érosion euclidienne
                      du solide, restreinte à m). Approximation assumée :
                      les interfaces inter-matériaux comptent comme des
                      surfaces d'attaque. until_material est ignoré.

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
