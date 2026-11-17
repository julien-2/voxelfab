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

    grow(grid, depth_voxels, from_id, product_id, expansion=1.0,
         into=None, mask=None)
        Croissance par conversion (oxydation thermique, siliciuration) :
        le matériau `from_id` réagit au contact du côté réactif (air +
        matériaux `into`) et se convertit en `product_id`.
        depth_voxels : épaisseur CONSOMMÉE de `from_id` (grandeur d'entrée
        de la réaction). expansion : épaisseur de produit produite par
        voxel consommé (Si -> SiO2 : 2.27, soit ~44 % de l'oxyde sous la
        surface d'origine et ~56 % au-dessus ; rapport de volumes,
        indépendant de l'orientation cristalline — seule la cinétique en
        dépend, et elle se règle par depth_nm à chaque étape du deck).
        - consommation : voxels de `from_id` à distance <= depth_voxels
          du côté réactif (même convention que etch isotropic : les
          matériaux inertes bloquent la réaction) ; si le matériau est
          plus mince que depth_voxels, il est consommé en entier.
        - croissance : budget proportionnel par colonne,
          b(x, y) = round(nb voxels consommés (x, y) × (expansion − 1)),
          propagé au voisinage (maximum filter) pour atteindre aussi les
          parois latérales ; le front avance par itérations de la croix
          6-voisine depuis la zone consommée, dans le côté réactif
          uniquement (il ne pousse ni dans `from_id`, ni dans les
          matériaux inertes), chaque colonne étant limitée à son budget
          (ordre déterministe : pas croissant, puis z croissant).
        - into : matériaux que le produit peut remplacer côté réactif
          (ex. le Ni d'une siliciuration) ; l'excédent de métal au-delà
          du budget reste en place — on l'enlève par gravure, comme un
          vrai dip de nettoyage post-siliciuration.
        - mask : limite la réaction aux colonnes ouvertes.

    epitaxy[seed](grid, depth_voxels, material, seed, mask=None)
        Épitaxie sélective (SEG) conforme : dépose `material` uniquement
        depuis les surfaces de la graine `seed` exposées à l'air, en
        suivant la géodésie à travers l'air (croix 6-voisine, distance
        L1 <= depth_voxels). Les matériaux inertes (oxyde, nitrure)
        bloquent la croissance : pas de dépôt au-dessus du masque, même
        en face. `mask` restreint en plus les colonnes. Croissance sur
        toutes les faces exposées (fond de fenêtre + parois de graine).
        Coins légèrement plus « pointus » que deposit conformal (L1
        géodésique vs boule exacte), même convention que les grands
        rayons des autres ops.

    epitaxy[crystal](grid, depth_voxels, material, orientation="100",
                     facet_rates=None, seed=None, mask=None)
        Épitaxie facettée : la construction SEG classique — la croissance
        verticale est limitée par des plans cristallographiques lents
        partant des bords de la graine exposée (facettes {111} à 54,7°
        du plan de wafer (100) ; toit ou diamant dans les fenêtres
        étroites, ex. SiGe épitaxié en source/drain).
        - facet_rates : {famille: taux relatif} sur '100'/'110'/'111',
          normalisé au maximum ; défaut : Si CVD typique
          {'100': 1.0, '110': 0.6, '111': 0.25}.
        - orientation : '100'/'110'/'111' = plan de wafer (normale =
          axe z grille) ; la famille la plus parallèle au wafer donne
          le terme de croissance verticale : depth_voxels × taux / n_z.
        - chaque famille émet, depuis les colonnes de bord de la graine
          exposée, des plans inclinés avançant de taux × depth_voxels en
          distance normale ; un plan de facette contient l'arête sur
          laquelle il repose : il ne s'ancre que sur les bords
          PERPENDICULAIRES à sa montée (pente en x -> bords en x ; pente
          en x et y -> coins seulement ; ex. facettes {111} aux coins
          d'une fenêtre alignée aux axes, {110} le long des bords) ; un
          voisin hors grille n'est pas un bord (le wafer continue) ; la
          surface résultante est le minimum entre le terme vertical et
          tous les plans de facettes — près des bords, les facettes
          (plus basses) dominent, d'où le toit facetté ; loin des bords,
          la croissance verticale reprend.
        - la croissance ne démarre que sur la graine exposée à l'air ;
          pas de croissance hors graine (pas de débordement au-dessus
          du masque), pas de surplomb (carte de hauteur) ; les facettes
          quasi verticales (n_z <= 0.05) sont ignorées.
        - mask : restreint les colonnes de remplissage.

    etch_crystal(grid, depth_voxels, material, orientation="100",
                 facet_rates=None, mask=None)
        Gravure cristallographique anisotrope (KOH, TMAH) : miroir de
        epitaxy[crystal] — la profondeur gravie par colonne est le
        minimum entre le terme vertical (famille la plus parallèle au
        wafer : depth_voxels, c'est la référence de profondeur, les
        autres taux sont relatifs à elle) et les plans lents partant
        des bords du matériau exposé (mêmes règles d'ancrage que
        epitaxy[crystal] : bords perpendiculaires à la montée, voisin
        hors grille = pas un bord). Un plan plus rapide que le terme
        vertical ne lie jamais : seules les familles lentes limitent.
        - facet_rates : défaut KOH typique {'100': 1.0, '110': 1.6,
          '111': 0.1} — {111} quasi gravure-arrêt, {110} rapide.
        - seul `material` est retiré, depuis sa surface exposée à l'air,
          vers le bas ; arrêt au premier voxel qui n'est pas `material`
          (masque dur, matériau d'arrêt) ; la surface couverte par un
          autre matériau n'est pas exposée -> protégée.
        - repère : sur wafer (100), les arêtes de masque alignées aux
          axes grille sont des <100> cristallographiques -> parois
          latérales = plans {110} (rapides avec les taux KOH) -> fond
          plat + facettes {111} aux coins ; la fosse trapézoïdale
          classique {111} à 54,7° demande des arêtes à 45° (fenêtre en
          losange, ex. layout GDS).
        - limites (carte de hauteur, comme epitaxy[crystal]) : pas de
          sapement latéral sous le masque ; facettes quasi verticales
          (n_z <= 0.05) ignorées ; familles limitées à
          {100}/{110}/{111}.
        - mask : restreint les colonnes gravies.

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
