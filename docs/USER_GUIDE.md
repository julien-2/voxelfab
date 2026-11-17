# voxelfab — Guide utilisateur (v0.1)

*Simulation de procédés de fabrication virtuelle sur grilles voxel —
dans l'esprit de SEMulator3D, mais géométrique (morphologie mathématique),
pas physique.*

Ce guide couvre l'utilisation du logiciel **et** l'extension du moteur avec
vos propres opérations. Pour l'architecture produit et la feuille de route,
voir [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table des matières

1. [Concepts](#1-concepts)
2. [Installation](#2-installation)
3. [Premiers pas](#3-premiers-pas)
4. [Le deck YAML — référence complète](#4-le-deck-yaml--référence-complète)
5. [CLI — référence](#5-cli--référence)
6. [Sorties et formats](#6-sorties-et-formats)
7. [Déterminisme et tests golden](#7-déterminisme-et-tests-golden)
8. [Import de layout GDS/OASIS](#8-import-de-layout-gdsoasis)
9. [Ajouter ses propres opérations](#9-ajouter-ses-propres-opérations)
10. [Dépannage](#10-dépannage)

---

## 1. Concepts

| Concept | Description |
| --- | --- |
| **Voxel** | Plus petit volume de la grille (un cube de `pitch_nm` de côté). La structure est un tableau 3D `uint8` où chaque voxel porte l'**id d'un matériau** (0 = air). |
| **Pitch** | Taille d'un voxel en nm. Toute la résolution du simulateur : 2 nm/voxel = détails de 2 nm représentables. Axes : `grid[x, y, z]`, `z` vertical, la surface en `z = nz-1`. |
| **Matériau** | Déclaré par **nom** dans le deck (`Si`, `SiO2`…), avec une couleur pour les rendus. Jamais d'indices numériques en entrée utilisateur. |
| **Deck** | Fichier YAML décrivant le procédé : grille, matériaux, liste ordonnée d'étapes, sorties. C'est **le** fichier que vous éditez. |
| **Backend** | Implémentation du moteur (CPU aujourd'hui, GPU Taichi en v0.2). Même deck + même backend versionné → même structure. |
| **Structure** | Résultat 3D : fichier HDF5 + hash `structure_sha256` (SHA-256 des octets de la grille). |

Le moteur est **délibérément géométrique** : un dépôt conformal est une
dilation mathématique, pas une équation de cinétique chimique. C'est ce qui
le rend rapide (1–100 M voxels), déterministe et prévisible.

---

## 2. Installation

Avec [uv](https://docs.astral.sh/uv/) (recommandé) :

```bash
uv pip install .                 # coeur : numpy, scipy, matplotlib, pyyaml, h5py
uv pip install ".[bench]"        # + Numba (fortement recommandé : gravures directionnelles parallélisées)
uv pip install ".[layout]"       # + import GDS/OASIS (gdstk, shapely)
uv pip install ".[view]"         # + export VTK (pyvista)
uv pip install -e ".[dev]"       # développement (pytest)
```

Python ≥ 3.10. Aucune dépendance GPU requise.

---

## 3. Premiers pas

```bash
vf demo -o out/                  # écrit demo_deck.yaml puis l'exécute
```

Sorties dans `out/` :

| Fichier | Contenu |
| --- | --- |
| `structure.h5` | La structure 3D (grille + matériaux + provenance) |
| `mos3d.png` | Vue de dessus + coupes x-z et y-z |
| `report.json` | Métrologie virtuelle + provenance + `structure_sha256` |

Le deck de démo est un MOSFET planaire complet : tranchées d'isolation STI →
remplissage oxyde → CMP → empilement de grille → gravure de grille →
spacers → épitaxie source/drain sélective → ILD → contacts tungstène.
Ouvrez `demo_deck.yaml` : chaque étape est commentée dans ce guide (§4).

Lire la structure en Python :

```python
from voxelfab.export import load_h5
g = load_h5("out/structure.h5")     # -> VoxelGrid
print(g.data.shape, g.pitch)        # (120, 60, 64) 2.0
```

Ou à la main :

```python
import h5py
with h5py.File("out/structure.h5", "r") as f:
    print(dict(f.attrs))            # provenance, structure_sha256
    print(f["materials"][:])        # table (id, name, color)
    grid = f["grid"][:]             # uint8 (nx, ny, nz)
```

---

## 4. Le deck YAML — référence complète

### 4.1 Choisir sa résolution

Dans voxelfab, la résolution est **un seul paramètre : `pitch_nm`** — la
taille d'un voxel en nanomètres. Tout le reste en découle.

**La chaîne de conversion.** Les étapes du deck parlent en nm ; le moteur
les convertit en voxels à l'exécution : `n = round(thickness_nm / pitch)`
(minimum 1 voxel si l'épaisseur est > 0).

| `thickness_nm` | `pitch_nm` | Voxels réellement déposés/gravés |
| --- | --- | --- |
| 4.0 | 2.0 | 2 |
| 1.0 | 2.0 | **1** (arrondi 0.5 → 0, forcé à 1) |
| 0.4 | 2.0 | **1** (minimum 1 si > 0) |
| 4.0 | 1.0 | 4 |

Deux conséquences :

- **Quantification** : un dépôt de 1 nm avec un pitch de 2 nm dépose quand
  même 1 voxel (2 nm). Le moteur ne représente rien en dessous du pitch.
  Règle pratique : **`pitch_nm` ≤ plus petite dimension critique / 2**.
  Un oxyde de grille de 4 nm veut un pitch de 2 nm (2 voxels, minimum pour
  voir quelque chose) ou 1 nm (4 voxels, plus précis, 8× plus cher).
- **La résolution est la même dans les 3 directions** : `pitch_nm` s'applique
  aussi à `z` (les gravures descendent voxel par voxel).

**Ce qu'on configure.**

```yaml
grid: {pitch_nm: 2.0, nx: 120, ny: 60, nz: 64}
```

- `pitch_nm` : la résolution physique (nm/voxel)
- `nx, ny, nz` : la taille de la boîte de simulation en voxels — l'étendue
  simulée est `nx × pitch` nm en x, etc.

Avec un **layout** GDS/OAS (§8), `nx`/`ny` sont ignorés : ils sont déduits
de la bounding box du layout et du pitch. C'est donc **le pitch qui décide
de la finesse** de rasterisation du layout ; seuls `pitch_nm` et `nz`
restent à vous.

**Le piège : `target_z` et `to_plane_z`.** Ces deux paramètres (`cmp`,
`fill`) sont en **index de voxel**, pas en nm — ils se lisent directement
sur la géométrie. Si vous changez le pitch, ils ne sont plus valables :
le plan STI de la démo est `target_z: 19` à pitch 2.0 (38 nm au-dessus du
bas de grille), donc `target_z: 38` à pitch 1.0 pour le même plan physique.
C'est la seule partie du DSL qui dépend de la résolution.

**Coût mémoire.** La grille fait 1 octet/voxel (uint8) ; comptez 2 à 3× la
taille de la grille en temporaires pendant les opérations booléennes
(la gravure directionnelle Numba travaille en place, quasi sans
temporaire). `vf bench 256` chronomètre chaque type d'étape à 16,7 M voxels.

| Grille | Voxels | RAM grille | Étendue physique @ 2 nm |
| --- | --- | --- | --- |
| 120×60×64 | 0,46 M | ~0,5 Mo | 240×120×128 nm |
| 500³ | 125 M | ~125 Mo | 1 µm cube |
| 100 M | ~460³ | ~100 Mo | ~0,92 µm cube |

**Effet sur le déterminisme** : le hash `structure_sha256` dépend du pitch
— changer `pitch_nm` change la structure octet par octet. C'est normal :
le golden ne se régénère que si la **sémantique** change (§7), pas si vous
expérimentez le pitch dans votre propre deck.

### 4.2 Le squelette du deck

```yaml
version: 1                      # version du format (obligatoire)
grid:                           # obligatoire (ou un layout, §8)
  pitch_nm: 2.0
  nx: 120
  ny: 60
  nz: 64
materials:                      # noms -> couleur
  Si:   {color: '#4d79ff'}
  SiO2: {color: '#c9c9c9'}
layout:                         # optionnel : masques depuis GDS/OAS (§8)
  file: inv.gds
  cell: TOP
  bias_nm: -2.0
  layers: {gate: [10, 0]}
steps:                          # liste ordonnée d'étapes
  - {op: substrate, material: Si, thickness_nm: 40}
outputs:                        # h5 | views | vtk | stl | report
  - {type: h5, file: structure.h5}
  - {type: report, file: report.json}
```

**Règles d'unités** : toute longueur est en **nm** (`*_nm`) ; les plans
verticaux (`target_z`, `to_plane_z`) sont en **index de voxel** car ils
sont lus directement sur la géométrie (voir §4.1). Les matériaux sont
toujours référencés par nom.

### 4.3 Les étapes (`steps`)

#### `substrate` — substrat

```yaml
- {op: substrate, material: Si, thickness_nm: 40}
```

Remplit le fond de la grille (`z < épaisseur`) avec le matériau.

#### `deposit` — dépôt

```yaml
- {op: deposit, material: SiO2, thickness_nm: 4}                       # conformal (défaut)
- {op: deposit, material: Si, thickness_nm: 12,
   mode: directional, on_materials: [Si]}                              # sélectif
```

| Mode | Sémantique | Usage typique |
| --- | --- | --- |
| `conformal` | Couche d'épaisseur constante sur **toute surface exposée** (dilation euclidienne du solide). | CVD, oxyde de grille, spacers (avant RIE) |
| `directional` | Dépôt **vertical** au-dessus des surfaces exposées, pas de recouvrement latéral. | PVD, bouchons de contacts, épitaxie |

`on_materials: [Si, poly]` (directional uniquement) : dépôt sélectif —
seules les colonnes dont le **matériau de surface** est listé reçoivent le
dépôt. C'est ainsi qu'on modélise une épitaxie sélective.

> Note de précision : en conformal, un rayon ≤ 4 voxels utilise une boule
> euclidienne exacte ; au-delà, une boule L1 (octaèdre) approxime la sphère
> — les coins sont légèrement plus « pointus ». C'est un compromis
> performance assumé et documenté.

#### `grow` — croissance par conversion (oxydation, siliciuration)

```yaml
# Oxydation thermique : 10 nm de SiO2 consomment 4,4 nm de Si
- {op: grow, material: SiO2, from: Si, depth_nm: 4.4, expansion: 2.27}

# Siliciuration : le Ni déposé juste avant réagit avec le Si
- {op: deposit, material: Ni, thickness_nm: 10}
- {op: grow, material: NiSi, from: Si, depth_nm: 18.3, expansion: 1.39,
   into: [Ni]}
- {op: etch, depth_nm: 5, materials: [Ni]}   # dip de nettoyage du Ni excédentaire
```

Contrairement à `deposit` (ajout de matière) ou `etch` (retrait), `grow`
**convertit** un matériau en un autre à leur interface : c'est l'émulation
de l'oxydation thermique, de la nitruration ou de la siliciuration.

| Paramètre | Rôle |
| --- | --- |
| `from` | matériau consommé par la réaction (obligatoire) |
| `material` | produit de la réaction (obligatoire) |
| `depth_nm` | épaisseur **consommée** de `from` — la grandeur d'entrée de la réaction |
| `expansion` | épaisseur de produit produite par unité consommée (défaut 1.0) |
| `into` | matériaux que le produit peut aussi remplacer côté réactif |
| `mask` | limite la réaction aux colonnes ouvertes |

Sémantique :

- le produit occupe la zone consommée **plus** une croissance de
  `(expansion − 1) × depth_nm` du côté réactif (air et matériaux `into`).
  Pour Si → SiO2, `expansion: 2.27` reproduit le rapport de volumes
  Deal-Grove : **~44 % de l'oxyde sous la surface d'origine, ~56 %
  au-dessus** — un oxyde de 10 nm consomme 4,4 nm de Si ;
- **l'orientation cristalline n'intervient pas sur la géométrie** : le
  ratio 44/56 est un rapport de volumes, indépendant de l'orientation ;
  seule la cinétique en dépend (le plan (111) s'oxyde ~1,7× plus vite que
  (100)), et en émulation cela se règle par le `depth_nm` de chaque étape ;
- la réaction n'attaque que les interfaces `from` / côté réactif : les
  matériaux inertes (oxyde, nitrure) **masquent** le matériau consommé,
  comme un vrai masque d'oxydation ;
- si `from` est plus mince que `depth_nm` (poly épuisé, îlot mince), il
  est consommé en entier et la croissance est **réduite au prorata** ;
- la croissance suit toutes les surfaces exposées (fond de tranchée,
  parois latérales) avec un budget proportionnel par colonne ;
- siliciuration : `into: [Ni]` convertit le métal dans la limite du
  budget ; l'excédent de métal reste en place et s'enlève par gravure,
  exactement comme le dip de nettoyage qui suit une vraie siliciuration.

Limites assumées (émulation, pas simulation) : pas de « bird's beak »
LOCOS (le front ne se propage pas sous le nitrure à travers l'oxyde
existant) et pas de poussée mécanique (l'oxyde bloqué par un spacer ne
soulève rien).

#### `epitaxy` — épitaxie sélective et facettée

```yaml
# SEG simple : croissance conformale depuis le Si exposé uniquement
- {op: epitaxy, mode: seed, material: Si, depth_nm: 40, seed: Si}

# Épitaxie facettée : facettes {111} à 54,7° sur wafer (100)
- {op: epitaxy, mode: crystal, material: SiGe, depth_nm: 60,
   seed: Si, orientation: '100',
   facet_rates: {'100': 1.0, '110': 0.6, '111': 0.25}}
```

Contrairement à `deposit` (dépose sur **toutes** les surfaces), `epitaxy`
ne fait croître que depuis une **graine monocristalline** (`seed`) : c'est
l'épitaxie sélective (SEG). L'oxyde et le nitrure bloquent la croissance,
comme dans un vrai procédé.

| Paramètre | Rôle |
| --- | --- |
| `material` | matériau déposé (obligatoire) |
| `seed` | matériau graine monocristallin — la croissance ne part que de lui (obligatoire) |
| `depth_nm` | épaisseur crue (cf. sémantique par mode) |
| `mode` | `seed` (défaut) ou `crystal` |
| `orientation` | plan de wafer : `'100'`, `'110'` ou `'111'` (mode crystal) |
| `facet_rates` | taux relatifs par famille `{hkl}` (mode crystal) |
| `mask` | limite les colonnes de croissance |

**Mode `seed` — SEG conforme.** Croissance sur toutes les faces exposées
de la graine (fond de fenêtre, parois), en suivant la géodésie à travers
l'air : un dépôt de 40 nm dans une fenêtre d'oxyde remplit la fenêtre mais
ne dépose **rien** au-dessus de l'oxyde, même en face — la matière ne peut
pas « sauter » le masque. C'est la différence avec `deposit conformal`.

**Mode `crystal` — épitaxie facettée.** La construction SEG classique :
la croissance verticale est limitée par des **plans cristallographiques
lents** partant des bords de la graine exposée. Sur wafer (100), les
facettes {111} montent à **54,7°** du plan de wafer (45° en coupe axiale,
54,7° en coupe diagonale) ; dans une fenêtre étroite, les facettes des
bords opposés se rejoignent et forment le **toit en diamant** typique du
SiGe épitaxié en source/drain. Les taux (`facet_rates`, normalisés au
maximum) décident de la hauteur des facettes : `{111}` lent → facettes
dominantes près des bords ; tous égaux → croissance verticale plate.

- `depth_nm` = épaisseur verticale crue de la famille la plus parallèle
  au wafer (sur (100) avec `{'100': 1.0}` : `depth_nm` exactement) ;
- défaut : `{'100': 1.0, '110': 0.6, '111': 0.25}` (Si CVD typique) ;
- l'orientation `'110'`/`'111'` tourne le repère cristallin : les angles
  de facettes et la famille dominante suivent la cristallographie
  (la famille la plus parallèle au wafer (100)/(110)/(111) est
  {100}/{110}/{111} respectivement) ;
- chaque facette ne s'ancre que sur les bords de graine
  **perpendiculaires à sa montée** (un plan de facette contient l'arête
  sur laquelle il repose) : sur une fenêtre alignée aux axes, les
  facettes {111} partent des **coins** et les {110} le long des bords ;
  un bord de grille n'est pas un bord de graine (le wafer continue) —
  un substrat nu croît donc verticalement, sans facette ;
- la croissance ne démarre que sur la graine exposée à l'air : pas de
  débordement au-dessus du masque, pas de croissance sur l'oxyde.

Limites assumées : croissance verticale (carte de hauteur) — pas de
surplomb ni de recouvrement latéral au-dessus du masque (FLO) ; facettes
quasi verticales ignorées ; familles limitées à {100}/{110}/{111}.

#### `etch` — gravure

```yaml
- {op: etch, depth_nm: 60, materials: [poly],
   mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
- {op: etch, depth_nm: 4, mode: isotropic, materials: [SiO2],
   mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
- {op: etch, depth_nm: 60, until: Si, mask: {layer: contact_s}}
```

| Mode | Sémantique |
| --- | --- |
| `directional` (défaut) | RIE vertical : depuis le sommet solide de chaque colonne ouverte du masque, descend **en sautant les voxels d'air**, retire jusqu'à `depth_nm` de matériau autorisé, **s'arrête** sur `until` (matériau d'arrêt) ou sur un matériau non autorisé. |
| `isotropic` | Attaque uniforme dans toutes les directions depuis toute surface exposée à l'air (érosion euclidienne du solide). |

Paramètres : `materials` (sélectivité — absent = grave tout), `until`
(matériau d'arrêt, directional seulement), `mask` (§4.4).

> Limite : tous les matériaux listés sont gravés **au même rythme**. Pour
> des vitesses différentielles (sélectivité réelle), voir `etch_selective`.

#### `etch_selective` — gravure à sélectivité différentielle

```yaml
- {op: etch_selective, depth_nm: 30, rates: {SiO2: 10.0, SiN: 1.0},
   mask: {layer: pad_open}}
- {op: etch_selective, depth_nm: 20, mode: isotropic,
   rates: {SiO2: 8.0, Si: 1.0}, mask: {layer: open}}
```

`rates` associe chaque matériau à sa **vitesse relative** ; le matériau de
référence (taux 1.0) est gravé de `depth_nm` — les autres proportionnellement.

| Mode | Sémantique |
| --- | --- |
| `directional` (défaut) | Même parcours par colonne que `etch` directional, mais avec un **budget de temps** : retirer un voxel du matériau m coûte `1/rate_m`. Arrêt sur `until`, sur un matériau absent de `rates`, ou budget épuisé. |
| `isotropic` | Chaque matériau m est érodé du rayon `round(depth_nm × rate_m)`. Approximation documentée : les interfaces inter-matériaux comptent comme des surfaces d'attaque. `until` ignoré. |

Points clés :

- les matériaux **absents de `rates` ne sont jamais gravés** et bloquent la
  gravure comme un matériau d'arrêt — c'est le comportement attendu d'un
  wet etch sélectif (ex. : grave SiO2, stop sur SiN) ;
- `depth_nm` est la profondeur dans le matériau de référence : avec
  `rates: {SiO2: 10.0, SiN: 1.0}` et `depth_nm: 30`, le SiO2 est gravé de
  300 nm « équivalents » et le SiN de 30 ;
- **équivalence garantie** : `rates: {m: 1.0}` reproduit exactement `etch`
  (directional et isotropic) — c'est testé (`tests/test_etch_selective.py`),
  et le futur backend GPU devra le respecter aussi.

#### `etch_crystal` — gravure cristallographique anisotrope (KOH)

```yaml
# KOH : fosse auto-limitée à parois {111} (fenêtre à arêtes <110>, ex. GDS)
- {op: etch_crystal, material: Si, depth_nm: 300, orientation: '100'}
```

Miroir de `epitaxy crystal` : la profondeur gravie par colonne est le
minimum entre le terme vertical (famille la plus parallèle au wafer) et
les **plans lents partant des bords du matériau exposé** (mêmes règles
d'ancrage). Un plan plus rapide que le terme vertical ne lie jamais :
seules les familles lentes protègent.

| Paramètre | Rôle |
| --- | --- |
| `material` | matériau gravé (obligatoire) — un seul ; tous les autres matériaux servent de masque dur |
| `depth_nm` | profondeur verticale du fond plat (référence = famille la plus parallèle au wafer, taux 1.0) |
| `orientation` | plan de wafer : `'100'` (défaut), `'110'` ou `'111'` |
| `facet_rates` | taux relatifs par famille `{hkl}` ; défaut KOH typique `{'100': 1.0, '110': 1.6, '111': 0.1}` — {111} quasi gravure-arrêt, {110} rapide |
| `mask` | limite les colonnes gravies |

Comportements :

- **fenêtre alignée aux axes sur (100)** : les arêtes de masque y sont
  des `<100>` cristallographiques → parois latérales = plans {110}
  (rapides avec les taux KOH) → fond plat + facettes {111} aux coins ;
- **fenêtre à arêtes à 45°** (losange, via un layout GDS) : les arêtes
  sont des `<110>` → parois {111} à **54,7°** → la fosse trapézoïdale
  classique du KOH ; fenêtre étroite → fosse **auto-limitée** en pointe
  (le terme vertical n'est jamais atteint : c'est le principe de la
  membrane KOH) ;
- **arrêt** au premier voxel qui n'est pas `material` (masque dur, oxyde
  enfoui) ; la surface couverte par un autre matériau n'est pas exposée
  → protégée.

Limites assumées (carte de hauteur, comme `epitaxy crystal`) : pas de
sapement latéral sous le masque ; facettes quasi verticales ignorées ;
familles limitées à {100}/{110}/{111}.

#### `litho` — lithographie + transfert

```yaml
- {op: litho, mask: {layer: sti}, depth_nm: 12}
```

Sucre syntaxique : gravure `directional` à travers le masque (sans
sélectivité ni matériau d'arrêt par défaut).

#### `fill` — remplissage de tranchées

```yaml
- {op: fill, material: SiO2, to_plane_z: 19}
```

Remplit les volumes d'air **connectés à la surface** et situés sous le plan
`to_plane_z` (index de voxel, voir §4.1). Les cavités fermées ne sont pas
remplies — comme dans un vrai procédé de spin-on/flowable.

#### `cmp` — planarisation mécanique

```yaml
- {op: cmp, target_z: 19}
- {op: cmp, target_z: 52, stop_on: Si}
```

Abrase **tout voxel solide au-dessus du plan** `target_z` (index de voxel,
voir §4.1). `stop_on` : matériau jamais traversé (première occurrence
depuis la surface). Retire typiquement le relief d'un dépôt conformal ou
les bouchons après remplissage.

### 4.4 Les masques

Un masque est un champ 2D booléen sur `(x, y)` : `True` = zone ouverte
(gravée/déposée). Quatre formes, composables :

```yaml
mask: {layer: sti}                       # couche du layout (§8)
mask: {rect: [x0, x1, y0, y1]}           # rectangle, fractions de grille (0..1)
mask: {bands: [[0.0, 0.18], [0.82, 1.0]]}  # bandes ouvertes en x, sur tout y
mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}   # complément, composable
```

`invert` s'applique à n'importe quel masque, y compris `{layer: ...}` —
c'est ainsi qu'on grave *partout sauf* la grille.

### 4.5 Les sorties (`outputs`)

| Type | Fichier | Dépendance |
| --- | --- | --- |
| `h5` | Structure HDF5 canonique (§6) | cœur (h5py) |
| `views` | PNG : vue de dessus + coupes x-z / y-z | cœur (matplotlib) |
| `report` | JSON : métrologie + provenance | cœur |
| `vtk` | Export ParaView/PyVista | extra `[view]` |
| `stl` | STL par matériau (marching cubes) | scikit-image + numpy-stl |

---

## 5. CLI — référence

```
vf run deck.yaml [-o OUTDIR] [--backend cpu]
vf info deck.yaml
vf demo [-o OUTDIR]
vf bench [N]                    # micro-benchmark sur grille N³ (défaut 256)
```

- `vf run` : exécute le deck ; `-o` définit le répertoire des sorties
  (défaut : celui du deck).
- `vf info` : valide et résume le deck **sans l'exécuter** — utile pour
  déboguer un deck (version, grille, matériaux, étapes, sorties).
- `vf bench` : chronomètre chaque type d'étape ; sert de test de fumée
  (128 → 2 M voxels, 256 → 16,7 M, 430 → ~80 M).
- `--backend` : réservé au backend GPU (v0.2) ; `cpu` est la seule valeur
  aujourd'hui.

Codes de sortie : `0` succès, `2` erreur de deck (message explicite).

---

## 6. Sorties et formats

### 6.1 `structure.h5` — la structure (format canonique)

```
/grid        uint8 (nx, ny, nz)   chunké (128³ max) + gzip + shuffle
/materials   table (id, name, color) des matériaux, air exclu
attrs        pitch_nm, structure_sha256
             prov_deck_sha256, prov_deck, prov_grid, prov_pitch_nm,
             prov_voxelfab_version, prov_backend
```

Le `structure_sha256` est le hash SHA-256 des **octets bruts de la grille**
NumPy — jamais des octets du fichier HDF5 (qui dépendent de la version de
la libhdf5). Deux runs identiques produisent des fichiers identiques :
aucun timestamp ni timing n'est écrit dans ce fichier.

### 6.2 `report.json` — métrologie + provenance

```json
{
  "provenance": {
    "deck": "decks/demo_deck.yaml",
    "deck_sha256": "…",
    "grid": [120, 60, 64],
    "pitch_nm": 2.0,
    "voxelfab_version": "0.1.0",
    "backend": "cpu",
    "steps": [{"n": 1, "op": "substrate", "s": 0.002}, …]
  },
  "volume_fractions": {"Si": 0.31, "SiO2": 0.42, …},
  "structure_sha256": "e1316a29…"
}
```

Le bloc `provenance` rend tout résultat rejouable et auditable :
`deck_sha256` identifie le deck exact (contenu, pas le chemin),
`voxelfab_version` + `backend` identifient le moteur.

### 6.3 Métrologie en Python

```python
from voxelfab import Metrology
met = Metrology(grid)
met.thickness_at(x, y, mid)   # épaisseur verticale d'un matériau (voxels)
met.top_material(x, y)        # matériau de surface de la colonne
met.cd(y, mid)                # plus grande largeur continue sur la coupe y (CD)
met.report()                  # fractions volumiques
```

### 6.4 Visualiser dans ParaView

**ParaView n'ouvre pas `structure.h5` directement** : c'est un HDF5
maison (`/grid` uint8 + table `/materials`), pas un fichier Xdmf ou VTK
HDF. Trois routes, de la plus simple à la plus intégrée.

#### Route A — la sortie `vtk` native (recommandée)

```bash
uv pip install ".[view]"          # pyvista
```

```yaml
outputs:
  - {type: h5,   file: structure.h5}
  - {type: vtk,  file: structure.vtk}
```

`vf run` produit un `structure.vtk` que ParaView ouvre tel quel : une
image 3D avec un champ de cellules `material` (0 = air, puis les ids de
la table `/materials`). Le pas vaut 1 unité = 1 voxel (coordonnées en
voxels, pas en nm).

#### Route B — pont Xdmf (le `.h5` reste la source unique)

Un petit fichier `.xmf` pointe vers `/grid` ; ParaView lit le HDF5 à
travers. Attention : Xdmf énumère les dimensions **du plus rapide au plus
lent** (z, y, x), l'inverse de NumPy. Générateur une fois pour toutes :

```python
import h5py

with h5py.File("out/structure.h5") as f:
    nx, ny, nz = f["grid"].shape
    p = float(f.attrs["pitch_nm"])

xml = f"""<?xml version="1.0" ?>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="voxelfab">
      <Topology TopologyType="3DCoRectMesh" NumberOfElements="{nz+1} {ny+1} {nx+1}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>
        <DataItem Format="XML" Dimensions="3">{p} {p} {p}</DataItem>
      </Geometry>
      <Attribute Name="material" AttributeType="Scalar" Center="Cell">
        <DataItem Format="HDF" Dimensions="{nz} {ny} {nx}"
                  NumberType="UInt" Precision="1">structure.h5:/grid</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
open("out/structure.xmf", "w").write(xml)
```

Poser `structure.xmf` **à côté de** `structure.h5`, ouvrir le `.xmf` dans
ParaView. Avantage sur la route A : les coordonnées sont en
**nanomètres réels** (le pas vient de `pitch_nm` lu dans les attrs) et le
`.h5` reste le format canonique.

#### Route C — conversion ponctuelle en `.vti` (sans relancer le deck)

```python
import h5py, numpy as np, pyvista as pv

with h5py.File("out/structure.h5") as f:
    data = f["grid"][()]
    p = float(f.attrs["pitch_nm"])

nx, ny, nz = data.shape
img = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1),
                   spacing=(p, p, p))
img.cell_data["material"] = data.flatten(order="F").astype(np.int32)
img.save("out/structure.vti")
```

#### Réglages utiles dans ParaView

- **Masquer l'air** : l'id 0 remplit toute la boîte → filtre **Threshold**
  sur `material` (Minimum = 1), ou **Clip** pour une vue en coupe ;
- **Couleurs** : `Color by → material` en couleurs discrètes
  (*Rescale Mode → Never*), et reproduire les couleurs du deck depuis la
  table `/materials` du `.h5` (id, nom, couleur) ;
- **Coupes** : filtre **Slice** (plans x-z / y-z propres, à comparer aux
  `views` PNG) ; **Save Screenshot** pour exporter une vue.

---

## 7. Déterminisme et tests golden

**Contrat** : même deck + même layout + même version du moteur → structure
**byte-identique** (`structure_sha256`).

Le test `tests/test_golden.py` rejoue le deck de démo et compare le hash à
une référence commitée (`tests/golden/golden.json`). Si le test échoue :

- **changement involontaire** (refactor, montée de version numpy/scipy) →
  traquer la divergence, ne **jamais** régénérer la référence à la légère ;
- **changement voulu de sémantique** → régénérer
  (`VF_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`) et documenter
  le diff dans le message de commit.

Le hash dépend des versions numpy/scipy : une montée de dépendance est un
événement « golden-reviewable ».

---

## 8. Import de layout GDS/OASIS

Extra `[layout]` (gdstk + shapely). Le bloc `layout:` du deck remplace les
masques géométriques :

```yaml
layout:
  file: inv.gds          # .gds / .gds.gz / .oas
  cell: TOP              # cellule ; absent = première top-cell
  bias_nm: -2.0          # biais de masque : + élargit, - rétrécit (nm)
  layers:                # nom logique -> (layer, datatype)
    sti:        [1, 0]
    gate:       [10, 0]
    contact_s:  [20, 0]
    contact_d:  [21, 0]
```

Comportement :

- la **grille est déduite** de la bounding box de la cellule et du pitch —
  `nx`/`ny` du bloc `grid` sont ignorés (seuls `pitch_nm` et `nz` comptent) ;
- les **centres des voxels** échantillonnent le layout (alignement < ½ voxel
  garanti par construction) ;
- les références SREF/AREF sont aplaties ; les FlexPath/RobustPath sont
  convertis en polygones (avertissement sinon) ;
- `bias_nm` compense la gravure latérale (via `shapely.buffer`) ;
- avertissement si une couche demandée est absente ou rasterise vide
  (grille trop grossière).

En Python, `masks_from_layout()` offre en plus le mode `evenodd=True`
(règle even-odd : polygones imbriqués = trous), pas encore exposé dans le
DSL.

---

## 9. Ajouter ses propres opérations

Le moteur est conçu pour ça : **le registre d'opérations**. Chaque étape
est une fonction pure `(grid, params) → grid` enregistrée par backend, et
l'orchestrateur ne connaît que le registre — jamais les implémentations.

### 9.1 Architecture

```
src/voxelfab/ops/
├── base.py    # le registre + la SÉMANTIQUE DE RÉFÉRENCE (docstrings)
└── cpu.py     # implémentations NumPy/SciPy/Numba
```

```python
# ops/base.py
REGISTRIES: dict          # backend -> {(op, mode): fonction}

@register("cpu", "etch", "wet")
def etch_wet(grid, t, materials=None, until_material=None, mask=None): ...

# n'importe où dans run.py
fn = get_step("cpu", "etch", "wet")
```

### 9.2 Le contrat (à respecter absolument)

1. **Unités : voxels, pas nm.** La conversion nm → voxels est faite par
   l'orchestrateur (`grid.voxels(thickness_nm)` : arrondi, minimum 1 si
   > 0). Votre fonction reçoit des entiers.
2. **Modification en place** de `grid.data` (uint8), ou nouvelle grille —
   mais le runner attend une mutation de la grille passée.
3. **Déterminisme** : aucun aléatoire, aucun parcours dépendant du
   parallélisme qui changerait le résultat. Mêmes entrées → mêmes octets.
4. **Sémantique documentée** : le docstring dans `base.py` est le contrat
   que tout futur backend (GPU v0.2) devra reproduire **voxel pour voxel**.
5. **Hygiène mémoire** : uint8/bool partout, LUT 256 entrées
   (`grid.select(materials)`) plutôt que `np.isin`, pas de temporaire
   float64 sur la grille complète.

### 9.3 Tutoriel : ajouter une opération `smooth`

Objectif : adoucir les coins d'une structure par **ouverture
morphologique** (érosion puis dilation) — utile après un CMP ou une épi.

**Étape 1 — la sémantique dans `ops/base.py`** (la docstring est le
contrat) :

```python
#    smooth(grid, t, mask=None)
#        Ouverture morphologique (érosion puis dilation, rayon t) :
#        adoucit les coins convexes sans déplacer les surfaces planes.
```

**Étape 2 — l'implémentation dans `ops/cpu.py`** :

```python
@register("cpu", "smooth")
def smooth(grid, t, mask=None):
    solid = grid.solid()
    opened = _erode(_dilate(solid, t), t)
    if mask is not None:
        opened |= solid & ~mask[:, :, None]      # hors masque : intact
    grid.data[opened] = grid.data[solid]         # conserve les matériaux
    grid.data[~opened] = 0
```

*(Version volontairement simple pour le tutoriel ; elle réécrit les ids
de matériaux par-dessus le masque ouvert — à affiner selon le besoin.)*

**Étape 3 — le branchement DSL dans `run.py`** (`DeckRunner._exec`) :

```python
elif op == "smooth":
    t = g.voxels(self._thickness(step, "depth_nm", "thickness_nm"))
    get_step(self.backend, "smooth")(g, t,
                                     mask=self._mask(step.get("mask"), where))
    self._log(f"smooth {t} vx")
```

**Étape 4 — le test** (`tests/test_smooth.py`) :

```python
from voxelfab import VoxelGrid
from voxelfab.ops import get_step

def test_smooth_rounds_corners():
    g = VoxelGrid(20, 20, 20, pitch=1.0)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :10] = SiO2
    g.data[10, 10, 10] = SiO2            # voxel isolé flottant
    get_step("cpu", "smooth")(g, 2)
    assert g.data[10, 10, 10] == 0       # supprimé par l'ouverture
    assert g.data[5, 5, 5] == SiO2       # le substrat plat est intact
```

**Étape 5 — vérifier le golden** : `uv run pytest tests/ -v`. Le test
golden ne doit bouger que si vous avez modifié la sémantique d'une
opération **existante**.

### 9.4 Nouvelle op vs nouveau mode

- Nouveau **mode** d'une op existante (`etch` wet, deposit spin-on…) :
  `@register("cpu", "etch", "wet")` + une entrée dans la branche DSL
  existante. C'est le cas le plus courant.
- Nouvelle **op** (schéma de paramètres inédit) : nouvelle clé dans
  `_exec` + validation explicite des paramètres (message d'erreur clair,
  cf. `_thickness`, `_mask`).

### 9.5 Checklist avant de partager une opération

- [ ] docstring de sémantique dans `base.py`
- [ ] unités en voxels, conversion faite côté deck (`*_nm`)
- [ ] déterministe (pas d'ordre de parcours non déterministe)
- [ ] test unitaire sur petite grille + golden vert
- [ ] erreurs de validation claires (paramètre manquant/inconnu)
- [ ] (à la v0.2) implémentation GPU équivalente voxel pour voxel

---

## 10. Dépannage

| Symptôme | Cause | Solution |
| --- | --- | --- |
| `clé YAML booléenne (on/yes/off/no…)` | YAML 1.1 convertit les clés non cotées `on:`/`no:` en booléens | écrire `on_materials:` ou coter : `'on': …` |
| `matériau inconnu : 'SiO2'` | matériau utilisé avant d'être déclaré | le déclarer dans `materials:` |
| `couche inconnue (op etch) : 'gate'` | nom absent du bloc `layout.layers` | vérifier l'orthographe et le bloc `layers:` |
| `[gds] ATTENTION couche … vide` | couche plus fine que le pitch | réduire `pitch_nm` ou vérifier le layer/datatype |
| `grid: nx/ny/nz requis` | deck sans bloc `grid` ni `layout` | ajouter l'un des deux |
| golden test échoue | sémantique modée ou dépendance mise à jour | diff voulu ? régénérer + documenter ; sinon investiguer |
| gravure directionnelle lente | Numba absent | `uv pip install ".[bench]"` |
| `matplotlib absent : pas de figure` | import matplotlib impossible | vérifier l'environnement |

---

*voxelfab v0.1 — licence MIT, © 2026 Julien Del.*
