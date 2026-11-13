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
uv pip install ".[bench]"        # + Numba (fortement recommandé : gravure directionnelle parallélisée)
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

### 4.1 Squelette

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
sont lus directement sur la géométrie. Les matériaux sont toujours
référencés par nom.

### 4.2 Les étapes (`steps`)

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
(matériau d'arrêt, directional seulement), `mask` (§4.3).

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
`to_plane_z` (index de voxel). Les cavités fermées ne sont pas remplies —
comme dans un vrai procédé de spin-on/flowable.

#### `cmp` — planarisation mécanique

```yaml
- {op: cmp, target_z: 19}
- {op: cmp, target_z: 52, stop_on: Si}
```

Abrase **tout voxel solide au-dessus du plan** `target_z` (index de voxel).
`stop_on` : matériau jamais traversé (première occurrence depuis la
surface). Retire typiquement le relief d'un dépôt conformal ou les bouchons
après remplissage.

### 4.3 Les masques

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

### 4.4 Les sorties (`outputs`)

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
