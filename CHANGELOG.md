# Changelog

Format : versions stables alignées sur `pyproject.toml` (semver). Le hash
`structure_sha256` du deck de démo (`e1316a2970a1e6f38612e6ee98d954123e5fe173f2acd06053d8cb63283743c1`)
garantit la reproductibilité de chaque version.

## Non publié (v0.2)

- **Backend GPU Taichi** (`src/voxelfab/ops/gpu.py`, extra `[gpu]`) :
  toutes les étapes (substrate, deposit conformal/directionnel, etch
  isotrope/directionnel, etch_selective, grow, epitaxy seed/crystal,
  etch_crystal, fill, cmp), **voxel-exact vs CPU** — même `structure_sha256`
  (vérifié : tests d'équivalence `tests/test_equiv_gpu.py`, 14 tests, +
  deck de démo rejoué sur les deux backends, hash identique au golden
  `e1316a29…`). Noyaux Taichi par shape de grille (closures sur champs
  `ti.field`), déterminisme : réductions min/max + atomics entiers,
  arithmétique f64 dans le même ordre que NumPy. `r > 4` : croix 6-voisine
  itérée (comme `binary_dilation(iterations=r)` CPU), pas de boucle sur
  toute la boule L1.
- `vf run/demo --backend gpu` ; `vf bench [n] --backend all` imprime le
  comparatif cpu vs gpu (warm-up JIT Taichi non chronométré côté GPU).
- Mesuré sur GTX 1660 (CUDA) vs CPU Numba 4 cœurs, noyaux chauds :
  256³ (16,8 M vx) ×2,8–×13,7 ; 464³ (100 M vx) ×2,2–×7,4 ;
  640³ (262 M vx) ×1,0–×7,2 selon l'étape (les morphismes larges et la
  gravure directionnelle gagnent le plus ; le transfert PCIe hôte↔device
  par étape ~2×N octets et le JIT Taichi (~0,3 s/noyau, une fois par shape
  et par processus) pénalisent les petites grilles).

## Non publié (v0.1.x)

- CLI `vf xmf [FICHIER]` : pont Xdmf (.xmf) écrit à côté d'un
  `structure.h5` pour visualisation ParaView en nanomètres réels —
  le `.h5` reste la source unique (référence relative, couple
  déplaçable) ; `export.write_xmf()` côté API Python ;
- test `tests/test_xmf.py` : contrat du pont Xdmf vérifié au parseur
  XML (ordre z-y-x, `Center="Cell"`, HDF UInt précision 1, référence
  relative, pas en nm réels, fichier absent → None) ;
- guide : §6.4 route B simplifiée (`vf xmf`), nouvelle §6.5
  « Débutants : visualiser une coupe avec matplotlib » (niveaux 0/1/2,
  pièges d'axes documentés).

## v0.1.0 — 2026-11-17

Première version stable : **émulation géométrique** (morphologie
mathématique, pas résolution d'équations) de procédés de fabrication
virtuelle sur grilles voxel, déterministe, backend CPU.

### Moteur de procédés (NumPy/SciPy/Numba, tout vectorisé)

- `substrate`, `deposit` (conformal euclidien, directionnel PVD, sélectif
  via `on_materials`), `etch` (directionnel avec matériau d'arrêt `until`,
  isotrope euclidien), `fill` (tranchées connectées à la surface), `cmp`
  (planarisation, `stop_on`), `litho` (sucre syntaxique) ;
- `etch_selective` : sélectivité différentielle par budget de temps
  (`rates`), directional et isotropic, équivalence garantie avec `etch` ;
- `grow` : croissance par conversion — oxydation thermique (rapport de
  volumes Deal-Grove ~44/56), siliciuration (`into`), consommation au
  prorata des couches minces ;
- `epitaxy` : épitaxie sélective — mode `seed` (SEG conforme géodésique,
  bloquée par le masque) et mode `crystal` (facettes {hkl} lentes, plans
  (100)/(110)/(111), ancres perpendiculaires à la montée, toit en
  diamant des sources/drain SiGe) ;
- `etch_crystal` : gravure cristallographique anisotrope KOH/TMAH —
  miroir de `epitaxy crystal` (plans lents qui protègent), fosse
  trapézoïdale {111} à 54,7°, fosse auto-limitée (membrane), arrêt sur
  masque dur.

### Masques et layout

- masques composites : `rect`, `bands`, `invert`, `layer` ;
- import GDS/OASIS (`gdstk` + `shapely`) : biais de masque `bias_nm`,
  grille déduite de la bounding box, SREF/AREF aplatis.

### Plateforme

- deck YAML v1 + CLI `vf` (`run`, `info`, `demo`, `bench`) ;
- sorties : `structure.h5` canonique (grille uint8 chunkée + table
  matériaux + provenance + `structure_sha256`), `views` PNG, `report`
  JSON (métrologie virtuelle), `vtk` (ParaView/PyVista), `stl` par
  matériau ;
- déterminisme contractuel : tests golden rejouant le deck de démo ;
- 35 tests, CI verte ; doc : `README.md`, `docs/USER_GUIDE.md`,
  `docs/ARCHITECTURE.md`.

### Hors périmètre de la v0.1 (documenté)

- backend GPU Taichi avec équivalence voxel-exacte CPU↔GPU (v0.2) ;
- familles {hkl} au-delà de {100}/{110}/{111} ; sapement latéral sous
  masque et surplomb FLO (cartes de hauteur).

## v0.1.0-rc — versions de développement

Travail préparatoire poussé sur `main` (nouvelle op = un commit de
sémantique + kernel + tests + docs) : `etch_selective`, `grow`,
`epitaxy` (seed puis crystal), `etch_crystal`.
