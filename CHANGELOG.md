# Changelog

Format : versions stables alignées sur `pyproject.toml` (semver). Le hash
`structure_sha256` du deck de démo (`e1316a2970a1e6f38612e6ee98d954123e5fe173f2acd06053d8cb63283743c1`)
garantit la reproductibilité de chaque version.

## Unreleased — 2026-11-20

- **Batching GPU des petites étapes** (issue #5, piste batching) : une
  séquence consécutive `[substrate?] + deposit mode=directional` (sans
  `on_materials`) est fusionnée en **UN noyau** `k_stack` (un thread par
  colonne, même sémantique que `k_substrate` + `k_dep_dir` empilés) —
  bit-identique (contrat cpu/gpu, 95 tests OK, golden `e1316a29…` inchangé).
  Désactivé si une étape a un `checkpoint` ou si `history:` est actif.
  Auto-grow `nz_max` géré sur le batch. Mesuré 256³ x64 : ×1,1 ; le gain
  vise vulkan software à 10⁸ (~0,1 s/lancement économisé par étape fusionnée).

- **Sélection auto de l'arch Taichi** (issue #5, piste arch) : `VOXELFAB_TAICHI_ARCH`
  force `cuda|vulkan|x64|cpu` ; défaut `auto` = `cuda` si device CUDA détecté
  (`gpu_available`), sinon micro-bench x64 vs vulkan en **subprocess**
  (ti.init non réinitialisable in-process ; noyau dans un vrai fichier .py),
  résultat caché dans `~/.voxelfab/arch_cache.json` (`VOXELFAB_ARCH_CACHE=0`
  pour re-mesurer). Sur ce PC le bench a choisi `vulkan`. Tests :
  `tests/test_arch_select.py` (4 tests). 91 passed. Piste restante sur #5 :
  batching des petites étapes en un seul noyau.

- **perf(gpu): `fill` en union-find GPU** (issue #3, O(N) au lieu de O(P·N)) :
  labélisation des composantes d'air 6-connexes par `atomic_min` +
  compression de chemins (`k_uf_pass`/`k_uf_compress`, nouveaux champs
  `lbl` i32 3D + `topflag` u8 1D), racines des composantes touchant le plan
  top marquées après convergence, `fill_uf_set`. **Bit-identique** vs
  `ndimage.label` CPU (label final = min d'index par composante, indépendant
  de l'ordre) : 87 tests OK, golden `e1316a29…` inchangé. Mesuré (taichi
  x64, warm-up) : fill 256³ **14,6 s -> 1,2 s (×12)**, 128³ 1,17 s -> 0,18 s
  (×6,6) ; le vieux BFS ping-pong supprimé. `epitaxy seed` reste en BFS borné
  (profondeur `depth`, 1 noyau/passe depuis le commit précédent, ×1,3-1,4).

- **perf(gpu): BFS ping-pong de `fill` / `epitaxy seed` allégé** (issue #3) :
  une passe BFS = UN seul noyau ping-pong (`seed_pass_ab/ba`,
  `fill_pass_ab/ba` : `dst = src | (dil6(src) & air & ~src)`) au lieu de 2
  noyaux, et pour `fill` le flag de convergence n'est plus lu en D2H à
  chaque passe mais tous les 8 passes. Bit-identique (87 tests d'équivalence
  passent, golden `e1316a29…` inchangé). Mesuré (taichi x64, warm-up) :
  fill 256³ 14,6s -> 10,5s (×1,4), 128³ 1,17s -> 0,77s (×1,5) ;
  seed 256³ 1,79s -> 1,31s (×1,4). Sur vulkan software (overhead ~0,1 s/launch)
  le gain est plus marqué ; reste en piste : union-find/labélisation O(N).

- **Auto-grow de `nz` aussi en backend gpu** (issue #11, suite) : avec
  `grid.nz_max`, le mode device-résident est désactivé (mode par-op) pour
  grandir la grille côté hôte entre les étapes (`DeckRunner._estimate_z_needed`
  + `ensure_nz` avant substrate/deposit/grow/epitaxy) ; le ctx Taichi est
  recréé automatiquement à la nouvelle shape (`_CTX_CACHE` keyed by shape),
  bascule en tuilé si hors VRAM. Test d'équivalence gpu/cpu avec nz_max
  (octets identiques, même nz final). 87 passed.

- **Historique du procédé dans UN seul fichier HDF5** (issue #9) : nouveau
  bloc `history: {file: history.h5, mode: delta|summary|full, every: N}`.
  Mode `delta` (défaut) = runs RLE axe x des voxels modifiés par étape
  (`runs` (m,4) + `runs_mats`), hash chaîné `step_sha256` par étape ;
  `step_0000` = état initial. Lecture : `load_history()` + `grid_at(path, k)`
  (reconstruction byte-exact, testée vs runs tronqués et vs golden démo) ;
  CLI `vf history FICHIER [--step K] [--format h5|png|vtk]`. Taille mesurée
  (256³, 3 dépôts minces) : delta 42,9 ko vs full 166 ko. GPU résident : D2H
  temporaire par étape via `sync_to_host`. Golden `e1316a29…` inchangé.
  Tests : `tests/test_history.py` (8 tests). 86 passed.

- **Checkpoints entre étapes** (issue #12, sous-issue de #9) : sauvegarde de
  l'état de la grille après une étape clef — `checkpoint: true | [formats] |
  {format: fichier}` dans le step, ou global `checkpoints: {every: N, dir,
  formats}`. Formats `png` / `h5` (rechargeable, attr `prov_checkpoint_step`) /
  `vtk` (pyvista) / `stl` ; nommage auto `ckpt/step_XXXX_<op>.<ext>`.
  En backend gpu device-résident : D2H temporaire via `sync_to_host` sans
  quitter le mode résident. Aucun checkpoint par défaut -> golden
  `e1316a29…` et perfs inchangés. Tests : `tests/test_checkpoints.py`
  (6 tests, dont égalité octet/octet checkpoint vs run tronqué). 78 passed.

- **Substrat multicouche déclaratif + `nz` auto-extensible** (issue #11, CPU) :
  - nouveau bloc `substrate:` exécuté avant `steps`, ordre bottom-first
    (ex. SOI : handle Si / BOX SiO2 / top Si) ; erreur claire si l'épaisseur
    totale dépasse `nz`/`nz_max` ;
  - `grid.nz_max` optionnel : si une op déborde en z (substrate, deposit
    conformal/directionnel, grow), la grille s'agrandit toute seule
    (`VoxelGrid.ensure_nz`, `nz` doublé jusqu'à `nz_max`, message
    `[grid] nz auto-grow`) ; sans `nz_max`, troncature au sommet avec
    `WARNING` (comportement v0.2.2, mais désormais signalé). En backend
    `gpu` l'auto-grow n'est pas supporté (champs Taichi à shape fixe) :
    warning au démarrage, troncature — sous-issue restant sur #11.
  - `{op: substrate}` dans `steps` inchangé (rétrocompatible), golden
    `e1316a29…` inchangé. Tests : `tests/test_substrate_nz.py` (6 tests).
    72 passed.

- **Specs de couches v2 dans le bloc `layout:`** (issue #10) :
  polarité **clear/dark** (`polarity: clear` = complémentaire imprimé dans la
  résine, appliqué après merge), **merge multi-datatypes**
  (`gate: [10, [0, 4]]`, ou `[[20, 0], [21, 0]]`), **biais individuel**
  (`bias_nm` local prioritaire sur le global). Format v1
  (`layers: {nom: [layer, datatype]}`) inchangé et rétrocompatible — golden
  `e1316a29…` inchangé. `layout.py` : `parse_layer_spec` +
  `layer_masks_from_layout` ; `masks_from_layout` accepte un dict de biais
  par `(layer, datatype)`. Fix compat gdstk 1.x (`lib.top_level()`).
  Tests : `tests/test_layout.py` (10 tests, mini GDS gdstk). 66 passed.

- **`--backend auto` implémenté** (`vf run` / `vf demo`, cf. ARCHITECTURE.md
  §5 « Backend selection ») : `ops.resolve_backend('auto')` -> `gpu` si
  taichi est installé **et** qu'un device CUDA est détecté
  (`ops/gpu.py::gpu_available`, sonde `cuInit`/`cuDeviceGetCount` sans effet
  de bord), sinon `cpu` — taichi présent sans carte CUDA ne suffit pas
  (Taichi tomberait en fallback CPU lent). La résolution est faite dans
  `DeckRunner`, donc `report.json` et le HDF5 gardent le **backend effectif**
  (reproductibilité). Hash du deck de démo inchangé (`e1316a29…`, contrat
  voxel-exact CPU/GPU). Test : `tests/test_backend_auto.py`.

## v0.2.2 — 2026-11-19

Publiée le 2026-11-19 : PR #7 mergée dans `main` (merge commit `767bcd6`),
tag `v0.2.2`. Hash du deck de démo inchangé (`e1316a29…`, CPU et GPU).

- **Tiling + halos au-delà de la VRAM** (cf. ARCHITECTURE.md §5, objectif 10⁹
  voxels) dans `ops/gpu.py` : budget VRAM auto-détecté (`cuInit` +
  `cuMemGetInfo`, env `VOXELFAB_GPU_VRAM_BYTES`, `VOXELFAB_FORCE_TILED=1`
  pour forcer). Si la grille tient : mode device-résident inchangé. Sinon :
  exécution tuilée, `grid.data` reste le backing store hôte —
  - ops « stencil » (substrate, deposit conformal, etch isotropic,
    etch_selective isotropic) : **tuiles z** avec halo exact = rayon r
    (hors grille : border_value CPU, dilate=0/erode=1) ; halos lus dans un
    **snapshot d'entrée** (sinon les tuiles déjà écrites contaminent le halo
    des suivantes) ;
  - ops « colonne » (deposit/etch directionnels, etch_selective dir, cmp) :
    **tuiles (x, y) à z complet, sans halo** (colonnes indépendantes), masque
    découpé par tuile ;
  - propagation globale (grow, epitaxy seed/crystal, etch_crystal, fill) :
    erreur claire en mode tuilé (à tuiler plus tard : itération à point fixe).
  - tailles de tuiles fixées une fois par (nx, ny)/nz (caches `_FH_CACHE` /
    `_TILE_CACHE`) : sinon chaque op aurait sa propre shape de ctx et
    recompilerait tous les noyaux (les noyaux en closure sont compilés par
    instance de ctx) — cause d'un hang à 10⁹ avant ce fix.
  - `vf bench 1000 --backend all` fonctionne à **10⁹ vx** (GTX 1660, 6 Go,
    2026-11-19), mode tuilé auto (hors VRAM) — ms CPU / GPU, × = CPU/GPU :
    conformal t=5 : 9213/2714 ×3,4 ; t=20 : 10348/5693 ×1,8 ;
    dir t=10 : 4286/4276 ×1,0 ; iso t=5 : 9234/2854 ×3,2 ;
    iso t=20 : 10440/6566 ×1,6 ; etch dir t=30 : 312/3261 ×0,1 ;
    cmp : 4066/5587 ×0,7 ; height_map : 1102/985 ×1,1.
    Lecture : le tiling rend 10⁹ possible ; le GPU gagne sur les stencils
    (calcul dense), mais les ops « colonne » (etch dir, cmp) paient 2×N
    octets de PCIe par étape côté GPU, ce que le CPU (déjà en RAM) ne paie
    pas → parité à légèrement perdant à cette échelle sur cette carte.
  - Test `test_tiled_mode_matches_cpu` : octets identiques au CPU en mode
    tuilé forcé (budget VRAM minuscule). 53 tests passent.
- Coût connu en mode tuilé : ~1 compile de noyaux par shape de tuile (2 shapes
  typiques à 10⁹, ~15 noyaux) + copie snapshot hôte (N octets, stencils
  uniquement) + H2D/D2H (2×N) par étape.

## v0.2.1 — 2026-11-19

Tag `v0.2.1` sur `main` (`94a443c`). Hash du deck de démo inchangé
(`e1316a29…`, vérifié CPU et GPU).

- **Mode device-résident** (backend GPU) : `ops/gpu.py` gagne
  `bind_resident(grid)` / `release_resident(grid)` ; `DeckRunner.run()`
  l'active automatiquement pour `--backend gpu` (un seul H2D en entrée,
  un seul D2H avant les exports — la grille reste dans le `ti.field` pendant
  tout le deck). Le wrapper `_gpu_op` saute alors ses syncs et fait
  `ti.sync()` par étape (les lancements Taichi sont asynchrones : sans ça,
  le chrono ne mesure que le launch). Supprime ~2×N octets de PCIe par
  étape. Test `test_resident_mode_matches_per_op_mode` : octets identiques
  au mode par-op et au CPU. `vf bench` mesure en mode résident côté GPU.
- Mesures 2026-11-19 (GTX 1660, noyaux chauds, mode résident) :
  256³ ×8,0–×153,5 ; 464³ (100 M vx) **×8,0–×39,8** (conformal t=5 ×31,
  etch dir t=30 ×39,8, iso t=5 ×21,8) — dans la fourchette 10–50× de la
  roadmap à 10⁸ vx. (Avant, sync par étape : ×2,3–×7,9 à 464³.)
- Deck démo (0,46 M vx) : hash inchangé `e1316a29…` des deux backends.

## v0.2.0 — 2026-11-19

Publiée le 2026-11-19 : PR #6 mergée dans `main` (merge commit `2dd3a36`),
tag `v0.2.0`. Le hash du deck de démo reste `e1316a29…` (inchangé, vérifié
côté CPU **et** GPU).

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
