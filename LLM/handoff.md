# Handoff — voxelfab : backend GPU Taichi (v0.2) implémenté + validé

**Dernière mise à jour : 2026-11-20 (UTC)** — convention de mise à jour en §0.
**Branche d'origine : `v0.2-taichi-gpu`**, commit `8e167c0`. **Ce fichier vit
sur `main`** (intégré via PR #6 mergée le 2026-11-19 ; puis v0.2.1
device-résident et v0.2.2 tiling via PR #7, tags `v0.2.0`–`v0.2.2`).

## 0. Convention de mise à jour (multi-PC)

On travaille sur **plusieurs machines** : ce handoff est le point de passage
obligatoire entre deux sessions, quelle que soit la machine.

- Toute session qui reprend le travail **ajoute une entrée datée**
  (AAAA-MM-JJ, UTC) en tête du journal ci-dessous : machine, branche,
  ce qui a été fait, résultats (tests, hash, perfs), blocages.
- Ne pas réécrire l'historique : les entrées s'ajoutent, on ne les efface pas.
- En fin de session, commit + push de ce fichier sur `main` avant de
  changer de PC.

### Journal des mises à jour

- **2026-11-20** (ce PC, `main`) — `--backend auto` implémenté
  (`ops.resolve_backend`, résolu dans `DeckRunner` ; `vf run|demo --backend
  auto` ; test `tests/test_backend_auto.py`). `auto` -> `gpu` si taichi
  installé, sinon `cpu`. Sur ce PC : taichi présent, `auto` -> `gpu`, mais
  `ti.init` tombe sur `arch=x64` (pas de CUDA) -> backend gpu en fallback
  CPU Taichi (lent). Note : « auto » teste la présence de taichi, pas celle
  d'un GPU NVIDIA. `uv run pytest tests/ -q` -> 55 passed, hash golden
  `e1316a29…` OK des deux côtés.
- **2026-11-20** (ce PC, `main`) — mise en place de cette convention
  multi-PC (section §0 + journal daté). État constaté : PR #6 et #7 déjà
  mergées le 2026-11-19, tags `v0.2.0`–`v0.2.2` posés.
- **2026-11-19** (PC GTX 1660, branche `v0.2-taichi-gpu`, ~09:15–12:00
  UTC+1) — implémentation + validation du backend GPU Taichi (détail dans
  les §1–§8 ci-dessous), puis v0.2.1 device-résident et v0.2.2 tiling le
  même jour.

## 1. Contexte et ce qui a été fait

Au départ, `main` (v0.1.0) **n'avait aucun code Taichi** : l'extra `[gpu]`
était déclaré dans `pyproject.toml` et le backend GPU était seulement
documenté (roadmap v0.2 + « Taichi design notes » dans `docs/ARCHITECTURE.md`
§5, contrat voxel-exact dans `src/voxelfab/ops/base.py`). Le travail a donc
été d'**implémenter** le backend GPU selon ce contrat, puis de le **valider
sur ce PC** (GTX 1660) :

- `src/voxelfab/ops/gpu.py` (nouveau, ~1200 lignes) : backend `"gpu"` Taichi
  CUDA, **les 12 opérations** : substrate, deposit conformal/directionnel,
  etch isotrope/directionnel, etch_selective (directional + isotropic),
  grow, epitaxy seed/crystal, etch_crystal, fill, cmp.
- `tests/test_equiv_gpu.py` (nouveau) : 14 tests d'équivalence CPU↔GPU
  (chaque op comparée octet par octet + deck démo complet rejoué sur les
  deux backends). Skippés auto si `taichi` absent.
- `src/voxelfab/ops/__init__.py`, `bench.py`, `cli.py` : `vf run/demo
  --backend gpu` ; `vf bench [n] --backend all` imprime le comparatif
  cpu vs gpu (warm-up JIT non chronométré côté GPU).
- `CHANGELOG.md` (entrée v0.2 avec mesures), `README.md` (install `[gpu]`,
  quickstart `--backend`).

## 2. Résultats de validation (à re-vérifier sur une autre machine)

- `python -m pytest tests/ -q` → **51 passed** (37 existants + 14 nouveaux).
- `vf demo --backend cpu` vs `--backend gpu` → `structure_sha256` identique
  **et égal au golden** commité : `e1316a2970a1e6f38612e6ee98d954123e5fe173f2acd06053d8cb63283743c1` ;
  grilles `structure.h5` byte-identiques entre backends.
- Perf mesurées sur ce PC (voir §3), **noyaux Taichi chauds** (warm-up JIT
  exclu), `vf bench n --backend all`, 2026-11-19 — accélération × = CPU/GPU :

| étape | 128³ (2,1 M vx) | 256³ (16,8 M vx) | 464³ (100 M vx) | 640³ (262 M vx) |
|---|---|---|---|---|
| deposit conformal t=5 | ×4,6 | ×7,8 | ×7,9 | ×7,2 |
| deposit conformal t=20 | ×3,5 | ×8,0 | ×6,6 | ×5,4 |
| deposit directional t=10 | ×2,3 | ×4,1 | ×4,2 | ×3,6 |
| etch isotropic t=5 | ×4,5 | ×5,8 | ×7,2 | ×6,6 |
| etch isotropic t=20 | ×3,4 | ×4,8 | ×5,0 | ×4,2 |
| etch directional t=30 | ×61,2 | ×13,3 | ×2,3 | ×1,0 |
| cmp | ×2,2 | ×2,6 | ×2,7 | ×2,7 |
| height_map (hôte des 2 côtés) | ×1,0 | ×1,0 | ×1,0 | ×1,1 |

Temps absolus (ms, CPU / GPU) à 464³ = 100 M vx : conformal t=5 → 984/124,
conformal t=20 → 1217/183, directional t=10 → 474/113, iso t=5 → 1017/141,
iso t=20 → 1212/242, dir t=30 → 254/110, cmp → 401/146, height_map → 116/112.

Deck démo complet `vf demo` (120×60×64 = 0,46 M vx, 17 étapes, process complet,
meilleur de 3 runs) : wall-clock **CPU 3,18 s vs GPU 4,61 s** ; somme des étapes
0,38 s (CPU) vs 1,86 s (GPU, JIT ~1,3 s + transferts PCIe inclus). Hash identique
des deux côtés (`e1316a29…` = golden).

Lecture : à partir de ~2 M de voxels le GPU gagne sur **toutes** les étapes
(×2,2–×13,3 à 256³, ×2,3–×7,9 à 464³). En dessous (~0,5 M vx), le GPU perd en
wall-clock à cause du JIT one-shot et du transfert PCIe par étape. Le cap
« 10–50× à 10⁸ vx » de la roadmap n'est pas atteint sur une GTX 1660 (192 Go/s)
: on mesure ×2,3–×7,9 à 100 M vx. Leviers : mode device-résident pendant tout
un run (supprime ~2×N octets de PCIe par étape), GPU plus récent.

## 3. Environnement de référence (PC de test, 2026-11-19)

- Windows, **NVIDIA GeForce GTX 1660 (6 Go)**, driver 591.44, CUDA 13.1
- Python 3.10.10 (`C:\Users\emy&jul\AppData\Local\Programs\Python\Python310`)
- `pip install -e ".[dev,bench,gpu]"` → numpy, scipy, matplotlib, pyyaml, h5py,
  numba, pytest, **taichi 1.7.4** (backend CUDA OK, f64 OK)
- CPU : 4 cœurs (Numba `parallel=True` dessus)
- Dépôt cloné dans `C:\Users\emy&jul\Documents\Mistral\voxelfab`

## 4. Reprendre sur un autre ordinateur (Mistral ou autre LLM)

```bash
git clone https://github.com/julien-2/voxelfab.git
cd voxelfab
git checkout v0.2-taichi-gpu          # branche du backend GPU
python -m pip install -e ".[dev,bench,gpu]"   # taichi requis pour le GPU
python -m pytest tests/ -q            # 51 passed attendu (14 skippés sans taichi)
python -m voxelfab.cli bench 464 --backend all   # comparatif cpu vs gpu
python -m voxelfab.cli demo -o out_cpu --backend cpu
python -m voxelfab.cli demo -o out_gpu --backend gpu
# structure_sha256 doit être e1316a29... des deux côtés (golden)
```

Prérequis GPU : carte NVIDIA avec pilote récent (Taichi embarque son propre
runtime CUDA). Sans GPU, `ti.init` fallback sur `ti.cpu` dans `ops/gpu.py`
(lent mais fonctionnel pour les tests d'équivalence).

## 5. Architecture du backend GPU — décisions et pièges (à ne pas casser)

- **Contrat** (`ops/base.py`) : le GPU doit produire une grille **identique
  octet pour octet** au CPU → `structure_sha256` identique. Porté par
  `tests/test_equiv_gpu.py`.
- Un `_GpuCtx` **par shape de grille** (cache `_CTX_CACHE`) : champs `ti.field`
  (u8 3D ×6, i32 2D ×2, f64 2D ×4, flag scalaire) + **noyaux définis en
  closure** sur ces champs (Taichi capture les champs à la compilation).
- Chaque étape fait `ctx.d.from_numpy(grid.data)` en entrée et
  `grid.data = ctx.d.to_numpy()` en sortie : `grid.data` (NumPy) reste la
  source de vérité, HDF5/métrologie/vues inchangés.
- Déterminisme : réductions min/max uniquement, `ti.atomic_add/or` entiers
  pour les compteurs, **f64 partout** (`ti.init(default_fp=ti.f64)`) avec les
  mêmes opérations dans le même ordre que NumPy (`_rint` = round-half-even
  réimplémenté, `_refl` = bord scipy 'reflect', min-scans séquentiels
  bit-identiques à `np.minimum.accumulate`).
- Morphismes : `r <= 4` → un noyau avec la boule euclidienne exacte
  (dx²+dy²+dz² ≤ r²+r) ; `r > 4` → **r itérations de la croix 6-voisine**
  (ping-pong w1↔w2, w1 sauvegardé dans w5) — équivalent à
  `binary_dilation/erosion(iterations=r)` CPU, et ~1000× plus rapide qu'une
  boucle sur tous les offsets de la boule L1.
- `epitaxy crystal` / `etch_crystal` : plans de facettes calculés côté hôte
  (`_crystal_planes` réutilisé depuis `cpu.py`), min-pyramide séparable en
  noyaux par ligne ; **deux noyaux séparés** pour surface exposée puis bords
  (sinon course entre threads sur les champs 2D). Etch : le terme parallèle
  et les ancres sont des **profondeurs** (`v*t*inv_nz`, sans `zseed`) —
  ne pas confondre avec epitaxy (hauteur = `zseed + …`).

### Pièges Taichi 1.7.4 rencontrés (ne pas régresser)
1. Les noyaux doivent vivre dans un **vrai fichier .py** — `python -c` échoue
   (Taichi lit le source via `inspect`).
2. **Ne PAS mettre `from __future__ import annotations`** dans `gpu.py` :
   les annotations des noyaux doivent être de vrais objets (`ti.f64`),
   pas des chaînes (PEP 563 casse `extract_arguments`).
3. `range()` dans un noyau accepte **1 ou 2 arguments max** — pas de
   `range(a, -1, -1)` ; utiliser `for kk in range(n): k = n-1-kk`.
4. `ti.f64(littéral_entier)` est interdit → `ti.cast(nz, ti.f64)`.
5. Rebinder une variable globale `ti.field` après compilation ne marche pas
   (le noyau garde l'ancien champ) → d'où le pattern closure-par-ctx.
6. Champs passés en argument de noyau : `ti.types.field` n'existe pas en
   1.7.4 → closures ou `ti.types.ndarray(...)` (qui resynchronise
   hôte↔device à chaque appel — éviter en boucle).

## 6. État git (mis à jour 2026-11-19 après merge)

- **PR #6 `v0.2-taichi-gpu` → `main` mergée** (merge commit `2dd3a36`, PR
  fermée `merged: true`) : https://github.com/julien-2/voxelfab/pull/6
- **Tags `v0.2.0`, `v0.2.1`, `v0.2.2`** posés sur `main` ; versions
  `0.1.0 → 0.2.0 → 0.2.1 → 0.2.2` dans `pyproject.toml` +
  `src/voxelfab/_version.py` (+ test_h5, CHANGELOG, README).
  v0.2.1 = mode device-résident (`94a443c`) ; v0.2.2 = tiling + halos 10⁹
  (PR #7, merge `767bcd6`, commit `b909011`).
- Historique : `8e167c0` feat backend GPU · `eba491d` + `9ba2179` handoff ·
  merge `2dd3a36` · puis commit de release v0.2.0.
- Commits signés « Generated by Mistral Vibe / Co-Authored-By: Mistral Vibe ».
- Identité git locale du clone : `julien-2 <julien-2@users.noreply.github.com>`.

## 7. Prochaines étapes possibles

1. ~~Ouvrir la PR et taguer v0.2.0~~ **fait le 2026-11-19** (PR #6 mergée,
   tag `v0.2.0`).
2. ~~Mode device-resident~~ **implémenté le 2026-11-19** : `bind_resident()` /
   `release_resident()` dans `ops/gpu.py`, activé automatiquement par
   `DeckRunner.run()` pour `--backend gpu` (un seul H2D/D2H par run au lieu
   d'un par étape). **Piège** : les lancements de noyaux Taichi sont
   asynchrones — le wrapper fait `ti.sync()` par étape en mode résident,
   sinon le chrono ne mesure que le launch (~2 ms) et les speedups
   affichés sont fictifs (×100+). Avec sync : 464³ (100 M vx) ×8,0–×39,8,
   256³ ×8,0–×153,5 (GTX 1660, noyaux chauds) — la fourchette 10–50× de la
   roadmap est atteinte à 10⁸ vx. Test : `test_resident_mode_matches_per_op_mode`
   (octets identiques au mode par-op et au CPU). Bench : `vf bench` tourne en
   mode résident côté GPU.
3. ~~Tiling + halos pour viser 10⁹ vx~~ **implémenté le 2026-11-19** dans
   `ops/gpu.py` : budget VRAM auto (`cuInit`+`cuMemGetInfo`, env
   `VOXELFAB_GPU_VRAM_BYTES`, `VOXELFAB_FORCE_TILED=1` pour forcer). Grille qui
   tient → device-résident ; sinon tuilé, `grid.data` = backing store hôte :
   - stencils (substrate, deposit conformal, etch iso, etch_selective iso) :
     **tuiles z + halo exact = r**, hors grille = border_value CPU, halos lus
     dans un **snapshot d'entrée** (`grid.data.copy()`, sinon les tuiles déjà
     écrites contaminent le halo des suivantes — bug trouvé et corrigé) ;
   - ops colonne (deposit/etch dir, etch_selective dir, cmp) : **tuiles (x,y)
     à z complet, sans halo**, masque découpé par tuile ;
   - tailles de tuiles **fixées une fois** (`_FH_CACHE`/`_TILE_CACHE`) : sinon
     chaque op change de shape de ctx et recompile tous les noyaux (hang à
     10⁹ avant ce fix — noyaux en closure compilés par instance de ctx) ;
   - grow / epitaxy seed|crystal / etch_crystal / fill (propagation globale)
     : erreur claire en mode tuilé — à tuiler par itération à point fixe.
   - Résultats 10⁹ vx (1000³, GTX 1660, 2026-11-19), `vf bench 1000
     --backend all`, ms CPU/GPU : conformal t=5 9213/2714, t=20 10348/5693,
     dir t=10 4286/4276, iso t=5 9234/2854, t=20 10440/6566, etch dir t=30
     312/3261, cmp 4066/5587, height_map 1102/985. Le GPU gagne sur les
     stencils (×1,6–×3,4) ; les ops colonne paient 2×N octets PCIe/op côté
     GPU → parité/perte (×0,1–×1,0) à cette échelle sur cette carte.
   - Test : `test_tiled_mode_matches_cpu` (octets = CPU en tuilé forcé).
4. Bench nightly (10⁷/10⁸ vx) comme prévu au quality gate §8 d'ARCHITECTURE.md.
5. ~~`vf run --backend auto`~~ **implémenté le 2026-11-20** :
   `ops.resolve_backend('auto')` -> `gpu` si taichi présent, sinon `cpu`,
   résolu dans `DeckRunner` (report.json/HDF5 gardent le backend effectif) ;
   `vf run|demo --backend auto` ; test `tests/test_backend_auto.py` (deck démo
   en auto, hash golden `e1316a29…` inchangé).

## 8. Fichiers clés à lire en priorité

- `src/voxelfab/ops/base.py` — contrat sémantique de référence (à lire avant
  toute modif d'op)
- `src/voxelfab/ops/cpu.py` — implémentation CPU de référence (NumPy/Numba)
- `src/voxelfab/ops/gpu.py` — backend Taichi (cette PR)
- `tests/test_equiv_gpu.py` — tests d'équivalence (release gate)
- `docs/ARCHITECTURE.md` §5 — design notes GPU + §8 quality gates
- `src/voxelfab/bench.py`, `src/voxelfab/cli.py` — `vf bench --backend all`
