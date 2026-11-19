# Handoff — voxelfab : backend GPU Taichi (v0.2) implémenté + validé

**Date : 2026-11-19** (travail fait le 19 nov 2026, ~09:15–12:00 UTC+1)
**Branche : `v0.2-taichi-gpu`**, commit `8e167c0` — poussé sur
`https://github.com/julien-2/voxelfab` (`main` inchangé : `fb4ec5b`).
PR à ouvrir si besoin : https://github.com/julien-2/voxelfab/pull/new/v0.2-taichi-gpu

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
- Perf mesurées sur ce PC (voir §3), noyaux Taichi chauds :

| étape | 256³ (16,8 M vx) | 464³ (100 M vx) | 640³ (262 M vx) |
|---|---|---|---|
| deposit conformal t=5 | ×7–8 | ×7,4 | ×7,2 |
| deposit conformal t=20 | ×7–8 | ×6,6 | ×5,4 |
| deposit directional t=10 | ×4 | ×3,7 | ×3,6 |
| etch isotropic t=5 | ×6,7 | ×6,7 | ×6,6 |
| etch isotropic t=20 | ×5 | ×4,9 | ×4,2 |
| etch directional t=30 | ×13 | ×2,2 | ×1,0 |
| cmp | ×3 | ×2,5 | ×2,7 |

Lecture : le GPU gagne sur **toutes** les étapes ; le cap « 10–50× à 10⁸ vx »
de la roadmap n'est pas atteint sur une GTX 1660 (carte milieu de gamme,
192 Go/s) — on mesure ×2,2–×7,4 à 464³. Deux overheads mesurés : JIT Taichi
one-shot (~0,3 s/noyau, par shape de grille et par processus) et le transfert
PCIe hôte↔device à chaque étape (~2×N octets, ~40 ms à 100 M vx).

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

## 6. État git

- Branche locale + distante : `v0.2-taichi-gpu` @ `8e167c0`
  (7 fichiers, +1479/−26), tracking `origin/v0.2-taichi-gpu`, tree propre.
- Commit signé « Generated by Mistral Vibe / Co-Authored-By: Mistral Vibe ».
- Identité git locale du clone : `julien-2 <julien-2@users.noreply.github.com>`.

## 7. Prochaines étapes possibles

1. Ouvrir la PR `v0.2-taichi-gpu` → `main` (ou merger), puis taguer `v0.2.0`
   et mettre à jour `pyproject.toml` version si souhaité.
2. **Mode device-resident** : garder `grid.data` sur le device pendant tout
   un run (sync unique en fin de deck) pour supprimer ~2×N octets de PCIe
   par étape — c'est le levier principal pour viser les 10–50×.
3. Bench nightly (10⁷/10⁸ vx) comme prévu au quality gate §8 d'ARCHITECTURE.md.
4. Optionnel : `vf run --backend auto` (GPU si taichi présent, sinon CPU),
   comme documenté mais pas encore implémenté.

## 8. Fichiers clés à lire en priorité

- `src/voxelfab/ops/base.py` — contrat sémantique de référence (à lire avant
  toute modif d'op)
- `src/voxelfab/ops/cpu.py` — implémentation CPU de référence (NumPy/Numba)
- `src/voxelfab/ops/gpu.py` — backend Taichi (cette PR)
- `tests/test_equiv_gpu.py` — tests d'équivalence (release gate)
- `docs/ARCHITECTURE.md` §5 — design notes GPU + §8 quality gates
- `src/voxelfab/bench.py`, `src/voxelfab/cli.py` — `vf bench --backend all`
