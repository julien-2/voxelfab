"""
Tests d'équivalence CPU <-> GPU (backend Taichi) — v0.2.

    uv run pytest tests/test_equiv_gpu.py -v

CONTRAT (ops/base.py) : le backend GPU doit produire une grille identique
octet pour octet au backend CPU pour les mêmes entrées. Chaque test applique
la même étape à deux grilles identiques (une par backend) et compare les
octets ; le dernier test rejoue tout le deck de démo sur les deux backends
et compare les structure_sha256.

Ignoré automatiquement si l'extra [gpu] (taichi) n'est pas installé.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless : aucun rendu pendant le test

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DEMO_DECK, DeckRunner, VoxelGrid  # noqa: E402
from voxelfab.ops import available_backends, get_step  # noqa: E402

pytestmark = pytest.mark.skipif(
    "gpu" not in available_backends(),
    reason="extra [gpu] (taichi) non installée : uv pip install \".[gpu]\"")

NX, NY, NZ = 24, 20, 32


def _build():
    """Grille de test avec marches et bords dans les deux axes (pour
    solliciter les facettes et les masques)."""
    g = VoxelGrid(NX, NY, NZ, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    SiN = g.add_material("SiN", "#8bc34a")
    Ni = g.add_material("Ni", "#888888")
    g.data[:, :, :12] = Si
    g.data[:, :, 12:18] = SiO2
    g.data[4:20, 4:16, 12:26] = Si        # mesa perçant l'oxyde
    g.data[8:12, 8:12, 18:26] = 0         # tranchée dans le mesa
    g.data[1:3, 1:3, 26:30] = SiN         # plot isolé
    g.data[18:23, 15:19, 26:30] = Ni      # bloc métal (siliciuration)
    return g, dict(Si=Si, SiO2=SiO2, SiN=SiN, Ni=Ni)


def _check(op, mode, *args, **kwargs):
    """Applique l'étape (op, mode) sur deux grilles identiques, une par
    backend, et exige des octets identiques."""
    fn_cpu = get_step("cpu", op, mode)
    fn_gpu = get_step("gpu", op, mode)
    g1, _ = _build()
    g2, _ = _build()
    fn_cpu(g1, *args, **kwargs)
    fn_gpu(g2, *args, **kwargs)
    assert g1.data.tobytes() == g2.data.tobytes(), \
        f"divergence CPU/GPU pour ({op!r}, {mode!r}) {args} {kwargs}"


def _mask_rect(x0=0.2, x1=0.8, y0=0.2, y1=0.8):
    m = np.zeros((NX, NY), bool)
    m[int(NX * x0):int(NX * x1), int(NY * y0):int(NY * y1)] = True
    return m


def test_substrate():
    _check("substrate", None, 1, 7)       # material id 1 = SiO2 dans _build


def test_deposit_conformal():
    for t in (1, 3, 5, 9):                 # t > 4 : boule L1 (itérations CPU)
        _check("deposit", "conformal", 2, t)


def test_deposit_directional():
    _check("deposit", "directional", 3, 6)
    _check("deposit", "directional", 1, 6, on_materials=(1,))  # sélectif


def test_etch_isotropic():
    _check("etch", "isotropic", 2)
    _check("etch", "isotropic", 6)
    _check("etch", "isotropic", 3, materials=(2,), mask=_mask_rect())


def test_etch_directional():
    _check("etch", "directional", 8)
    _check("etch", "directional", 8, materials=(1,), mask=_mask_rect())
    _check("etch", "directional", 30, until_material=1, mask=_mask_rect())


def test_etch_selective_directional():
    _, m = _build()
    _check("etch_selective", "directional", 8, {m["SiO2"]: 3.0, m["Si"]: 1.0})
    _check("etch_selective", "directional", 10, {m["SiO2"]: 1.0},
           until_material=m["Si"], mask=_mask_rect())


def test_etch_selective_isotropic():
    _, m = _build()
    _check("etch_selective", "isotropic", 6, {m["SiO2"]: 2.0, m["Si"]: 0.5},
           mask=_mask_rect())


def test_grow():
    _, m = _build()
    _check("grow", None, 4, m["Si"], m["SiO2"], expansion=2.27)
    _check("grow", None, 3, m["Si"], m["SiO2"], expansion=2.0,
           mask=_mask_rect())
    _check("grow", None, 9, m["Si"], m["Ni"], expansion=2.54 / 1.83,
           into=(m["Ni"],))                # siliciuration (into)


def test_epitaxy_seed():
    _, m = _build()
    _check("epitaxy", "seed", 5, m["Si"], m["Si"])
    _check("epitaxy", "seed", 6, m["Si"], m["Si"], mask=_mask_rect())


def test_epitaxy_crystal():
    _, m = _build()
    _check("epitaxy", "crystal", 12, m["Si"], seed=m["Si"])
    _check("epitaxy", "crystal", 8, m["Si"], orientation="110", seed=m["Si"],
           mask=_mask_rect())
    _check("epitaxy", "crystal", 6, m["Si"], orientation="111",
           facet_rates={"100": 1.0, "110": 0.7, "111": 0.3}, seed=m["Si"])


def test_etch_crystal():
    _, m = _build()
    _check("etch_crystal", None, 10, m["Si"])
    _check("etch_crystal", None, 8, m["Si"], orientation="110",
           mask=_mask_rect())
    _check("etch_crystal", None, 6, m["Si"],
           facet_rates={"100": 1.0, "110": 1.6, "111": 0.1})


def test_fill():
    _check("fill", None, 2, 20)


def test_cmp():
    _check("cmp", None, 20)
    _check("cmp", None, 20, stop_on=1)


def test_resident_mode_matches_per_op_mode():
    """Le mode device-résident (bind/release autour d'une séquence d'étapes)
    doit produire exactement les mêmes octets que le mode par-op (sync
    hôte↔device à chaque étape)."""
    from voxelfab.ops.gpu import bind_resident, release_resident
    seq = [
        ("deposit", "conformal", (2, 5), {}),
        ("etch", "directional", (8,), {"mask": _mask_rect()}),
        ("etch", "isotropic", (3,), {"materials": (2,)}),
        ("deposit", "directional", (1, 6), {"on_materials": (1,)}),
        ("cmp", None, (20,), {"stop_on": 1}),
        ("fill", None, (2, 20), {}),
    ]
    g_batch, _ = _build()      # mode par-op (défaut)
    g_res, _ = _build()        # mode device-résident
    for op, mode, args, kw in seq:
        get_step("gpu", op, mode)(g_batch, *args, **kw)
    bind_resident(g_res)
    try:
        for op, mode, args, kw in seq:
            get_step("gpu", op, mode)(g_res, *args, **kw)
    finally:
        release_resident(g_res)
    assert g_batch.data.tobytes() == g_res.data.tobytes(), \
        "mode device-résident != mode par-op"
    # et les deux valent le CPU
    g_cpu, _ = _build()
    for op, mode, args, kw in seq:
        get_step("cpu", op, mode)(g_cpu, *args, **kw)
    assert g_cpu.data.tobytes() == g_res.data.tobytes()


def test_tiled_mode_matches_cpu(monkeypatch):
    """Mode tuilé (grille hors VRAM, halos exacts) == CPU, octet pour octet.
    Budget VRAM minuscule + VOXELFAB_FORCE_TILED=1 -> beaucoup de tuiles
    (zslab avec halo pour les stencils, tuiles xy pour les colonnes)."""
    from voxelfab.ops import gpu as gpu_mod
    monkeypatch.setenv("VOXELFAB_FORCE_TILED", "1")
    monkeypatch.setenv("VOXELFAB_GPU_VRAM_BYTES", str(60 * 1024))
    gpu_mod._BUDGET = None          # recalculé avec l'env du test
    _, m = _build()
    seq = [
        ("substrate", None, (2, 9), {}),
        ("deposit", "conformal", (2, 5), {}),          # boule r=5 <= ... r>4
        ("deposit", "conformal", (3, 9), {}),          # croix itérée
        ("deposit", "directional", (1, 6), {"on_materials": (1,)}),
        ("etch", "isotropic", (3,), {"materials": (2,), "mask": _mask_rect()}),
        ("etch", "isotropic", (6,), {}),               # érosion r=6
        ("etch", "directional", (8,), {"until_material": 1,
                                       "mask": _mask_rect()}),
        ("etch_selective", "directional", (6,),
         {"rates": {m["SiO2"]: 2.0, m["Si"]: 0.5}}),
        ("etch_selective", "isotropic", (5,),
         {"rates": {m["SiO2"]: 2.0, m["Si"]: 0.5}, "mask": _mask_rect()}),
        ("cmp", None, (20,), {"stop_on": 1}),
    ]
    g_cpu, _ = _build()
    g_tiled, _ = _build()
    for op, mode, args, kw in seq:
        get_step("cpu", op, mode)(g_cpu, *args, **kw)
        get_step("gpu", op, mode)(g_tiled, *args, **kw)   # tuilé (env)
    assert g_cpu.data.tobytes() == g_tiled.data.tobytes(), \
        "mode tuilé != CPU"
    gpu_mod._BUDGET = None          # restore pour les autres tests


def test_demo_deck_identical_cpu_gpu(tmp_path):
    """Deck de démo complet (17 étapes) : structure_sha256 identique."""
    deck = yaml.safe_load(DEMO_DECK)
    deck["outputs"] = []                  # pas de rendu ni d'export
    hashes = {}
    for be in ("cpu", "gpu"):
        out = tmp_path / be
        runner = DeckRunner(deck, deck_path=ROOT / "decks" / "demo_deck.yaml",
                            outdir=out, backend=be)
        runner.run()
        report = json.loads((out / "report.json").read_text())
        hashes[be] = report["structure_sha256"]
        assert report["provenance"]["backend"] == be
    assert hashes["cpu"] == hashes["gpu"], \
        f"deck démo : hash CPU {hashes['cpu'][:16]} != GPU {hashes['gpu'][:16]}"
