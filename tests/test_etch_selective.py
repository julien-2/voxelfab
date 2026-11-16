"""
Tests de l'opération etch_selective (gravure à sélectivité différentielle).

    uv run pytest tests/test_etch_selective.py -v

Inclut deux tests d'équivalence de contrat : au taux uniforme 1.0,
etch_selective doit reproduire exactement etch (directional et isotropic).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DeckRunner, VoxelGrid  # noqa: E402
from voxelfab.deck import DeckError  # noqa: E402
from voxelfab.ops import get_step  # noqa: E402


def _stack(nx=10, ny=10, nz=40, si_th=10, sio2_th=20):
    """Si sur z in [0, si_th), SiO2 sur z in [si_th, si_th+sio2_th), air au-dessus."""
    g = VoxelGrid(nx, ny, nz, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :si_th] = Si
    g.data[:, :, si_th:si_th + sio2_th] = SiO2
    return g, Si, SiO2


def test_directional_differential_rate():
    """SiO2 3× plus vite que Si : depth 5 -> 15 voxels de SiO2, Si intact."""
    g, Si, SiO2 = _stack()
    etch = get_step("cpu", "etch_selective", "directional")
    etch(g, 5, {SiO2: 3.0, Si: 1.0})
    # colonne centrale : 20 voxels de SiO2 disponibles, 15 retirés
    assert g.data[5, 5, 29] == 0          # sommet gravé
    assert g.data[5, 5, 15] == 0          # 15e voxel retiré (z 29..15)
    assert g.data[5, 5, 14] == SiO2       # le SiO2 restant
    assert g.data[5, 5, 9] == Si          # Si intact (budget épuisé)
    assert (g.data == Si).sum() == 10 * 10 * 10


def test_directional_stops_on_unlisted_material():
    """Si absent de rates : il bloque comme un matériau d'arrêt."""
    g, Si, SiO2 = _stack()
    etch = get_step("cpu", "etch_selective", "directional")
    etch(g, 100, {SiO2: 1.0})             # budget énorme, Si non listé
    assert g.data[5, 5, 10] == 0          # tout le SiO2 est parti
    assert g.data[5, 5, 9] == Si          # Si intact


def test_directional_until_material():
    g, Si, SiO2 = _stack()
    etch = get_step("cpu", "etch_selective", "directional")
    etch(g, 100, {SiO2: 1.0, Si: 1.0}, until_material=Si)
    assert g.data[5, 5, 10] == 0          # SiO2 entièrement gravé
    assert g.data[5, 5, 9] == Si          # arrêt sur Si


def test_equivalence_with_etch_directional():
    """Taux uniforme 1.0 == etch directional (contrat)."""
    g1, Si, SiO2 = _stack()
    g2, _, _ = _stack()
    get_step("cpu", "etch_selective", "directional")(g1, 7, {SiO2: 1.0})
    get_step("cpu", "etch", "directional")(g2, 7, materials=(SiO2,))
    assert g1.data.tobytes() == g2.data.tobytes()


def test_equivalence_with_etch_isotropic():
    """Taux uniforme 1.0 == etch isotropic (contrat)."""
    g1, Si, SiO2 = _stack()
    g2, _, _ = _stack()
    get_step("cpu", "etch_selective", "isotropic")(g1, 3, {SiO2: 1.0})
    get_step("cpu", "etch", "isotropic")(g2, 3, materials=(SiO2,))
    assert g1.data.tobytes() == g2.data.tobytes()


def test_isotropic_differential_rate():
    """SiO2 rate 2.0, depth 3 -> rayon d'érosion 6 pour le SiO2 seul."""
    g = VoxelGrid(20, 20, 30, pitch=1.0)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :20] = SiO2              # bloc plein, air au-dessus
    etch = get_step("cpu", "etch_selective", "isotropic")
    etch(g, 3, {SiO2: 2.0})               # rayon 6 pour SiO2
    assert g.data[10, 10, 19] == 0        # gravé (6 voxels depuis l'air)
    assert g.data[10, 10, 14] == 0        # z in [14, 20) gravé
    assert g.data[10, 10, 13] == SiO2     # préservé


def test_mask_limits_both_modes():
    for mode in ("directional", "isotropic"):
        g, Si, SiO2 = _stack()
        mask = np.zeros((g.nx, g.ny), bool)
        mask[:5, :] = True                # moitié x < 5 ouverte seulement
        etch = get_step("cpu", "etch_selective", mode)
        etch(g, 5, {SiO2: 1.0}, mask=mask)
        assert g.data[2, 5, 25] == 0      # côté ouvert : gravé
        assert g.data[7, 5, 25] == SiO2   # côté masqué : intact


def test_deck_validation_errors():
    base = {"version": 1,
            "grid": {"pitch_nm": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "materials": {"SiO2": {"color": "#ffffff"}}}
    # rates manquant
    deck = dict(base, steps=[{"op": "etch_selective", "depth_nm": 2}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
    # taux <= 0
    deck = dict(base, steps=[{"op": "etch_selective", "depth_nm": 2,
                              "rates": {"SiO2": 0}}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
    # matériau inconnu
    deck = dict(base, steps=[{"op": "etch_selective", "depth_nm": 2,
                              "rates": {"Au": 1.0}}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
