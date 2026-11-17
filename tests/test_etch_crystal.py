"""
Tests de l'opération etch_crystal (gravure anisotrope KOH/TMAH).

    uv run pytest tests/test_etch_crystal.py -v

Sémantique (cf. ops/base.py) : miroir de epitaxy[crystal] — la profondeur
gravie par colonne est le minimum entre le terme vertical (famille la
plus parallèle au wafer, taux de référence 1.0) et les plans lents
partant des bords du matériau exposé (ancrage perpendiculaire à la
montée). Seul `material` est retiré, arrêt au premier voxel qui n'est
pas lui.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DeckRunner, VoxelGrid  # noqa: E402
from voxelfab.deck import DeckError  # noqa: E402
from voxelfab.ops import get_step  # noqa: E402


def _koh_wafer(win="rect", buried=None):
    """Si (z < 20) + masque SiO2 (z 20..23) ; fenêtre carrée [12, 20)²
    (win='rect'), losange |x-16|+|y-16| <= 8 (win='diamond') ou pas de
    fenêtre (win=None). buried=(z0, z1) : oxyde enfoui dans le Si."""
    g = VoxelGrid(32, 32, 40, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :20] = Si
    if win is not None:
        g.data[:, :, 20:24] = SiO2
    if buried is not None:
        g.data[:, :, buried[0]:buried[1] + 1] = SiO2
    if win == "rect":
        mask = np.zeros((32, 32), bool)
        mask[12:20, 12:20] = True
        get_step("cpu", "etch", "directional")(g, 4, mask=mask)
    elif win == "diamond":
        xs = np.arange(32)
        dia = (np.abs(xs[:, None] - 16) + np.abs(xs[None, :] - 16)) <= 8
        get_step("cpu", "etch", "directional")(g, 4, mask=dia)
    return g, Si, SiO2


def _top(g, mid, x, y):
    zs = np.nonzero(g.data[x, y, :] == mid)[0]
    return int(zs.max()) if zs.size else -1


# --------------------------------------------------------------------------
# Géométries de fosse
# --------------------------------------------------------------------------

def test_etch_crystal_rect_window_corner_facets():
    """Fenêtre alignée aux axes sur (100) : arêtes <100> cristallo ->
    parois latérales = plans {110} (rapides avec les taux KOH, ne lient
    pas) -> fond plat limité par les facettes {111} des coins :
    profondeur = rint(0.1*10*sqrt(3) + dist. L1 au coin le plus proche).
    Profil à y = 16 : 5, 6, 7, 8, 8, 7, 6, 5."""
    g, Si, SiO2 = _koh_wafer("rect")
    get_step("cpu", "etch_crystal")(g, 10, Si)
    tops = [_top(g, Si, x, 16) for x in range(12, 20)]
    assert tops == [14, 13, 12, 11, 11, 12, 13, 14]
    assert g.data[16, 16, 11] == Si and g.data[16, 16, 12] == 0   # centre : 8
    assert g.data[12, 16, 14] == Si and g.data[12, 16, 15] == 0   # bord : 5
    assert g.data[12, 12, 17] == Si and g.data[12, 12, 18] == 0   # coin : 2
    # masque et substrat hors fenêtre intacts
    assert g.data[10, 16, 19] == Si and g.data[10, 16, 20] == SiO2


def test_etch_crystal_diamond_window_trapezoid():
    """Fenêtre en losange (arêtes <110> cristallo) : parois {111} lentes
    le long des arêtes -> fosse trapézoïdale classique, auto-limitée :
    profondeur = rint(0.1*12*sqrt(3) + (8 - r)) < 12 — le terme vertical
    n'est jamais atteint (membrane KOH)."""
    g, Si, SiO2 = _koh_wafer("diamond")
    get_step("cpu", "etch_crystal")(g, 12, Si)
    tops = [_top(g, Si, 16 + r, 16) for r in (0, 2, 4, 6, 8)]
    assert tops == [9, 11, 13, 15, 17]
    assert g.data[16, 16, 9] == Si and g.data[16, 16, 10] == 0
    assert g.data[16, 16, 8] == Si                # auto-limitée (< 12)
    # masque intact hors losange
    assert g.data[10, 22, 19] == Si and g.data[10, 22, 20] == SiO2


def test_etch_crystal_bare_wafer_no_edges():
    """Substrat nu : pas de bord -> pas de facette, gravure verticale
    pure de depth voxels (taux de référence {100} = 1.0)."""
    g, Si, SiO2 = _koh_wafer(None)
    get_step("cpu", "etch_crystal")(g, 5, Si)
    col = g.data[5, 5, :]
    assert col[14] == Si and col[15] == 0
    assert g.data[0, 5, 15] == 0                  # bord de grille : plat aussi


def test_etch_crystal_orientation_110():
    """Wafer (110) : les plans {111} (taux 0.1) partent des bords en y
    (slopes (0,1)) -> rigoles peu profondes près des bords ; profondeur
    = rint(0.1*10*sqrt(2) + dist. au bord en y)."""
    g, Si, SiO2 = _koh_wafer("rect")
    get_step("cpu", "etch_crystal")(g, 10, Si, orientation="110")
    tops = [_top(g, Si, 16, y) for y in range(12, 20)]
    assert tops == [18, 17, 16, 15, 15, 16, 17, 18]
    assert g.data[16, 10, 19] == Si and g.data[16, 10, 20] == SiO2


# --------------------------------------------------------------------------
# Sélectivité et arrêts
# --------------------------------------------------------------------------

def test_etch_crystal_no_exposure_no_etch():
    """Matériau jamais exposé à l'air (sous masque complet) : grille
    inchangée octet pour octet."""
    g, Si, SiO2 = _koh_wafer(None)
    before = g.data.copy()
    get_step("cpu", "etch_crystal")(g, 10, Si)
    assert g.data.tobytes() == before.tobytes()


def test_etch_crystal_stops_on_buried_oxide():
    """Arrêt au premier voxel qui n'est pas `material` : l'oxyde enfoui
    n'est pas traversé, le Si dessous est protégé."""
    g, Si, SiO2 = _koh_wafer("rect", buried=(14, 15))
    get_step("cpu", "etch_crystal")(g, 6, Si)
    assert g.data[16, 16, 16] == 0                # gravé au-dessus de l'arrêt
    assert g.data[16, 16, 15] == SiO2             # arrêt : oxyde intact
    assert g.data[16, 16, 14] == SiO2
    assert g.data[16, 16, 13] == Si               # Si sous l'oxyde : protégé


# --------------------------------------------------------------------------
# Validation DSL
# --------------------------------------------------------------------------

def _deck(**step):
    base = {"version": 1,
            "grid": {"pitch_nm": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "materials": {"Si": {"color": "#4d79ff"}},
            "steps": [dict({"op": "etch_crystal", "material": "Si",
                            "depth_nm": 2}, **step)]}
    return base


def test_etch_crystal_deck_validation_errors():
    with pytest.raises(DeckError):
        DeckRunner(_deck(orientation="211")).run()
    with pytest.raises(DeckError):
        DeckRunner(_deck(facet_rates={"210": 1.0})).run()
    with pytest.raises(DeckError):
        DeckRunner(_deck(facet_rates={"100": 0})).run()
    with pytest.raises(DeckError):
        DeckRunner(_deck(facet_rates={})).run()
    deck = _deck()
    deck["steps"] = [{"op": "etch_crystal", "depth_nm": 2}]  # material absent
    with pytest.raises(DeckError):
        DeckRunner(deck).run()


def test_etch_crystal_deck_end_to_end(tmp_path):
    """Branchements DSL : conversion nm -> voxels + exécution complète."""
    deck = {"version": 1,
            "grid": {"pitch_nm": 1.0, "nx": 8, "ny": 8, "nz": 12},
            "materials": {"Si": {"color": "#4d79ff"}},
            "steps": [{"op": "substrate", "material": "Si",
                       "thickness_nm": 4},
                      {"op": "etch_crystal", "material": "Si",
                       "depth_nm": 2}]}
    runner = DeckRunner(deck, outdir=tmp_path)
    runner.run()
    g = runner.g
    assert g.data[4, 4, 2] == 0 and g.data[4, 4, 1] == g.mat["Si"]
