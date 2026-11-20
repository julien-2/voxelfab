"""
Tests du bloc `layout:` — specs de couches v2 (issue #10) :
polarité clear/dark, merge multi-datatypes, biais individuel par couche,
avec rétrocompatibilité du format v1 `layers: {nom: [layer, datatype]}`.

    uv pip install gdstk shapely     # extra [layout]
    uv run pytest tests/test_layout.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

gdstk = pytest.importorskip("gdstk")
shapely = pytest.importorskip("shapely")

from voxelfab.grid import VoxelGrid  # noqa: E402
from voxelfab.layout import (  # noqa: E402
    layer_masks_from_layout, masks_from_layout, parse_layer_spec)
from voxelfab import DeckRunner  # noqa: E402

PITCH_NM = 20.0          # 20 nm/voxel -> 4 µm = 200 voxels


@pytest.fixture(scope="module")
def gds_file(tmp_path_factory):
    """Mini layout 4x4 µm : (10,0)=carré (0,0)-(2,2), (10,4)=carré
    (1,1)-(3,3) (chevauche (10,0)), (99,0)=plein champ (0,0)-(4,4)."""
    lib = gdstk.Library()
    cell = gdstk.Cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (2, 2), layer=10, datatype=0))
    cell.add(gdstk.rectangle((1, 1), (3, 3), layer=10, datatype=4))
    cell.add(gdstk.rectangle((0, 0), (4, 4), layer=99, datatype=0))
    lib.add(cell)
    path = tmp_path_factory.mktemp("layout") / "test.gds"
    lib.write_gds(str(path))
    return str(path)


@pytest.fixture(scope="module")
def grid():
    return VoxelGrid(200, 200, 10, pitch=PITCH_NM)


# --- parsing des specs ------------------------------------------------------

def test_parse_spec_v1_list():
    s = parse_layer_spec("gate", [10, 0])
    assert s == {"ld": [(10, 0)], "polarity": "dark", "bias_nm": 0.0}


def test_parse_spec_multi_datatypes():
    assert parse_layer_spec("g", [10, [0, 4]])["ld"] == [(10, 0), (10, 4)]
    assert parse_layer_spec("g", [[10, 0], [11, 2]])["ld"] == [(10, 0), (11, 2)]


def test_parse_spec_dict_polarity_bias():
    s = parse_layer_spec("f", {"ld": [99, 0], "polarity": "clear",
                               "bias_nm": -5.0}, default_bias=-2.0)
    assert s == {"ld": [(99, 0)], "polarity": "clear", "bias_nm": -5.0}
    # sans bias local -> hérite du biais global
    s = parse_layer_spec("f", {"ld": [99, 0], "polarity": "clear"},
                         default_bias=-2.0)
    assert s["bias_nm"] == -2.0


def test_parse_spec_erreurs():
    with pytest.raises(ValueError):
        parse_layer_spec("g", {"ld": [10, 0], "polarity": "blue"})
    with pytest.raises(ValueError):
        parse_layer_spec("g", [10])          # pas 2 éléments
    with pytest.raises(ValueError):
        parse_layer_spec("g", "gate")        # pas une liste
    with pytest.raises(ValueError):
        parse_layer_spec("g", [10, ["a"]])   # datatype non entier


# --- masques nommés ----------------------------------------------------------

def test_retrocompat_v1(gds_file, grid):
    """Format v1 [layer, datatype] : même masque que l'API basse niveau."""
    got = layer_masks_from_layout(gds_file, grid, {"gate": [10, 0]})
    ref = masks_from_layout(gds_file, grid, layers=((10, 0),))
    assert np.array_equal(got["gate"], ref[(10, 0)])


def test_merge_multi_datatypes(gds_file, grid):
    got = layer_masks_from_layout(gds_file, grid, {"gate": [10, [0, 4]]})["gate"]
    ref = masks_from_layout(gds_file, grid, layers=((10, 0), (10, 4)))
    assert np.array_equal(got, ref[(10, 0)] | ref[(10, 4)])
    # le merge couvre bien les deux carrés (chevauchement (1,1)-(2,2) µm)
    assert got[25, 25] and got[125, 125] and not got[175, 25]


def test_polarity_clear_is_complement(gds_file, grid):
    dark = layer_masks_from_layout(gds_file, grid, {"f": [99, 0]})["f"]
    clear = layer_masks_from_layout(
        gds_file, grid, {"f": {"ld": [99, 0], "polarity": "clear"}})["f"]
    assert np.array_equal(clear, ~dark)


def test_bias_local_overrides_global(gds_file, grid):
    base = layer_masks_from_layout(gds_file, grid, {"g": [10, 0]})["g"]
    local = layer_masks_from_layout(
        gds_file, grid, {"g": {"ld": [10, 0], "bias_nm": -100.0}})["g"]
    glob = layer_masks_from_layout(gds_file, grid, {"g": [10, 0]},
                                   bias_nm=-100.0)["g"]
    # -100 nm = -5 voxels de chaque côté : le masque rétrécit
    assert 0 < local.sum() < base.sum()
    assert np.array_equal(local, glob)       # biais local == biais global ici


def test_bias_conflict_raises(gds_file, grid):
    layers = {"a": {"ld": [10, 0], "bias_nm": -100.0},
              "b": {"ld": [10, 0], "bias_nm": 0.0}}
    with pytest.raises(ValueError, match="conflit de bias"):
        layer_masks_from_layout(gds_file, grid, layers)


# --- intégration DeckRunner --------------------------------------------------

def test_deck_runner_avec_layout(gds_file, tmp_path):
    deck = {
        "version": 1,
        "grid": {"pitch_nm": PITCH_NM, "nz": 10},   # nx/ny déduits du layout
        "materials": {"Si": {"color": "#4d79ff"}},
        "layout": {"file": gds_file, "layers": {"gate": [10, 0]}},
        "steps": [
            {"op": "substrate", "material": "Si", "thickness_nm": 100},  # 5 vx
            {"op": "etch", "depth_nm": 40, "mask": {"layer": "gate"}},    # 2 vx
        ],
        "outputs": [],
    }
    r = DeckRunner(deck, outdir=tmp_path, backend="cpu")
    r.run()
    assert (r.g.nx, r.g.ny) == (200, 200)          # déduit de la bbox 4 µm
    si = r.mat["Si"]
    # sous le masque gate (carré 2 µm -> voxels < 100) : 2 vx gravés au sommet
    assert r.g.data[50, 50, 4] == 0 and r.g.data[50, 50, 3] == 0
    assert r.g.data[50, 50, 2] == si
    # hors masque : rien n'est gravé
    assert r.g.data[150, 150, 4] == si
