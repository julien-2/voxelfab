"""
Tests de l'opération epitaxy (SEG conforme + épitaxie facettée).

    uv run pytest tests/test_epitaxy.py -v

Sémantique (cf. ops/base.py) :
- mode seed    : croissance conformale géodésique depuis la graine
                 exposée à l'air uniquement (les matériaux inertes
                 bloquent, pas de dépôt au-dessus du masque) ;
- mode crystal : construction SEG classique — croissance verticale
                 limitée par des plans {hkl} lents partant des bords de
                 la graine (facettes {111} à 54,7° sur wafer (100)).
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


def _flat_wafer(nx=10, ny=10, nz=30, si_th=20):
    g = VoxelGrid(nx, ny, nz, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    g.data[:, :, :si_th] = Si
    return g, Si


def _seg_wafer(nx=32, ny=32, nz=40):
    """Substrat Si (z < 8) + oxyde (z 8..13) percé d'une fenêtre carrée
    x, y in [12, 20) — le Si est exposé au fond de la fenêtre."""
    g = VoxelGrid(nx, ny, nz, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    GaAs = g.add_material("GaAs", "#ffcccc")   # déclaré mais jamais placé
    g.data[:, :, :8] = Si
    g.data[:, :, 8:14] = SiO2
    win = np.zeros((nx, ny), bool)
    win[12:20, 12:20] = True
    get_step("cpu", "etch", "directional")(g, 6, mask=win)
    return g, Si, SiO2, GaAs, win


# --------------------------------------------------------------------------
# mode seed — SEG conforme
# --------------------------------------------------------------------------

def test_seed_flat_growth():
    """Sur substrat nu : la graine est partout, croissance verticale de
    depth voxels ( comme un dépôt conformal sur surface plane)."""
    g, Si = _flat_wafer()
    get_step("cpu", "epitaxy", "seed")(g, 5, Si, Si)
    col = g.data[5, 5, :]
    assert col[19] == Si and col[24] == Si
    assert col[25] == 0
    assert (g.data[:, :, :20] == Si).all()      # substrat intact


def test_seed_window_confined_by_geodesy():
    """SEG : la croissance remplit la fenêtre (fond + hauteur) mais ne
    dépose RIEN au-dessus de l'oxyde : la géodésie à travers l'air est
    bloquée par les matériaux inertes."""
    g, Si, SiO2, GaAs, win = _seg_wafer()
    get_step("cpu", "epitaxy", "seed")(g, 6, Si, Si)
    assert g.data[15, 15, 8] == Si              # fond de fenêtre rempli
    assert g.data[15, 15, 13] == Si             # 6 voxels exactement
    assert g.data[15, 15, 14] == 0              # pas de débordement vertical
    assert g.data[10, 15, 13] == SiO2           # oxyde intact
    assert g.data[10, 15, 14] == 0              # pas de dépôt sur l'oxyde
    assert g.data[11, 15, 14] == 0              # (géodésie bloquée)


def test_seed_mask_limits_columns():
    g, Si = _flat_wafer()
    mask = np.zeros((g.nx, g.ny), bool)
    mask[:5, :] = True
    get_step("cpu", "epitaxy", "seed")(g, 3, Si, Si, mask=mask)
    assert g.data[2, 5, 22] == Si               # côté ouvert
    assert g.data[7, 5, 20] == 0                # côté masqué : rien


# --------------------------------------------------------------------------
# mode crystal — épitaxie facettée
# --------------------------------------------------------------------------

def test_crystal_faceted_mesa_roof():
    """Fenêtre 8×8, depth 12, rates Si typiques : les facettes {111}
    (slopes (1,1)) ne s'ancrent qu'aux COINS de la fenêtre, les {110}
    (slopes (1,0)/(0,1)) aux bords latéraux. Offset {111} =
    sqrt(3)*0.25*12 = 5.196 au-dessus de la graine (z = 7) -> hauteur
    au coin 12.2 ; le terme vertical (7 + 12 = 19) plafonne le centre.
    Profil le long de x à y = 16 (dist. L1 au coin le plus proche) :
    x=12 : 12.196+3 = 15.2 -> 15 ; x=13 : 16.2 -> 16 ; x=15 : 18.2 -> 18 ;
    x>=16 : plafonné à 19."""
    g, Si, SiO2, GaAs, win = _seg_wafer()
    get_step("cpu", "epitaxy", "crystal")(g, 12, Si, seed=Si)
    # coin de fenêtre (12,12) : facette {111} -> z = 7 + 5.196 = 12.2 -> 12
    assert g.data[12, 12, 12] == Si and g.data[12, 12, 13] == 0
    # milieu du bord (12,16) : 3 voxels du coin (12,19) -> 15.2 -> 15
    assert g.data[12, 16, 15] == Si and g.data[12, 16, 16] == 0
    assert g.data[13, 16, 16] == Si             # pente 1/voxel (facette)
    assert g.data[15, 16, 18] == Si and g.data[15, 16, 19] == 0
    assert g.data[16, 16, 19] == Si and g.data[16, 16, 20] == 0  # plafond 19
    assert g.data[19, 16, 15] == Si             # symétrie
    # pas de croissance hors de la graine : oxyde et air au-dessus intacts
    assert g.data[10, 16, 13] == SiO2
    assert g.data[10, 16, 14] == 0
    assert g.data[11, 16, 13] == SiO2           # pas de débordement latéral
    # fond de fenêtre rempli
    assert g.data[15, 15, 8] == Si


def test_crystal_flat_wafer_no_edges_no_facets():
    """Substrat nu : pas de bord de graine -> pas de facette, croissance
    verticale pure (taux {100} = 1.0 -> depth voxels)."""
    g, Si = _flat_wafer()
    get_step("cpu", "epitaxy", "crystal")(g, 5, Si, seed=Si)
    col = g.data[5, 5, :]
    assert col[24] == Si and col[25] == 0
    assert g.data[0, 5, 24] == Si               # bord de grille : plat aussi


def test_crystal_orientation_110():
    """Wafer (110) : la famille la plus parallèle est {110} (taux 0.6) ->
    croissance verticale = 0.6 × depth."""
    g, Si = _flat_wafer(nx=32, ny=32)
    get_step("cpu", "epitaxy", "crystal")(g, 5, Si, seed=Si,
                                          orientation="110")
    col = g.data[5, 16, :]
    assert col[22] == Si                        # 19 + 0.6*5 = 22
    assert col[23] == 0


def test_crystal_no_seed_no_growth():
    """Graine déclarée mais jamais placée (jamais exposée) : grille
    inchangée octet pour octet."""
    g, Si, SiO2, GaAs, win = _seg_wafer()
    before = g.data.copy()
    get_step("cpu", "epitaxy", "crystal")(g, 12, Si, seed=GaAs)
    assert g.data.tobytes() == before.tobytes()


# --------------------------------------------------------------------------
# Validation DSL
# --------------------------------------------------------------------------

def _deck(**step):
    base = {"version": 1,
            "grid": {"pitch_nm": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "materials": {"Si": {"color": "#4d79ff"},
                          "SiO2": {"color": "#c9c9c9"}}}
    return dict(base, steps=[step])


def test_deck_validation_errors():
    # mode inconnu
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Si",
                         depth_nm=2, mode="facet")).run()
    # orientation inconnue
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Si",
                         depth_nm=2, mode="crystal",
                         orientation="211")).run()
    # famille de facettes inconnue
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Si",
                         depth_nm=2, mode="crystal",
                         facet_rates={"210": 1.0})).run()
    # taux <= 0
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Si",
                         depth_nm=2, mode="crystal",
                         facet_rates={"100": 0})).run()
    # facet_rates vide
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Si",
                         depth_nm=2, mode="crystal",
                         facet_rates={})).run()
    # seed manquant
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", depth_nm=2)).run()
    # graine inconnue
    with pytest.raises(DeckError):
        DeckRunner(_deck(op="epitaxy", material="Si", seed="Ni",
                         depth_nm=2)).run()
