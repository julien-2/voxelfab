"""
Tests de l'opération grow (croissance par conversion : oxydation,
siliciuration).

    uv run pytest tests/test_grow.py -v

Sémantique (cf. ops/base.py) : `from` est consommé sur depth_voxels à
partir de son interface avec le côté réactif (air + `into`) ; le produit
occupe la zone consommée + une croissance proportionnelle
(expansion − 1) par voxel consommé, dans le côté réactif uniquement.
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


def _siwafer(nx=10, ny=10, nz=30, si_th=20):
    """Substrat Si sur z in [0, si_th), air au-dessus."""
    g = VoxelGrid(nx, ny, nz, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    g.data[:, :, :si_th] = Si
    return g, Si


def test_oxide_44_56_split():
    """Oxyde de 11 voxels : 5 consommés sous la surface, 6 au-dessus
    (expansion 2.27 -> ~44 % / ~56 %, cf. Deal-Grove)."""
    g, Si = _siwafer()
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    grow = get_step("cpu", "grow")
    grow(g, 5, Si, SiO2, expansion=2.27)
    col = g.data[5, 5, :]
    assert col[14] == Si          # Si non consommé (distance 6 > 5)
    assert col[15] == SiO2        # premier voxel consommé
    assert col[19] == SiO2        # surface d'origine
    assert col[25] == SiO2        # croissance au-dessus (6 voxels)
    assert col[26] == 0           # air intact
    assert (col == SiO2).sum() == 11


def test_conversion_only_equivalence_with_etch_isotropic():
    """expansion = 1.0 (conversion pure, sans croissance) == etch isotropic
    du matériau consommé + conversion des voxels retirés (contrat)."""
    g1, Si = _siwafer()
    SiO2 = g1.add_material("SiO2", "#c9c9c9")
    g2, _ = _siwafer()
    before = g2.data.copy()
    get_step("cpu", "grow")(g1, 4, Si, SiO2, expansion=1.0)
    get_step("cpu", "etch", "isotropic")(g2, 4, materials=(Si,))
    converted = (before == Si) & (g2.data == 0)
    g2.data[converted] = SiO2
    assert g1.data.tobytes() == g2.data.tobytes()


def test_thin_layer_proportional_growth():
    """Si plus mince que depth_nm : tout est consommé, la croissance est
    réduite au prorata (3 voxels consommés, expansion 2.0 -> 3 de croissance)."""
    g = VoxelGrid(10, 10, 20, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    g.data[:, :, :3] = Si
    get_step("cpu", "grow")(g, 5, Si, SiO2, expansion=2.0)
    col = g.data[5, 5, :]
    assert (col[:6] == SiO2).all()    # 3 consommés + 3 de croissance
    assert col[6] == 0                # pas de sur-croissance


def test_sidewall_and_trench_oxidation():
    """L'oxydation attaque toutes les surfaces exposées : fond d'une
    tranchée, parois latérales, et surface supérieure."""
    g, Si = _siwafer(nx=12, ny=6, nz=20, si_th=10)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    # tranchée : colonnes x in [4, 6) gravées jusqu'au fond
    mask = np.zeros((g.nx, g.ny), bool)
    mask[4:6, :] = True
    get_step("cpu", "etch", "directional")(g, 10, mask=mask)
    get_step("cpu", "grow")(g, 2, Si, SiO2, expansion=2.0)
    assert g.data[3, 2, 5] == SiO2    # paroi : consommée (distance 1)
    assert g.data[2, 2, 5] == SiO2    # paroi : consommée (distance 2)
    assert g.data[1, 2, 5] == Si      # au-delà de depth : intact
    assert g.data[4, 2, 0] == SiO2    # fond de tranchée oxydé
    assert g.data[4, 2, 9] == SiO2    # croissance latérale sur la paroi
    assert g.data[0, 2, 9] == SiO2    # surface supérieure oxydée
    assert g.data[0, 2, 7] == Si      # 2 voxels consommés seulement


def test_silicidation_with_into():
    """Siliciuration : le Si est consommé, le Ni est converti dans la
    limite du budget ; l'excédent de Ni reste (nettoyé par gravure)."""
    g = VoxelGrid(8, 8, 24, pitch=1.0)
    Si = g.add_material("Si", "#4d79ff")
    Ni = g.add_material("Ni", "#9e9e9e")
    NiSi = g.add_material("NiSi", "#616161")
    g.data[:, :, :10] = Si
    g.data[:, :, 10:15] = Ni
    # 1 nm Ni consomme 1.83 nm Si -> 2.54 nm NiSi : expansion = 2.54/1.83
    get_step("cpu", "grow")(g, 9, Si, NiSi, expansion=2.54 / 1.83,
                            into=(Ni,))
    col = g.data[5, 5, :]
    assert col[0] == Si               # non consommé (distance 10 > 9)
    assert col[9] == NiSi             # Si consommé converti
    assert col[12] == NiSi            # Ni converti (dans le budget)
    assert col[14] == Ni              # Ni excédentaire : intact
    assert col[15] == 0               # air


def test_mask_limits_reaction():
    g, Si = _siwafer(si_th=10)
    SiO2 = g.add_material("SiO2", "#c9c9c9")
    mask = np.zeros((g.nx, g.ny), bool)
    mask[:5, :] = True                # moitié x < 5 ouverte
    get_step("cpu", "grow")(g, 3, Si, SiO2, expansion=2.0, mask=mask)
    assert g.data[2, 5, 9] == SiO2    # côté ouvert : consommé + converti
    assert g.data[2, 5, 12] == SiO2   # croissance au-dessus
    assert g.data[7, 5, 9] == Si      # côté masqué : intact
    assert g.data[7, 5, 12] == 0


def test_deck_validation_errors():
    base = {"version": 1,
            "grid": {"pitch_nm": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "materials": {"SiO2": {"color": "#ffffff"}}}
    # 'from' manquant
    deck = dict(base, steps=[{"op": "grow", "material": "SiO2",
                              "depth_nm": 2}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
    # expansion <= 0
    deck = dict(base, steps=[{"op": "grow", "material": "SiO2",
                              "from": "SiO2", "depth_nm": 2,
                              "expansion": 0}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
    # matériau consommé inconnu
    deck = dict(base, steps=[{"op": "grow", "material": "SiO2",
                              "from": "Si", "depth_nm": 2}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
    # matériau 'into' inconnu
    deck = dict(base, steps=[{"op": "grow", "material": "SiO2",
                              "from": "SiO2", "depth_nm": 2,
                              "into": ["Ni"]}])
    with pytest.raises(DeckError):
        DeckRunner(deck).run()
