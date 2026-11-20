"""
Tests de l'issue #11 : substrat multicouche déclaratif (`substrate:`) et
auto-grow de nz (`grid.nz_max`).

    uv run pytest tests/test_substrate_nz.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DeckRunner  # noqa: E402
from voxelfab.deck import DeckError  # noqa: E402


def _deck(**over):
    deck = {
        "version": 1,
        "grid": {"pitch_nm": 10.0, "nx": 4, "ny": 4, "nz": 10},
        "materials": {"Si": {"color": "#4d79ff"}, "SiO2": {"color": "#c9c9c9"}},
        "steps": [],
        "outputs": [],
    }
    deck.update(over)
    return deck


# --- substrat multicouche ----------------------------------------------------

def test_substrate_multicouche_soi(tmp_path):
    """Ordre bottom-first : handle Si (fond z=0), BOX SiO2, top Si."""
    deck = _deck(substrate=[
        {"material": "Si", "thickness_nm": 200},    # 20 vx : z 0..19
        {"material": "SiO2", "thickness_nm": 40},    #  4 vx : z 20..23
        {"material": "Si", "thickness_nm": 20},      #  2 vx : z 24..25
    ])
    deck["grid"]["nz"] = 100                          # 1000 nm au total
    r = DeckRunner(deck, outdir=tmp_path, backend="cpu")
    r.run()
    si, sio2 = r.mat["Si"], r.mat["SiO2"]
    assert r.g.data[0, 0, 0] == si and r.g.data[0, 0, 19] == si
    assert r.g.data[0, 0, 20] == sio2 and r.g.data[0, 0, 23] == sio2
    assert r.g.data[0, 0, 24] == si and r.g.data[0, 0, 25] == si
    assert r.g.data[0, 0, 26] == 0                    # air au-dessus


def test_substrate_erreurs(tmp_path):
    with pytest.raises(DeckError, match="inconnu"):
        DeckRunner(_deck(substrate=[{"material": "Nope", "thickness_nm": 10}]),
                   outdir=tmp_path).run()
    with pytest.raises(DeckError, match="thickness_nm"):
        DeckRunner(_deck(substrate=[{"material": "Si", "thickness_nm": 0}]),
                   outdir=tmp_path).run()
    with pytest.raises(DeckError, match="dépasse nz"):
        # 2000 nm = 200 vx > nz=10, pas de nz_max
        DeckRunner(_deck(substrate=[{"material": "Si", "thickness_nm": 2000}]),
                   outdir=tmp_path).run()


def test_substrate_op_dans_steps_toujours_ok(tmp_path):
    """Rétrocompat : {op: substrate} dans steps marche toujours."""
    deck = _deck(steps=[{"op": "substrate", "material": "Si",
                         "thickness_nm": 30}])
    r = DeckRunner(deck, outdir=tmp_path)
    r.run()
    assert r.g.data[0, 0, 2] == r.mat["Si"] and r.g.data[0, 0, 3] == 0


# --- nz auto-grow (nz_max) ----------------------------------------------------

def test_nz_max_autogrow(tmp_path, capsys):
    deck = _deck()
    deck["grid"]["nz_max"] = 100
    deck["steps"] = [
        {"op": "substrate", "material": "Si", "thickness_nm": 20},   # h = 1
        {"op": "deposit", "material": "Si", "thickness_nm": 500,
         "mode": "directional"},                                     # +50 vx
    ]
    r = DeckRunner(deck, outdir=tmp_path)
    r.run()
    assert r.g.nz == 52                       # 2 + 50, auto-grow (marge 2*nz)
    assert "auto-grow" in capsys.readouterr().out
    assert r.g.data[0, 0, 51] == r.mat["Si"]  # rien n'est tronqué
    assert r.g.height_map().max() == 51


def test_sans_nz_max_tronque_avec_warning(tmp_path, capsys):
    deck = _deck(steps=[
        {"op": "substrate", "material": "Si", "thickness_nm": 20},
        {"op": "deposit", "material": "Si", "thickness_nm": 500,
         "mode": "directional"},
    ])
    r = DeckRunner(deck, outdir=tmp_path)
    r.run()
    assert r.g.nz == 10                       # pas d'auto-grow
    out = capsys.readouterr().out
    assert "WARNING" in out and "tronqué" in out
    assert r.g.height_map().max() == 9        # tronqué au sommet


def test_nz_max_depasse_tronque(tmp_path, capsys):
    deck = _deck()
    deck["grid"]["nz_max"] = 20
    deck["steps"] = [
        {"op": "substrate", "material": "Si", "thickness_nm": 20},
        {"op": "deposit", "material": "Si", "thickness_nm": 500,
         "mode": "directional"},
    ]
    r = DeckRunner(deck, outdir=tmp_path)
    r.run()
    assert r.g.nz == 20                       # grow jusqu'à nz_max puis clip
    assert "tronqué" in capsys.readouterr().out
    assert r.g.height_map().max() == 19
