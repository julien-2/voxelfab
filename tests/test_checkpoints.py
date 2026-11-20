"""
Tests des checkpoints entre étapes (issue #12) : sauvegarde de l'état de la
grille après une étape clef (png / h5 / vtk / stl), au niveau étape
(`checkpoint:`) ou global (`checkpoints: {every: …}`).

    uv run pytest tests/test_checkpoints.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")          # headless

from voxelfab import DeckRunner  # noqa: E402
from voxelfab.deck import DeckError  # noqa: E402
from voxelfab.export import load_h5  # noqa: E402

h5py = pytest.importorskip("h5py")


def _deck(steps, **over):
    deck = {
        "version": 1,
        "grid": {"pitch_nm": 10.0, "nx": 8, "ny": 8, "nz": 16},
        "materials": {"Si": {"color": "#4d79ff"}, "SiO2": {"color": "#c9c9c9"}},
        "steps": steps,
        "outputs": [],
    }
    deck.update(over)
    return deck


STEPS_2 = [
    {"op": "substrate", "material": "Si", "thickness_nm": 20},
    {"op": "deposit", "material": "SiO2", "thickness_nm": 30,
     "mode": "conformal"},
]


def test_checkpoint_named_files_and_content(tmp_path):
    """checkpoint: {h5: …, png: …} sur l'étape 1 -> fichiers nommés écrits,
    et le h5 contient exactement l'état après l'étape 1."""
    steps = [dict(STEPS_2[0], checkpoint={"h5": "apres_substrate.h5",
                                          "png": "apres_substrate.png"}),
             STEPS_2[1]]
    r = DeckRunner(_deck(steps), outdir=tmp_path)
    r.run()
    h5 = tmp_path / "apres_substrate.h5"
    png = tmp_path / "apres_substrate.png"
    assert h5.exists() and png.exists() and png.stat().st_size > 0
    with h5py.File(h5, "r") as f:
        assert int(f.attrs["prov_checkpoint_step"]) == 1
    # contenu = état intermédiaire : même grille qu'un run à 1 étape
    r1 = DeckRunner(_deck([STEPS_2[0]]), outdir=tmp_path / "ref")
    r1.run()
    g = load_h5(str(h5))
    assert np.array_equal(g.data, r1.g.data)


def test_checkpoint_global_every_autonaming(tmp_path):
    """checkpoints: {every: 1, dir: ckpt, formats: [h5]} -> nommage auto
    ckpt/step_XXXX_<op>.h5 pour chaque étape."""
    r = DeckRunner(_deck(STEPS_2, checkpoints={"every": 1, "dir": "ckpt",
                                               "formats": ["h5"]}),
                   outdir=tmp_path)
    r.run()
    assert (tmp_path / "ckpt" / "step_0001_substrate.h5").exists()
    assert (tmp_path / "ckpt" / "step_0002_deposit.h5").exists()
    # le checkpoint de l'étape 2 = état final du run
    g = load_h5(str(tmp_path / "ckpt" / "step_0002_deposit.h5"))
    assert np.array_equal(g.data, r.g.data)


def test_checkpoint_true_default_formats(tmp_path):
    """checkpoint: true -> formats par défaut (png, h5), nommage auto."""
    steps = [dict(STEPS_2[0], checkpoint=True)]
    DeckRunner(_deck(steps), outdir=tmp_path).run()
    assert (tmp_path / "ckpt" / "step_0001_substrate.png").exists()
    assert (tmp_path / "ckpt" / "step_0001_substrate.h5").exists()


def test_checkpoint_vtk(tmp_path):
    pytest.importorskip("pyvista")
    steps = [dict(STEPS_2[0], checkpoint=["vtk"])]
    DeckRunner(_deck(steps), outdir=tmp_path).run()
    vtk = tmp_path / "ckpt" / "step_0001_substrate.vtk"
    assert vtk.exists() and vtk.stat().st_size > 0


def test_checkpoint_format_inconnu(tmp_path):
    steps = [dict(STEPS_2[0], checkpoint=["xyz"])]
    with pytest.raises(DeckError, match="format de checkpoint inconnu"):
        DeckRunner(_deck(steps), outdir=tmp_path).run()


def test_sans_checkpoint_rien_ecrit(tmp_path):
    """Rétrocompat : sans checkpoint, aucun répertoire ckpt créé."""
    DeckRunner(_deck(STEPS_2), outdir=tmp_path).run()
    assert not (tmp_path / "ckpt").exists()
