"""
Tests du batching GPU (issue #5) : [substrate?] + deposits directionnels
consécutifs fusionnés en UN noyau `stack` — bit-identique vs étapes séparées
(contrat cpu/gpu), et désactivé si checkpoint ou history.

    uv run pytest tests/test_batching.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DeckRunner  # noqa: E402
from voxelfab.ops import available_backends  # noqa: E402

pytestmark = pytest.mark.skipif("gpu" not in available_backends(),
                                reason="taichi absent")


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


STACK = [
    {"op": "substrate", "material": "Si", "thickness_nm": 20},
    {"op": "deposit", "material": "SiO2", "thickness_nm": 30,
     "mode": "directional"},
    {"op": "deposit", "material": "Si", "thickness_nm": 10,
     "mode": "directional"},
]


def test_batch_stack_gpu_matches_cpu(tmp_path, capsys):
    """Le batch (3 étapes -> 1 noyau) produit les mêmes octets que le CPU
    (étapes séparées) — contrat voxel-exact."""
    rc = DeckRunner(_deck(STACK), outdir=tmp_path / "cpu", backend="cpu")
    rc.run()
    rg = DeckRunner(_deck(STACK), outdir=tmp_path / "gpu", backend="gpu")
    rg.run()
    out = capsys.readouterr().out
    assert "[batch] 3 étapes -> 1 noyau" in out
    assert np.array_equal(rg.g.data, rc.g.data)
    assert rg.g.height_map().max() == 5      # 2 + 3 + 1 vx


def test_no_batch_with_checkpoint(tmp_path, capsys):
    """Pas de batch si une étape du batch a un checkpoint."""
    steps = [dict(STACK[0]),
             dict(STACK[1], checkpoint={"h5": "mid.h5"}),
             STACK[2]]
    DeckRunner(_deck(steps), outdir=tmp_path, backend="gpu").run()
    out = capsys.readouterr().out
    assert "[batch]" not in out
    assert (tmp_path / "mid.h5").exists()


def test_no_batch_with_history(tmp_path, capsys):
    """Pas de batch si history actif (sinon les étapes intermédiaires
    manqueraient dans l'historique)."""
    DeckRunner(_deck(STACK, history={"file": "h.h5"}),
               outdir=tmp_path, backend="gpu").run()
    assert "[batch]" not in capsys.readouterr().out
    assert (tmp_path / "h.h5").exists()


def test_single_step_not_batched(tmp_path, capsys):
    """Une seule étape batchable ne déclenche pas de batch."""
    DeckRunner(_deck([STACK[0]]), outdir=tmp_path, backend="gpu").run()
    assert "[batch]" not in capsys.readouterr().out
