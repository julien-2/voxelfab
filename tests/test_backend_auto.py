"""
Test du backend 'auto' : résolu en 'gpu' si taichi est installé, sinon
'cpu' — et le deck de démo produit le hash golden quel que soit le backend.

    uv run pytest tests/test_backend_auto.py -v
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless : aucun rendu pendant le test

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxelfab import DEMO_DECK, DeckRunner  # noqa: E402
from voxelfab.ops import available_backends, resolve_backend  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "golden.json"

EXPECTED_AUTO = "gpu" if "gpu" in available_backends() else "cpu"


def test_resolve_backend():
    assert resolve_backend("auto") == EXPECTED_AUTO
    assert resolve_backend("cpu") == "cpu"
    if "gpu" in available_backends():
        assert resolve_backend("gpu") == "gpu"
    with pytest.raises(KeyError):
        resolve_backend("nope")


def test_run_deck_backend_auto(tmp_path):
    deck = yaml.safe_load(DEMO_DECK)
    deck["outputs"] = [{"type": "h5", "file": "structure.h5"}]
    runner = DeckRunner(deck, deck_path=ROOT / "decks" / "demo_deck.yaml",
                        outdir=tmp_path, backend="auto")
    runner.run()
    # le backend effectif est résolu et journalisé (reproductibilité)
    assert runner.backend == EXPECTED_AUTO
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["provenance"]["backend"] == EXPECTED_AUTO
    # contrat voxel-exact : hash golden identique quel que soit le backend
    golden = json.loads(GOLDEN.read_text())
    assert report["structure_sha256"] == golden["structure_sha256"]
