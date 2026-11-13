"""
Test du format de structure HDF5 : round-trip exact et provenance.

    uv run pytest tests/test_h5.py -v

Le fichier écrit par export_h5 doit recharger une grille byte-identique,
avec les mêmes matériaux et le même hash de structure que le run d'origine.
"""
import hashlib
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
from voxelfab.export import load_h5  # noqa: E402

h5py = pytest.importorskip("h5py")


def _run_demo(tmp_path, outputs):
    deck = yaml.safe_load(DEMO_DECK)
    deck["outputs"] = outputs
    runner = DeckRunner(deck, deck_path=ROOT / "decks" / "demo_deck.yaml",
                        outdir=tmp_path)
    runner.run()
    return runner


def test_h5_roundtrip(tmp_path):
    runner = _run_demo(tmp_path, [{"type": "h5", "file": "structure.h5"}])
    src = runner.g

    g = load_h5(tmp_path / "structure.h5")
    assert g.data.tobytes() == src.data.tobytes()
    assert g.pitch == src.pitch
    assert (g.nx, g.ny, g.nz) == (src.nx, src.ny, src.nz)
    assert [(g.materials.name(i), g.materials.color(i))
            for i in g.materials.ids] == \
           [(src.materials.name(i), src.materials.color(i))
            for i in src.materials.ids]


def test_h5_provenance_and_hash(tmp_path):
    runner = _run_demo(tmp_path, [{"type": "h5", "file": "structure.h5"}])
    expected_sha = hashlib.sha256(runner.g.data.tobytes()).hexdigest()

    with h5py.File(tmp_path / "structure.h5", "r") as f:
        assert f.attrs["structure_sha256"] == expected_sha
        assert f.attrs["prov_voxelfab_version"] == "0.1.0"
        assert f.attrs["prov_backend"] == "cpu"
        assert list(f.attrs["prov_grid"]) == \
            [runner.g.nx, runner.g.ny, runner.g.nz]
        # pas de timings : le fichier doit être reproductible
        assert "prov_steps" not in f.attrs
        assert f["grid"].compression == "gzip"
        mats = f["materials"][()]
        assert [m["name"].decode() for m in mats] == \
            [runner.g.materials.name(i) for i in runner.g.materials.ids
             if i != 0]


def test_h5_matches_report_hash(tmp_path):
    """Le hash dans structure.h5 == celui de report.json (même run)."""
    runner = _run_demo(tmp_path, [{"type": "h5", "file": "structure.h5"},
                                  {"type": "report", "file": "report.json"}])
    with h5py.File(tmp_path / "structure.h5", "r") as f:
        h5_sha = f.attrs["structure_sha256"]
    rep = json.loads((tmp_path / "report.json").read_text())
    assert h5_sha == rep["structure_sha256"]
