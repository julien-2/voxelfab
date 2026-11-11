"""
Test structure-golden : le deck de démo doit produire une structure voxel
byte-identique. Le hash de référence vit dans tests/golden/golden.json.

Premier run (enregistre la référence) :
    VF_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
Ensuite, à chaque changement :
    uv run pytest tests/test_golden.py

Si le test échoue après un changement VOULU de sémantique moteur,
régénérer la référence et documenter le diff dans le message de commit.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless : aucun rendu pendant le test

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vfdeck import DEMO_DECK, DeckRunner  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "golden.json"


def _run_demo(tmp_path):
    deck = yaml.safe_load(DEMO_DECK)
    deck["outputs"] = []              # pas de rendu ni d'export : test pur
    runner = DeckRunner(deck, deck_path=ROOT / "decks" / "demo_deck.yaml",
                        outdir=tmp_path)
    runner.run()                      # écrit report.json dans tmp_path
    report = json.loads((tmp_path / "report.json").read_text())
    return report["structure_sha256"], report["provenance"]


def test_demo_structure_matches_golden(tmp_path):
    h, prov = _run_demo(tmp_path)

    if not GOLDEN.exists():
        if os.environ.get("VF_UPDATE_GOLDEN") == "1":
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(json.dumps({
                "structure_sha256": h,
                "grid": prov["grid"],
                "pitch_nm": prov["pitch_nm"],
                "note": "régénérer avec VF_UPDATE_GOLDEN=1",
            }, indent=2))
            pytest.skip(f"golden créé : {h[:16]}…")
        pytest.fail("golden manquant — lancer avec VF_UPDATE_GOLDEN=1")

    expected = json.loads(GOLDEN.read_text())["structure_sha256"]
    assert h == expected, (
        "la structure a changé ! Si le changement de sémantique est voulu, "
        "régénérer avec VF_UPDATE_GOLDEN=1 et documenter le diff"
    )
