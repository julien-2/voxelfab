"""
Tests de l'historique de procédé dans un seul fichier HDF5 (issue #9) :
reconstruction byte-exact par deltas chaînés, modes summary/delta/full,
taille delta << full, et CLI `vf history`.

    uv run pytest tests/test_history.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")

from voxelfab import DeckRunner  # noqa: E402
from voxelfab.deck import DeckError  # noqa: E402
from voxelfab.history import grid_at, load_history  # noqa: E402

h5py = pytest.importorskip("h5py")

STEPS = [
    {"op": "substrate", "material": "Si", "thickness_nm": 20},
    {"op": "deposit", "material": "SiO2", "thickness_nm": 30, "mode": "conformal"},
    {"op": "etch", "depth_nm": 10, "materials": ["SiO2"], "mode": "isotropic"},
]


def _deck(history, steps=None):
    return {
        "version": 1,
        "grid": {"pitch_nm": 10.0, "nx": 8, "ny": 8, "nz": 16},
        "materials": {"Si": {"color": "#4d79ff"}, "SiO2": {"color": "#c9c9c9"}},
        "steps": STEPS if steps is None else steps,
        "history": history,
        "outputs": [],
    }


def test_history_delta_reconstruction_byte_exact(tmp_path):
    """grid_at(k) == grille du run tronqué à k étapes, pour tout k."""
    r = DeckRunner(_deck({"file": "history.h5", "mode": "delta"}),
                   outdir=tmp_path)
    r.run()
    h = tmp_path / "history.h5"
    meta = load_history(h)
    assert [s["op"] for s in meta["steps"]] == ["init", "substrate",
                                                "deposit", "etch"]
    assert meta["mode"] == "delta"
    for k in range(0, 4):
        ref = DeckRunner(_deck(None, steps=STEPS[:k]), outdir=tmp_path / f"r{k}")
        ref.run()
        g = grid_at(h, k)
        assert np.array_equal(g.data, ref.g.data), f"étape {k}"
    # l'état final reconstruit == état final du run
    assert np.array_equal(grid_at(h, 3).data, r.g.data)


def test_history_hash_chain(tmp_path):
    """Les hashs chaînés sont distincts et vérifiables (recalcul)."""
    import hashlib, json
    DeckRunner(_deck({"file": "h.h5", "mode": "delta"}), outdir=tmp_path).run()
    meta = load_history(tmp_path / "h.h5")
    shas = [s["step_sha256"] for s in meta["steps"]]
    assert len(set(shas)) == len(shas)          # tous distincts
    # re-vérifie la chaîne : sha_k dépend de sha_{k-1}
    with h5py.File(tmp_path / "h.h5", "r") as f:
        prev = "0" * 64
        for name in sorted(f["steps"].keys()):
            grp = f["steps"][name]
            h = hashlib.sha256()
            h.update(prev.encode())
            h.update(str(grp.attrs["op"]).encode())
            h.update(grp.attrs["params"].encode())
            h.update(grp["runs"][()].tobytes())
            h.update(grp["runs_mats"][()].tobytes())
            assert h.hexdigest() == grp.attrs["step_sha256"]
            prev = str(grp.attrs["step_sha256"])


def test_history_delta_much_smaller_than_full(tmp_path):
    """Grille DENSE (200³ aléatoire, peu compressible) + étape modifiant 1%
    des voxels : le mode delta (runs RLE des voxels modifiés) est largement
    plus compact que full (grille complète gzip par étape). C'est le cas
    des grosses simulations (10⁸–10⁹ vx) où full explose."""
    from voxelfab.grid import VoxelGrid
    from voxelfab.history import HistoryWriter

    # cas réaliste : grosse grille (256³ = 16,7 M vx), substrat uniforme,
    # puis dépôts de couches minces pleines (peu de voxels modifiés/étape)
    g = VoxelGrid(256, 256, 128, pitch=1.0)
    g.add_material("A", "#111111")
    g.add_material("B", "#222222")
    g.data[:, :, :50] = 1                          # substrat uniforme (init)

    states = []
    for mode, name in (("delta", "d.h5"), ("full", "f.h5")):
        with HistoryWriter(tmp_path / name, g, {}, mode=mode) as hw:
            hw.record(0, "init", {}, g)
            for k, (z0, z1, m) in enumerate([(50, 52, 2), (60, 62, 1),
                                             (70, 75, 2)], start=1):
                gk = g.copy()
                gk.data[:, :, z0:z1] = m           # couche pleine mince
                hw.record(k, "deposit", {"z": z0}, gk)
                if mode == "delta":
                    states.append(gk)

    sz_d = (tmp_path / "d.h5").stat().st_size
    sz_f = (tmp_path / "f.h5").stat().st_size
    assert sz_d * 3 < sz_f                        # delta gagne >= 3x ici (42,9k vs 166k)
    # reconstruction byte-exact en mode delta, à chaque étape
    for k, gk in enumerate(states, start=1):
        assert np.array_equal(grid_at(tmp_path / "d.h5", k).data, gk.data)


def test_history_demo_golden_reconstruction(tmp_path):
    """Deck démo complet en mode delta : l'état reconstruit à la dernière
    étape a le hash golden e1316a29… (contrat voxel-exact de l'historique)."""
    import hashlib
    import yaml
    from voxelfab import DEMO_DECK

    d = yaml.safe_load(DEMO_DECK)
    d["outputs"] = []
    d["history"] = {"file": "h.h5", "mode": "delta"}
    DeckRunner(d, outdir=tmp_path).run()
    meta = load_history(tmp_path / "h.h5")
    assert len(meta["steps"]) == 18               # init + 17 étapes
    g = grid_at(tmp_path / "h.h5", 17)
    assert hashlib.sha256(g.data.tobytes()).hexdigest() == \
        "e1316a2970a1e6f38612e6ee98d954123e5fe173f2acd06053d8cb63283743c1"


def test_history_summary_mode(tmp_path):
    DeckRunner(_deck({"file": "h.h5", "mode": "summary"}), outdir=tmp_path).run()
    meta = load_history(tmp_path / "h.h5")
    assert meta["mode"] == "summary"
    assert len(meta["steps"]) == 4
    with h5py.File(tmp_path / "h.h5", "r") as f:
        for name in f["steps"]:
            assert "runs" not in f["steps"][name]     # pas de payload
            assert int(f["steps"][name].attrs["n_changed"]) >= 0


def test_history_every(tmp_path):
    DeckRunner(_deck({"file": "h.h5", "mode": "delta", "every": 2}),
               outdir=tmp_path).run()
    meta = load_history(tmp_path / "h.h5")
    assert [s["index"] for s in meta["steps"]] == [0, 2]   # init + étape 2


def test_history_mode_invalide(tmp_path):
    with pytest.raises(DeckError, match="mode"):
        DeckRunner(_deck({"file": "h.h5", "mode": "bogus"}),
                   outdir=tmp_path).run()


def test_cli_history_list_and_export(tmp_path, monkeypatch, capsys):
    DeckRunner(_deck({"file": "h.h5", "mode": "delta"}), outdir=tmp_path).run()
    from voxelfab.cli import main
    monkeypatch.setattr(sys, "argv", ["vf", "history", str(tmp_path / "h.h5")])
    main()
    out = capsys.readouterr().out
    assert "étape   0" in out and "substrate" in out and "etch" in out
    # export de l'état étape 2
    monkeypatch.setattr(sys, "argv", ["vf", "history", str(tmp_path / "h.h5"),
                                      "--step", "2", "-o", str(tmp_path / "exp")])
    main()
    g = grid_at(tmp_path / "h.h5", 2)
    from voxelfab.export import load_h5
    assert np.array_equal(load_h5(str(tmp_path / "exp" / "step_0002.h5")).data,
                          g.data)
