"""
Tests de la sélection automatique de l'arch Taichi (issue #5) :
env VOXELFAB_TAICHI_ARCH, détection CUDA, micro-bench x64 vs vulkan caché.

    uv run pytest tests/test_arch_select.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ti = pytest.importorskip("taichi")

from voxelfab.ops import gpu as g  # noqa: E402


def test_env_force(monkeypatch):
    monkeypatch.setenv("VOXELFAB_TAICHI_ARCH", "cpu")
    assert g._select_arch() == ti.cpu
    monkeypatch.setenv("VOXELFAB_TAICHI_ARCH", "cuda")
    assert g._select_arch() == ti.cuda
    monkeypatch.setenv("VOXELFAB_TAICHI_ARCH", "vulkan")
    assert g._select_arch() == ti.vulkan
    monkeypatch.setenv("VOXELFAB_TAICHI_ARCH", "bogus")
    with pytest.raises(ValueError, match="VOXELFAB_TAICHI_ARCH"):
        g._select_arch()


def test_auto_cuda_if_device(monkeypatch):
    monkeypatch.delenv("VOXELFAB_TAICHI_ARCH", raising=False)
    monkeypatch.setattr(g, "gpu_available", lambda: True)
    assert g._select_arch() == ti.cuda


def test_auto_bench_pick_fastest_and_cache(monkeypatch, tmp_path):
    """Sans CUDA : micro-bench x64 vs vulkan, le plus rapide gagne, et le
    2e appel lit le cache (pas de re-bench)."""
    monkeypatch.delenv("VOXELFAB_TAICHI_ARCH", raising=False)
    monkeypatch.setattr(g, "gpu_available", lambda: False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))   # Path.home() Windows
    monkeypatch.setenv("HOME", str(tmp_path))
    times = {"x64": 0.5, "vulkan": 0.2}
    monkeypatch.setattr(g, "_bench_arch", lambda name: times[name])
    assert g._select_arch() == ti.vulkan               # vulkan plus rapide
    # cache écrit
    assert (tmp_path / ".voxelfab" / "arch_cache.json").exists()
    # 2e appel : cache hit, _bench_arch ne doit plus être appelé
    def boom(name):
        raise AssertionError("re-bench malgré le cache")
    monkeypatch.setattr(g, "_bench_arch", boom)
    assert g._select_arch() == ti.vulkan


def test_auto_bench_fallback_cpu_if_all_fail(monkeypatch, tmp_path):
    monkeypatch.delenv("VOXELFAB_TAICHI_ARCH", raising=False)
    monkeypatch.setattr(g, "gpu_available", lambda: False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(g, "_bench_arch", lambda name: float("inf"))
    assert g._select_arch() == ti.cpu                  # tout échoue -> cpu
