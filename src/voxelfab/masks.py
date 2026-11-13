"""Masques lithographiques géométriques (champs 2D booléens sur (x, y))."""

from __future__ import annotations

import numpy as np


def mask_line(nx: int, ny: int, x0: float, x1: float) -> np.ndarray:
    """Bande ouverte entre x0 et x1 (fractions de nx), sur tout y."""
    m = np.zeros((nx, ny), dtype=bool)
    m[int(x0 * nx):int(x1 * nx), :] = True
    return m


def mask_rectangle(nx: int, ny: int, x0: float, x1: float,
                   y0: float, y1: float) -> np.ndarray:
    """Rectangle ouvert (fractions de nx, ny)."""
    m = np.zeros((nx, ny), dtype=bool)
    m[int(x0 * nx):int(x1 * nx), int(y0 * ny):int(y1 * ny)] = True
    return m


def mask_bands(nx: int, ny: int, bands) -> np.ndarray:
    """Union de bandes ouvertes en x : [(x0, x1), ...] en fractions."""
    m = np.zeros((nx, ny), dtype=bool)
    for a, b in bands:
        m[int(a * nx):int(b * nx), :] = True
    return m


def invert(mask: np.ndarray) -> np.ndarray:
    return ~mask
