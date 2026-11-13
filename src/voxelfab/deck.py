"""Deck de procédé YAML : schéma v1, chargement, deck de démo embarqué.

Le deck est l'interface utilisateur du moteur : nm et noms de matériaux en
entrée, jamais d'indices de voxels. La validation d'exécution (étapes,
masques, matériaux) vit dans run.py ; ce module porte le chargement et les
erreurs communes.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class DeckError(Exception):
    """Erreur de deck, avec le contexte (numéro d'étape, paramètre fautif)."""


def load_deck(path: Path) -> dict:
    """Charge un deck YAML ; erreur claire si vide ou mal formé."""
    deck = yaml.safe_load(Path(path).read_text())
    if not isinstance(deck, dict):
        raise DeckError("deck vide ou mal formé")
    return deck


DEMO_DECK = """\
version: 1
grid:      {pitch_nm: 2.0, nx: 120, ny: 60, nz: 64}
materials:
  Si:   {color: '#4d79ff'}
  SiO2: {color: '#c9c9c9'}
  SiN:  {color: '#8bc34a'}
  poly: {color: '#e64a19'}
  W:    {color: '#5d4037'}
steps:
  - {op: substrate, material: Si, thickness_nm: 40}
  - {op: etch, depth_nm: 12, mask: {bands: [[0.0, 0.18], [0.82, 1.0]]}}
  - {op: fill, material: SiO2, to_plane_z: 19}
  - {op: cmp, target_z: 19, stop_on: Si}
  - {op: deposit, material: SiO2, thickness_nm: 4}
  - {op: deposit, material: poly, thickness_nm: 36}
  - {op: etch, depth_nm: 60, materials: [poly],
     mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
  - {op: etch, depth_nm: 4, mode: isotropic, materials: [SiO2],
     mask: {invert: {rect: [0.34, 0.52, 0.08, 0.92]}}}
  - {op: deposit, material: SiN, thickness_nm: 8}
  - {op: etch, depth_nm: 8, mode: directional, materials: [SiN]}
  - {op: deposit, material: Si, thickness_nm: 12, mode: directional, on_materials: [Si]}
  - {op: deposit, material: SiO2, thickness_nm: 50}
  - {op: cmp, target_z: 52}
  - {op: etch, depth_nm: 60, until: Si,
     mask: {rect: [0.18, 0.30, 0.30, 0.70]}}
  - {op: etch, depth_nm: 60, until: Si,
     mask: {rect: [0.70, 0.82, 0.30, 0.70]}}
  - {op: deposit, material: W, thickness_nm: 60, mode: directional}
  - {op: cmp, target_z: 52}
outputs:
  - {type: views, file: mos3d.png}
  - {type: report, file: report.json}
"""
