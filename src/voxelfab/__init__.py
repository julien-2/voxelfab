"""voxelfab — moteur de fabrication virtuelle voxel (type SEMulator3D).

API publique :
    VoxelGrid, Materials      -> grille multi-matériaux uint8
    mask_line/rectangle/bands, invert -> masques lithographiques 2D
    DeckRunner, run_deck      -> exécution d'un deck de procédé YAML
    DeckError, load_deck      -> chargement/validation de deck
    Metrology                 -> métrologie virtuelle
    ops                       -> registre des étapes par backend
    layout                    -> import GDS/OASIS (extra [layout])
    export                    -> coupes, vues 3D, VTK, STL
"""

from ._version import __version__
from .deck import DEMO_DECK, DeckError, load_deck
from .grid import Materials, VoxelGrid
from .metro import Metrology
from .masks import invert, mask_bands, mask_line, mask_rectangle
from .ops import available_backends, get_step  # enregistre le backend cpu
from .run import DeckRunner, run_deck

__all__ = [
    "VoxelGrid", "Materials",
    "mask_line", "mask_rectangle", "mask_bands", "invert",
    "DeckRunner", "run_deck", "DeckError", "load_deck", "DEMO_DECK",
    "Metrology", "available_backends", "get_step",
    "__version__",
]
