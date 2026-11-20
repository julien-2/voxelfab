"""Backends d'implémentation des étapes de procédé."""

from .base import available_backends, get_step, register, resolve_backend  # noqa: F401
from . import cpu  # noqa: F401  (enregistre le backend cpu)
from . import gpu  # noqa: F401  (enregistre le backend gpu si taichi présent)
