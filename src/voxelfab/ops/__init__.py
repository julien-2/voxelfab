"""Backends d'implémentation des étapes de procédé."""

from .base import available_backends, get_step, register  # noqa: F401
from . import cpu  # noqa: F401  (enregistre le backend cpu)
