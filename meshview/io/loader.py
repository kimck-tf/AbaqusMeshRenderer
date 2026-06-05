"""Extension-based parser registry.

``.inp`` and ``.blk`` both map to the same Abaqus parser — a ``.blk`` is an
Abaqus model-definition file, not a distinct format. New formats can be added
by registering another extension here without touching call sites.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from ..core.mesh import MeshData
from .abaqus_parser import parse_abaqus_file

# Extension (lower-case, with dot) -> parser callable taking a path.
_REGISTRY: dict[str, Callable[[str], MeshData]] = {
    ".inp": parse_abaqus_file,
    ".blk": parse_abaqus_file,
}


def supported_extensions() -> tuple[str, ...]:
    """Return the file extensions the loader can dispatch."""

    return tuple(sorted(_REGISTRY))


def register_parser(ext: str, parser: Callable[[str], MeshData]) -> None:
    """Register ``parser`` for files with extension ``ext`` (e.g. ``".inp"``)."""

    _REGISTRY[ext.lower()] = parser


def load_mesh(path: str) -> MeshData:
    """Load ``path`` into a :class:`MeshData`, dispatching on its extension."""

    ext = os.path.splitext(path)[1].lower()
    parser = _REGISTRY.get(ext)
    if parser is None:
        raise ValueError(
            f"unsupported file extension {ext!r}; "
            f"supported: {', '.join(supported_extensions())}"
        )
    return parser(path)
