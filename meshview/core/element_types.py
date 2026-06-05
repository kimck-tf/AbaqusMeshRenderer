"""Abaqus element-type -> local face topology tables.

The renderer never draws raw elements; it draws the *triangulated exterior
surface*.  To get there we need, for every supported Abaqus element type:

* its element *family* — ``"solid"`` (volume; only the outer skin is drawn) or
  ``"shell"`` (already a surface; drawn double-sided as-is);
* the number of *corner* nodes (second-order elements list their corner nodes
  first, then mid-side nodes, which we ignore for rendering);
* the *local face table* — each face as a tuple of local corner-node indices,
  wound so that a well-formed element yields outward-facing normals.

Element types are matched against a table.  Unknown types are reported by name
(see :func:`get_topology` returning ``None``) so the caller can warn instead of
silently dropping geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SOLID = "solid"
SHELL = "shell"


@dataclass(frozen=True)
class Topology:
    """Linear topology shared by one or more Abaqus element types."""

    name: str
    family: str
    n_corner: int
    # Faces as local corner-node indices (0-based). Tri faces have 3 entries,
    # quad faces have 4. Empty for shells (the element itself is the face).
    faces: tuple[tuple[int, ...], ...]


# --- Linear reference topologies -------------------------------------------

_TETRA = Topology(
    name="tetra",
    family=SOLID,
    n_corner=4,
    faces=((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)),
)

_HEXAHEDRON = Topology(
    name="hexahedron",
    family=SOLID,
    n_corner=8,
    faces=(
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ),
)

_WEDGE = Topology(
    name="wedge",
    family=SOLID,
    n_corner=6,
    faces=(
        (0, 1, 2),
        (3, 5, 4),
        (0, 3, 4, 1),
        (1, 4, 5, 2),
        (2, 5, 3, 0),
    ),
)

_PYRAMID = Topology(
    name="pyramid",
    family=SOLID,
    n_corner=5,
    faces=(
        (0, 1, 2, 3),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
)

_TRIANGLE = Topology(name="triangle", family=SHELL, n_corner=3, faces=((0, 1, 2),))
_QUAD = Topology(name="quad", family=SHELL, n_corner=4, faces=((0, 1, 2, 3),))


# --- Abaqus element type -> topology ---------------------------------------
#
# Keys are upper-cased Abaqus type names. Second-order variants map to the same
# linear topology; only the leading ``n_corner`` connectivity entries are used,
# so mid-side nodes never reach a face.

_TYPE_MAP: dict[str, Topology] = {}


def _register(topology: Topology, *type_names: str) -> None:
    for name in type_names:
        _TYPE_MAP[name.upper()] = topology


# Continuum tetrahedra (linear + quadratic)
_register(_TETRA, "C3D4", "C3D4H", "C3D10", "C3D10H", "C3D10M", "C3D10I")
# Continuum hexahedra (linear + quadratic, reduced/incompatible variants)
_register(
    _HEXAHEDRON,
    "C3D8", "C3D8R", "C3D8H", "C3D8I", "C3D8RH",
    "C3D20", "C3D20R", "C3D20H", "C3D20RH",
)
# Continuum wedges / triangular prisms
_register(_WEDGE, "C3D6", "C3D6H", "C3D15", "C3D15H")
# Continuum pyramids
_register(_PYRAMID, "C3D5", "C3D5H", "C3D13")
# Shells / membranes / rigid surfaces — triangles
_register(_TRIANGLE, "S3", "S3R", "STRI3", "S6", "M3D3", "R3D3")
# Shells / membranes / rigid surfaces — quads
_register(_QUAD, "S4", "S4R", "S4RS", "S4R5", "S8", "S8R", "M3D4", "M3D4R", "R3D4")


def get_topology(elem_type: str) -> Topology | None:
    """Return the :class:`Topology` for an Abaqus element type, or ``None``.

    Matching is case-insensitive and tolerant of surrounding whitespace.
    ``None`` signals an unsupported type so the caller can log it by name.
    """

    if elem_type is None:
        return None
    return _TYPE_MAP.get(elem_type.strip().upper())


def is_supported(elem_type: str) -> bool:
    """True if ``elem_type`` maps to a known topology."""

    return get_topology(elem_type) is not None


def triangulate_face(face: Sequence[int]) -> list[tuple[int, int, int]]:
    """Split a polygonal face into triangles using a fan from the first vertex.

    Triangles (3 verts) pass through unchanged; quads (4 verts) split into two.
    """

    n = len(face)
    if n == 3:
        return [(face[0], face[1], face[2])]
    if n == 4:
        return [(face[0], face[1], face[2]), (face[0], face[2], face[3])]
    raise ValueError(f"unsupported face with {n} vertices: {tuple(face)}")
