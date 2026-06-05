"""In-memory mesh model — compact numpy arrays, no per-element Python objects.

``MeshData`` is the hand-off type between the parser (:mod:`meshview.io`) and
the geometry/render layers. Nodes are stored once as an ``(N, 3)`` float array;
elements are grouped into per-type blocks whose connectivity is stored as a
2-D integer array of *node indices* (0-based, already resolved from the
original Abaqus node IDs). This keeps memory flat and lets surface extraction
and bounding-box maths run as vectorised numpy operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ElementBlock:
    """A contiguous block of elements that share one Abaqus type."""

    elem_type: str
    # (M, k) int array of node *indices* into MeshData.nodes.
    connectivity: np.ndarray
    # (M,) original Abaqus element IDs, parallel to connectivity rows.
    elem_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(self.connectivity.shape[0])


@dataclass
class MeshData:
    """Parsed mesh: nodes, element blocks, and named sets."""

    # (N, 3) float64 node coordinates.
    nodes: np.ndarray
    # (N,) original Abaqus node IDs, parallel to ``nodes`` rows.
    node_ids: np.ndarray
    element_blocks: list[ElementBlock] = field(default_factory=list)
    # Set name -> array of node indices / element indices.
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    elem_sets: dict[str, np.ndarray] = field(default_factory=dict)
    # Summary of element types that were parsed but not renderable:
    # {type_name: occurrence_count}.
    ignored_elements: dict[str, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return sum(block.count for block in self.element_blocks)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(min_xyz, max_xyz)`` over all nodes (each shape ``(3,)``)."""

        if self.n_nodes == 0:
            raise ValueError("cannot compute bounding box of an empty mesh")
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def bbox_center(self) -> np.ndarray:
        """Geometric centre of the axis-aligned bounding box."""

        lo, hi = self.bounding_box()
        return 0.5 * (lo + hi)

    def bbox_diagonal(self) -> float:
        """Length of the bounding-box diagonal (used to size the view)."""

        lo, hi = self.bounding_box()
        return float(np.linalg.norm(hi - lo))

    def centroid(self) -> np.ndarray:
        """Arithmetic mean of node coordinates, shape ``(3,)``."""

        if self.n_nodes == 0:
            raise ValueError("cannot compute centroid of an empty mesh")
        return self.nodes.mean(axis=0)
