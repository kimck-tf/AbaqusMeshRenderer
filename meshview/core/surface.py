"""Exterior surface extraction and triangulation.

The single most important performance idea in the viewer: never send element
interiors to the GPU. For a volume mesh, an internal face is shared by exactly
two elements, so it appears twice; a face on the outer skin appears once. We
therefore enumerate every element face, key it by its (order-independent) node
set, and keep only the faces that occur once. Shell elements are surfaces
already and pass straight through.

Output is a compact, indexed triangle soup (``Surface``) ready for an indexed
VBO/IBO upload: a unique vertex array, a flat ``uint32`` index array, smooth
per-vertex normals, and a per-triangle group id for set/part colouring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .element_types import SHELL, SOLID, get_topology, triangulate_face
from .mesh import MeshData


@dataclass
class Surface:
    """Triangulated render surface in compact, indexed form."""

    vertices: np.ndarray  # (V, 3) float32 — only nodes used by the surface
    normals: np.ndarray  # (V, 3) float32 — smooth, unit-length
    indices: np.ndarray  # (3T,) uint32 — triangle corner indices into vertices
    tri_group: np.ndarray  # (T,) int32 — group id per triangle (for colouring)
    group_names: list[str]  # group id -> human-readable name (block/set)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.indices.shape[0] // 3)


def _orient_outward(
    face: tuple[int, ...], elem_center: np.ndarray, nodes: np.ndarray
) -> tuple[int, ...]:
    """Reverse ``face`` if its geometric normal points toward ``elem_center``.

    Abaqus face node orderings are not consistently outward-wound, and input
    meshes may even contain inverted elements. Rather than trust the table
    winding, we orient each exterior face by the unambiguous geometric test:
    the outward normal must point *away* from the owning element's centroid.
    """

    pts = nodes[list(face)]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    face_center = pts.mean(axis=0)
    if float(normal @ (face_center - elem_center)) < 0.0:
        return tuple(reversed(face))
    return face


def _collect_faces(mesh: MeshData) -> tuple[list[tuple[int, ...]], list[int]]:
    """Return surviving faces (as node-index tuples) and their group ids.

    Solid faces are de-duplicated so only exterior faces survive; shell
    elements contribute every face. Each surviving solid face is oriented so
    its normal points outward (away from the owning element centroid).
    """

    nodes = mesh.nodes
    # key (sorted node indices) -> [oriented_face, count, elem_center, group_id]
    solid_faces: dict[tuple[int, ...], list] = {}
    shell_faces: list[tuple[int, ...]] = []
    shell_groups: list[int] = []

    for group_id, block in enumerate(mesh.element_blocks):
        topo = get_topology(block.elem_type)
        if topo is None:
            continue  # unsupported types are summarised elsewhere
        conn = block.connectivity
        if topo.family == SHELL:
            for row in conn:
                face = tuple(int(row[i]) for i in range(topo.n_corner))
                shell_faces.append(face)
                shell_groups.append(group_id)
            continue
        if topo.family == SOLID:
            for row in conn:
                corner_idx = [int(row[i]) for i in range(topo.n_corner)]
                elem_center = nodes[corner_idx].mean(axis=0)
                for local in topo.faces:
                    face = tuple(int(row[i]) for i in local)
                    key = tuple(sorted(face))
                    entry = solid_faces.get(key)
                    if entry is None:
                        solid_faces[key] = [face, 1, elem_center, group_id]
                    else:
                        entry[1] += 1

    faces: list[tuple[int, ...]] = []
    groups: list[int] = []
    for oriented, count, elem_center, group_id in solid_faces.values():
        if count == 1:  # exterior
            faces.append(_orient_outward(oriented, elem_center, nodes))
            groups.append(group_id)
    faces.extend(shell_faces)
    groups.extend(shell_groups)
    return faces, groups


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals, unit length.

    ``vertices`` is ``(V, 3)``; ``triangles`` is ``(T, 3)`` of vertex indices.
    Each triangle's geometric normal (magnitude ∝ area) is accumulated onto its
    three vertices, then every vertex normal is normalised.
    """

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    normals = np.zeros_like(verts)

    if tris.size:
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        face_n = np.cross(v1 - v0, v2 - v0)  # length ∝ 2 * area
        # Scatter-add each face normal onto its three vertices.
        for k in range(3):
            np.add.at(normals, tris[:, k], face_n)

    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-12
    normals[nonzero] /= lengths[nonzero, None]
    return normals.astype(np.float32)


def extract_surface(mesh: MeshData) -> Surface:
    """Extract the exterior surface of ``mesh`` as a compact triangle soup."""

    faces, face_groups = _collect_faces(mesh)

    triangles: list[tuple[int, int, int]] = []
    tri_groups: list[int] = []
    for face, group in zip(faces, face_groups, strict=True):
        for tri in triangulate_face(face):
            triangles.append(tri)
            tri_groups.append(group)

    if not triangles:
        return Surface(
            vertices=np.zeros((0, 3), np.float32),
            normals=np.zeros((0, 3), np.float32),
            indices=np.zeros((0,), np.uint32),
            tri_group=np.zeros((0,), np.int32),
            group_names=[b.elem_type for b in mesh.element_blocks],
        )

    tri_arr = np.asarray(triangles, dtype=np.int64)  # (T, 3) node indices

    # Compact: keep only nodes the surface actually uses, remap indices.
    used = np.unique(tri_arr)
    remap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    local_tris = remap[tri_arr]  # (T, 3) into compact vertex array

    vertices = mesh.nodes[used].astype(np.float32)
    normals = vertex_normals(vertices, local_tris)

    return Surface(
        vertices=vertices,
        normals=normals,
        indices=local_tris.reshape(-1).astype(np.uint32),
        tri_group=np.asarray(tri_groups, dtype=np.int32),
        group_names=[b.elem_type for b in mesh.element_blocks],
    )
