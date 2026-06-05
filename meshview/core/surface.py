"""Exterior surface extraction and triangulation (vectorised).

The single most important performance idea in the viewer: never send element
interiors to the GPU. For a volume mesh, an internal face is shared by exactly
two elements, so it appears twice; a face on the outer skin appears once. We
enumerate every element face, key it by its (order-independent) node set, and
keep only the faces that occur once. Shell elements are surfaces already and
pass straight through.

The whole pipeline is numpy-vectorised — faces are gathered, sorted, and
de-duplicated with array operations rather than per-element Python loops — so it
scales to hundreds of thousands of elements. Output is a compact, indexed
triangle soup (``Surface``) ready for an indexed VBO/IBO upload: a unique vertex
array, a flat ``uint32`` index array, smooth per-vertex normals, and a
per-triangle group id for set/part colouring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .element_types import SHELL, SOLID, get_topology
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
        for k in range(3):
            np.add.at(normals, tris[:, k], face_n)

    lengths = np.linalg.norm(normals, axis=1)
    nonzero = lengths > 1e-12
    normals[nonzero] /= lengths[nonzero, None]
    return normals.astype(np.float32)


def _gather_solid_faces(mesh: MeshData):
    """Collect candidate solid faces grouped by arity (3 = tri, 4 = quad).

    Returns ``{arity: (faces, groups, centers)}`` where ``faces`` is ``(K, arity)``
    node indices (element winding), ``groups`` is ``(K,)`` block ids, and
    ``centers`` is ``(K, 3)`` owning-element centroids (for outward orientation).
    """

    pools: dict[int, list] = {3: [], 4: []}
    for group_id, block in enumerate(mesh.element_blocks):
        topo = get_topology(block.elem_type)
        if topo is None or topo.family != SOLID:
            continue
        conn = block.connectivity
        m = conn.shape[0]
        if m == 0:
            continue
        centers = mesh.nodes[conn[:, : topo.n_corner]].mean(axis=1)  # (M, 3)
        for local in topo.faces:
            faces = conn[:, list(local)]  # (M, arity)
            arity = faces.shape[1]
            groups = np.full(m, group_id, dtype=np.int64)
            pools[arity].append((faces, groups, centers))

    result = {}
    for arity, parts in pools.items():
        if not parts:
            continue
        faces = np.concatenate([p[0] for p in parts], axis=0)
        groups = np.concatenate([p[1] for p in parts], axis=0)
        centers = np.concatenate([p[2] for p in parts], axis=0)
        result[arity] = (faces, groups, centers)
    return result


def _exterior_faces(faces: np.ndarray, groups: np.ndarray, centers: np.ndarray):
    """Keep faces whose node-set occurs once; return (faces, groups, centers)."""

    keys = np.sort(faces, axis=1)
    _uniq, first_idx, counts = np.unique(
        keys, axis=0, return_index=True, return_counts=True
    )
    keep = first_idx[counts == 1]
    return faces[keep], groups[keep], centers[keep]


def _orient_outward(faces: np.ndarray, centers: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Reverse each face whose geometric normal points toward its element centre.

    Abaqus face orderings are not consistently outward-wound (and meshes may
    contain inverted elements), so we orient by the unambiguous geometric test:
    the outward normal must point away from the owning element's centroid.
    """

    if faces.shape[0] == 0:
        return faces
    p0 = nodes[faces[:, 0]]
    p1 = nodes[faces[:, 1]]
    p2 = nodes[faces[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    face_center = nodes[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", normal, face_center - centers) < 0.0
    oriented = faces.copy()
    oriented[inward] = faces[inward, ::-1]
    return oriented


def _triangulate(faces: np.ndarray, groups: np.ndarray):
    """Fan-triangulate (K, arity) faces -> (T, 3) triangles with group ids."""

    if faces.shape[0] == 0:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
    arity = faces.shape[1]
    if arity == 3:
        return faces.astype(np.int64), groups.astype(np.int64)
    if arity == 4:
        t0 = faces[:, [0, 1, 2]]
        t1 = faces[:, [0, 2, 3]]
        tris = np.concatenate([t0, t1], axis=0)
        grp = np.concatenate([groups, groups])
        return tris.astype(np.int64), grp.astype(np.int64)
    raise ValueError(f"unsupported face arity {arity}")


def _shell_triangles(mesh: MeshData):
    """Triangulate shell elements directly (no de-duplication)."""

    tri_parts, grp_parts = [], []
    for group_id, block in enumerate(mesh.element_blocks):
        topo = get_topology(block.elem_type)
        if topo is None or topo.family != SHELL:
            continue
        conn = block.connectivity
        if conn.shape[0] == 0:
            continue
        faces = conn[:, : topo.n_corner]
        groups = np.full(faces.shape[0], group_id, dtype=np.int64)
        tris, grp = _triangulate(faces, groups)
        tri_parts.append(tris)
        grp_parts.append(grp)
    if not tri_parts:
        return np.zeros((0, 3), np.int64), np.zeros((0,), np.int64)
    return np.concatenate(tri_parts), np.concatenate(grp_parts)


def extract_surface(mesh: MeshData) -> Surface:
    """Extract the exterior surface of ``mesh`` as a compact triangle soup."""

    group_names = [b.elem_type for b in mesh.element_blocks]
    nodes = mesh.nodes

    tri_list, grp_list = [], []

    # Solid exterior faces, per arity, de-duplicated and oriented outward.
    for _arity, (faces, groups, centers) in _gather_solid_faces(mesh).items():
        ext_faces, ext_groups, ext_centers = _exterior_faces(faces, groups, centers)
        oriented = _orient_outward(ext_faces, ext_centers, nodes)
        tris, grp = _triangulate(oriented, ext_groups)
        tri_list.append(tris)
        grp_list.append(grp)

    # Shell elements pass through directly.
    shell_tris, shell_grp = _shell_triangles(mesh)
    if shell_tris.shape[0]:
        tri_list.append(shell_tris)
        grp_list.append(shell_grp)

    if not tri_list or sum(t.shape[0] for t in tri_list) == 0:
        return Surface(
            vertices=np.zeros((0, 3), np.float32),
            normals=np.zeros((0, 3), np.float32),
            indices=np.zeros((0,), np.uint32),
            tri_group=np.zeros((0,), np.int32),
            group_names=group_names,
        )

    tri_arr = np.concatenate(tri_list, axis=0)  # (T, 3) node indices
    tri_groups = np.concatenate(grp_list)

    # Compact: keep only nodes the surface uses, remap indices.
    used = np.unique(tri_arr)
    remap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    local_tris = remap[tri_arr]

    vertices = mesh.nodes[used].astype(np.float32)
    normals = vertex_normals(vertices, local_tris)

    return Surface(
        vertices=vertices,
        normals=normals,
        indices=local_tris.reshape(-1).astype(np.uint32),
        tri_group=tri_groups.astype(np.int32),
        group_names=group_names,
    )
