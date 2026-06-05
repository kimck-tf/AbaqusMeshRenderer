"""Tests for element-type tables and exterior-surface extraction."""

from __future__ import annotations

import numpy as np

from meshview.core.element_types import (
    get_topology,
    is_supported,
    triangulate_face,
)
from meshview.core.mesh import ElementBlock, MeshData
from meshview.core.surface import extract_surface, vertex_normals
from meshview.io.abaqus_parser import parse_abaqus_string
from meshview.io.loader import load_mesh

# --- element type tables ---------------------------------------------------

def test_hex_has_six_quad_faces():
    topo = get_topology("C3D8")
    assert topo.n_corner == 8
    assert len(topo.faces) == 6
    assert all(len(f) == 4 for f in topo.faces)


def test_tet_has_four_tri_faces():
    topo = get_topology("C3D4")
    assert len(topo.faces) == 4
    assert all(len(f) == 3 for f in topo.faces)


def test_second_order_uses_corner_nodes_only():
    # C3D10 (10-node tet) maps to the 4-node tetra topology.
    topo = get_topology("C3D10")
    assert topo.n_corner == 4
    max_index = max(i for face in topo.faces for i in face)
    assert max_index < 4  # mid-side nodes (>=4) never appear in a face


def test_unsupported_type_returns_none():
    assert get_topology("DASHPOTA") is None
    assert not is_supported("DASHPOTA")


def test_triangulate_quad_and_tri():
    assert triangulate_face((0, 1, 2)) == [(0, 1, 2)]
    assert triangulate_face((0, 1, 2, 3)) == [(0, 1, 2), (0, 2, 3)]


# --- surface extraction ----------------------------------------------------

def test_single_hex_surface_counts(fixtures_dir):
    mesh = load_mesh(str(fixtures_dir / "single_hex.inp"))
    surf = extract_surface(mesh)
    # 6 quad faces -> 12 triangles; all 8 nodes used.
    assert surf.n_triangles == 12
    assert surf.n_vertices == 8
    assert surf.indices.shape == (36,)


def test_two_hex_internal_face_removed(fixtures_dir):
    mesh = load_mesh(str(fixtures_dir / "two_hex.inp"))
    surf = extract_surface(mesh)
    # 12 faces total, 1 shared internal face removed -> 10 quads -> 20 tris.
    assert surf.n_triangles == 20
    assert surf.n_vertices == 12


def test_second_order_tet_excludes_midside_from_surface():
    # 10-node tet: corners 1-4, mid-side 5-10. Surface must use only corners.
    coords = {
        1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1),
        5: (0.5, 0, 0), 6: (0.5, 0.5, 0), 7: (0, 0.5, 0),
        8: (0, 0, 0.5), 9: (0.5, 0, 0.5), 10: (0, 0.5, 0.5),
    }
    lines = ["*NODE"]
    lines += [f"{nid},{x},{y},{z}" for nid, (x, y, z) in coords.items()]
    lines += ["*ELEMENT, TYPE=C3D10", "1," + ",".join(str(i) for i in range(1, 11))]
    mesh = parse_abaqus_string("\n".join(lines) + "\n")
    surf = extract_surface(mesh)
    assert surf.n_triangles == 4  # tet skin = 4 triangles
    assert surf.n_vertices == 4   # only corner nodes survive


def test_vertex_normals_unit_length():
    mesh = parse_abaqus_string(
        "*NODE\n"
        "1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n"
        "5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n"
        "*ELEMENT, TYPE=C3D8\n1,1,2,3,4,5,6,7,8\n"
    )
    surf = extract_surface(mesh)
    lengths = np.linalg.norm(surf.normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6)


def test_vertex_normals_point_outward_for_cube():
    mesh = parse_abaqus_string(
        "*NODE\n"
        "1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n"
        "5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n"
        "*ELEMENT, TYPE=C3D8\n1,1,2,3,4,5,6,7,8\n"
    )
    surf = extract_surface(mesh)
    center = surf.vertices.mean(axis=0)
    # For a convex cube the outward normal at each corner points away from
    # the centre: dot((vertex - center), normal) > 0.
    radial = surf.vertices - center
    dots = np.einsum("ij,ij->i", radial, surf.normals)
    assert np.all(dots > 0)


def test_shell_quad_passthrough():
    mesh = parse_abaqus_string(
        "*NODE\n1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n"
        "*ELEMENT, TYPE=S4R\n1,1,2,3,4\n"
    )
    surf = extract_surface(mesh)
    assert surf.n_triangles == 2  # one quad shell -> 2 triangles
    assert surf.n_vertices == 4


def test_direct_vertex_normals_helper():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    n = vertex_normals(verts, tris)
    # Triangle in the z=0 plane wound CCW -> +z normal.
    assert np.allclose(n[0], [0, 0, 1], atol=1e-6)


def test_empty_block_list_surface():
    mesh = MeshData(
        nodes=np.zeros((1, 3)),
        node_ids=np.array([1]),
        element_blocks=[ElementBlock("C3D8", np.zeros((0, 8), np.int64), np.array([], np.int64))],
    )
    surf = extract_surface(mesh)
    assert surf.n_triangles == 0
