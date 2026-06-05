"""Scale tests: surface extraction must stay fast and correct on big grids."""

from __future__ import annotations

import time

import numpy as np

from meshview.core.mesh import ElementBlock, MeshData
from meshview.core.surface import extract_surface


def build_hex_grid(n: int) -> MeshData:
    """An ``n × n × n`` block of C3D8 hexahedra (vectorised construction)."""

    side = n + 1
    # Node coordinates and a 3-D index -> node-index lookup.
    idx = np.arange(side**3).reshape(side, side, side)  # [i, j, k]
    ii, jj, kk = np.meshgrid(
        np.arange(side), np.arange(side), np.arange(side), indexing="ij"
    )
    nodes = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)

    i = np.arange(n)
    bi, bj, bk = np.meshgrid(i, i, i, indexing="ij")
    bi, bj, bk = bi.ravel(), bj.ravel(), bk.ravel()

    def node(di, dj, dk):
        return idx[bi + di, bj + dj, bk + dk]

    conn = np.stack([
        node(0, 0, 0), node(1, 0, 0), node(1, 1, 0), node(0, 1, 0),
        node(0, 0, 1), node(1, 0, 1), node(1, 1, 1), node(0, 1, 1),
    ], axis=1)
    eids = np.arange(1, conn.shape[0] + 1)
    block = ElementBlock("C3D8", conn, eids)
    return MeshData(nodes=nodes, node_ids=np.arange(1, nodes.shape[0] + 1), element_blocks=[block])


def test_grid_surface_counts():
    n = 12
    mesh = build_hex_grid(n)
    surf = extract_surface(mesh)
    # Box skin: 6 faces, each n*n unit quads, 2 triangles per quad.
    assert surf.n_triangles == 6 * n * n * 2
    # Surface vertices = shell of the (n+1)^3 lattice = total minus interior.
    side = n + 1
    interior = (side - 2) ** 3
    assert surf.n_vertices == side**3 - interior


def test_extraction_scales():
    """~30k elements should extract well under a second."""
    n = 32  # 32^3 = 32768 hexes, 35937 nodes
    mesh = build_hex_grid(n)
    t0 = time.perf_counter()
    surf = extract_surface(mesh)
    elapsed = time.perf_counter() - t0
    assert surf.n_triangles == 6 * n * n * 2
    assert elapsed < 2.0, f"extraction too slow: {elapsed:.2f}s"
