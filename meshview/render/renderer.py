"""moderngl renderer: one indexed VBO/IBO upload, shaded + wireframe passes.

The surface is uploaded once as an interleaved, indexed buffer (position,
normal, colour). Every frame only sets uniforms and issues draw calls — no
per-element Python loops, no re-uploads. Wireframe is the same geometry drawn a
second time in line mode with a polygon offset to avoid z-fighting against the
shaded fill.

This module is the boundary where moderngl enters the codebase; everything it
consumes (a :class:`~meshview.core.surface.Surface` and plain matrices) comes
from the GPU-free core/camera layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.surface import Surface

_SHADER_DIR = Path(__file__).parent / "shaders"

# Default qualitative palette (RGB 0..1) used to colour element sets/blocks.
DEFAULT_PALETTE = np.array([
    [0.40, 0.60, 0.85],
    [0.85, 0.55, 0.35],
    [0.45, 0.75, 0.50],
    [0.80, 0.45, 0.55],
    [0.60, 0.50, 0.80],
    [0.75, 0.70, 0.35],
], dtype=np.float32)

WIREFRAME_COLOR = (0.12, 0.12, 0.14)


def _load_shader(name: str) -> str:
    return (_SHADER_DIR / name).read_text(encoding="utf-8")


def _m4_bytes(m: np.ndarray) -> bytes:
    """Pack a 4x4 math matrix as column-major float32 for GLSL."""

    return np.ascontiguousarray(np.asarray(m).T, dtype="f4").tobytes()


def _m3_bytes(m: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(m).T, dtype="f4").tobytes()


def _vertex_colors(surface: Surface, palette: np.ndarray) -> np.ndarray:
    """Per-vertex RGB derived from per-triangle group ids."""

    colors = np.full((surface.n_vertices, 3), 0.7, dtype=np.float32)
    if surface.n_triangles == 0:
        return colors
    tris = surface.indices.reshape(-1, 3)
    per_tri = palette[surface.tri_group % len(palette)]
    for k in range(3):
        colors[tris[:, k]] = per_tri
    return colors


class MeshRenderer:
    """Uploads a surface to the GPU and draws it (shaded + optional wireframe)."""

    def __init__(self, ctx, surface: Surface, palette: np.ndarray | None = None):
        self.ctx = ctx
        self.surface = surface
        self.n_indices = int(surface.indices.shape[0])
        palette = DEFAULT_PALETTE if palette is None else np.asarray(palette, dtype=np.float32)

        self.program = ctx.program(
            vertex_shader=_load_shader("mesh.vert"),
            fragment_shader=_load_shader("mesh.frag"),
        )

        colors = _vertex_colors(surface, palette)
        interleaved = np.hstack([
            surface.vertices.astype("f4"),
            surface.normals.astype("f4"),
            colors,
        ])
        self.vbo = ctx.buffer(interleaved.tobytes())
        self.ibo = ctx.buffer(surface.indices.astype("u4").tobytes()) if self.n_indices else None
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
            index_buffer=self.ibo,
            index_element_size=4,
        )

    def render(
        self,
        mvp: np.ndarray,
        normal_matrix: np.ndarray,
        *,
        light_dir=(0.3, 0.4, 1.0),
        ambient: float = 0.25,
        show_surface: bool = True,
        show_wireframe: bool = True,
    ) -> None:
        if self.n_indices == 0:
            return
        prog = self.program
        prog["mvp"].write(_m4_bytes(mvp))
        prog["normal_matrix"].write(_m3_bytes(normal_matrix))
        prog["light_dir"].value = tuple(light_dir)
        prog["ambient"].value = float(ambient)

        if show_surface:
            prog["use_flat_color"].value = False
            self.ctx.wireframe = False
            self.vao.render()

        if show_wireframe:
            prog["use_flat_color"].value = True
            prog["flat_color"].value = WIREFRAME_COLOR
            # Pull the lines slightly toward the camera to beat z-fighting.
            self.ctx.polygon_offset = -1.0, -1.0
            self.ctx.wireframe = True
            self.vao.render()
            self.ctx.wireframe = False
            self.ctx.polygon_offset = 0.0, 0.0

    def release(self) -> None:
        for obj in (self.vao, self.vbo, self.ibo, self.program):
            if obj is not None:
                obj.release()
