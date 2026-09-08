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

import colorsys
from pathlib import Path

import numpy as np

from ..core.surface import Surface

_SHADER_DIR = Path(__file__).parent / "shaders"

# Qualitative palette generation (RGB 0..1) for element sets/blocks. The
# palette is built to fit the mesh, one colour per group, instead of cycling a
# fixed list — with a fixed list the 7th block wore the 1st block's colour.
_PALETTE_SATURATION = 0.55
_PALETTE_VALUE = 0.88
# Start the hue sweep on the blue the old fixed palette opened with, so a
# single-block mesh keeps the colour it has always had.
_HUE_OFFSET = 0.58
# Rec. 709 luminance weights, and the floor every group colour is lifted to.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_MIN_LUMINANCE = 0.50

# Ambient floor of the shading model. mesh.frag computes
# ``intensity = ambient + (1 - ambient) * |dot(n, l)|``, so a face turned
# edge-on to the light is drawn at exactly this fraction of its group colour.
# At 0.25 such faces came out *darker* than the viewport background and the
# silhouette sank into it; together with _MIN_LUMINANCE this keeps them clearly
# brighter (see tests/test_palette.py).
DEFAULT_AMBIENT = 0.45

WIREFRAME_COLOR = (0.12, 0.12, 0.14)


def build_palette(n_groups: int) -> np.ndarray:
    """Return ``(n, 3)`` float32 RGB — one distinct colour per element group.

    Hues are spread evenly around the colour wheel, so however many groups a
    mesh has, no two of them share a colour and neighbouring hues stay as far
    apart as the group count allows.
    """

    n = max(int(n_groups), 1)
    hues = (np.arange(n, dtype=np.float64) / n + _HUE_OFFSET) % 1.0
    rgb = np.array(
        [colorsys.hsv_to_rgb(h, _PALETTE_SATURATION, _PALETTE_VALUE) for h in hues]
    )
    # Blues and violets are intrinsically dark; blend those toward white until
    # they clear the luminance floor, so no group turns muddy under shading.
    lum = rgb @ _LUMA
    lift = np.clip((_MIN_LUMINANCE - lum) / (1.0 - lum), 0.0, 1.0)
    rgb += (1.0 - rgb) * lift[:, None]
    return rgb.astype(np.float32)


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
        if palette is None:
            palette = build_palette(len(surface.group_names))
        else:
            palette = np.asarray(palette, dtype=np.float32)

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
        ambient: float = DEFAULT_AMBIENT,
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
