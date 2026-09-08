"""Group-colour palette and the ambient floor that keeps shaded faces readable.

Both are pure numpy — importing :mod:`meshview.render.renderer` does not pull in
moderngl, so these run without an OpenGL context (unlike ``test_render_smoke``).
"""

from __future__ import annotations

import numpy as np
import pytest

from meshview.core.surface import Surface
from meshview.render.renderer import DEFAULT_AMBIENT, _vertex_colors, build_palette

# Rec. 709 relative luminance — how bright a colour reads on screen.
_LUMA = np.array([0.2126, 0.7152, 0.0722])
# Viewport clear colour (see ViewportConfig._bg in meshview/render/viewport.py).
_BACKGROUND = np.array([0.16, 0.17, 0.19])


def _luminance(rgb) -> np.ndarray:
    return np.asarray(rgb) @ _LUMA


@pytest.mark.parametrize("n_groups", [1, 2, 3, 6, 7, 12, 24])
def test_palette_has_one_colour_per_group(n_groups):
    palette = build_palette(n_groups)
    assert palette.shape == (n_groups, 3)
    assert palette.dtype == np.float32
    assert np.unique(palette, axis=0).shape[0] == n_groups


@pytest.mark.parametrize("n_groups", [2, 3, 6, 7, 12])
def test_palette_colours_stay_visually_apart(n_groups):
    palette = build_palette(n_groups).astype(np.float64)
    dist = np.linalg.norm(palette[:, None, :] - palette[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    assert dist.min() > 0.12


def test_palette_stays_within_the_unit_rgb_cube():
    palette = build_palette(9)
    assert palette.min() >= 0.0
    assert palette.max() <= 1.0


def test_group_colours_do_not_repeat_across_blocks():
    # The old fixed six-colour palette wrapped around at the 7th block.
    n_groups = 9
    surface = Surface(
        vertices=np.zeros((3 * n_groups, 3), np.float32),
        normals=np.zeros((3 * n_groups, 3), np.float32),
        indices=np.arange(3 * n_groups, dtype=np.uint32),
        tri_group=np.arange(n_groups, dtype=np.int32),
        group_names=[f"BLOCK{i}" for i in range(n_groups)],
    )
    colors = _vertex_colors(surface, build_palette(len(surface.group_names)))
    assert np.unique(colors, axis=0).shape[0] == n_groups


@pytest.mark.parametrize("n_groups", [1, 4, 8, 16, 24])
def test_shaded_faces_stay_brighter_than_the_viewport_background(n_groups):
    # mesh.frag: intensity = ambient + (1 - ambient) * |dot(n, l)|, so a face
    # turned edge-on to the light is drawn at exactly `ambient` brightness.
    darkest = build_palette(n_groups) * DEFAULT_AMBIENT
    assert _luminance(darkest).min() > _luminance(_BACKGROUND)
