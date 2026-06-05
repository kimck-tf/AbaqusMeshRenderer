"""GL smoke tests — gated on a real OpenGL context.

These exercise shader compilation and an offscreen render. Where no context is
available (headless CI without EGL/display) every test here is skipped, so the
suite never depends on GPU availability.
"""

from __future__ import annotations

import numpy as np
import pytest

from meshview.camera.camera import Camera
from meshview.core.surface import extract_surface
from meshview.io.loader import load_mesh

pytestmark = pytest.mark.gl


@pytest.fixture
def ctx():
    moderngl = pytest.importorskip("moderngl")
    try:
        context = moderngl.create_standalone_context()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no OpenGL context available: {exc}")
    yield context
    context.release()


def _render_to_fbo(ctx, surface, cam, size=(256, 256)):
    from meshview.render.renderer import MeshRenderer

    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture(size, 4)],
        depth_attachment=ctx.depth_texture(size),
    )
    fbo.use()
    ctx.enable(ctx.DEPTH_TEST)
    bg = (0.16, 0.17, 0.19, 1.0)
    ctx.clear(*bg)
    cam.aspect = size[0] / size[1]
    renderer = MeshRenderer(ctx, surface)
    mvp = cam.projection_matrix() @ cam.view_matrix()
    renderer.render(mvp, cam.view_matrix()[:3, :3])
    data = np.frombuffer(fbo.read(components=4), dtype=np.uint8).reshape(size[1], size[0], 4)
    renderer.release()
    fbo.release()
    return data, bg


def test_shader_compiles(ctx, fixtures_dir):
    from meshview.render.renderer import MeshRenderer

    surface = extract_surface(load_mesh(str(fixtures_dir / "single_hex.inp")))
    renderer = MeshRenderer(ctx, surface)
    assert renderer.program is not None
    renderer.release()


def test_cube_renders_visible_pixels(ctx, fixtures_dir):
    mesh = load_mesh(str(fixtures_dir / "single_hex.inp"))
    surface = extract_surface(mesh)
    cam = Camera()
    cam.fit_to_bbox(*mesh.bounding_box())
    data, bg = _render_to_fbo(ctx, surface, cam)

    bg_u8 = np.array([int(c * 255) for c in bg[:3]], dtype=np.uint8)
    non_bg = np.any(np.abs(data[:, :, :3].astype(int) - bg_u8) > 8, axis=2)
    # A fitted cube should cover a meaningful fraction of the viewport.
    assert non_bg.sum() > 0.05 * data.shape[0] * data.shape[1]
