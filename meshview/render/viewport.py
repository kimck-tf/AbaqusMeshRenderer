"""Window/input layer: wires controls -> camera -> renderer via moderngl-window.

This is the only module that knows about windows and live mouse/key callbacks.
It translates real events into the GPU-free :mod:`meshview.camera.controls`
events, reduces the view state, and asks the renderer to draw. Keeping the
translation this thin is what lets the interaction logic be unit-tested without
a window (see ``tests/test_controls.py``).
"""

from __future__ import annotations

import logging

import moderngl
import moderngl_window as mglw

from ..camera.camera import STANDARD_VIEWS, Camera
from ..camera.controls import (
    ClickEvent,
    DragEvent,
    KeyEvent,
    ScrollEvent,
    ViewState,
    reduce,
)
from ..core.surface import extract_surface
from ..io.loader import load_mesh
from .renderer import MeshRenderer

log = logging.getLogger("meshview.render")

_BUTTON_NAMES = {1: "left", 2: "right", 3: "middle"}  # mglw MouseButtons


class ViewportConfig(mglw.WindowConfig):
    """moderngl-window app: HyperMesh-style navigation of a mesh surface."""

    title = "meshview"
    gl_version = (3, 3)
    window_size = (1280, 800)
    aspect_ratio = None  # free aspect; we update the camera on resize
    resizable = True

    # Set by run_viewer() before the window is created.
    initial_mesh_path: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._bg = (0.16, 0.17, 0.19, 1.0)
        self.show_surface = True
        self.show_wireframe = True
        self.renderer: MeshRenderer | None = None

        width, height = self.wnd.size
        cam = Camera(aspect=width / height if height else 1.0)
        self.state = ViewState(camera=cam, home=cam.copy())

        self._press_button: str | None = None
        self._moved = False

        if self.initial_mesh_path:
            self.load(self.initial_mesh_path)

    # --- mesh loading ---
    def load(self, path: str) -> None:
        try:
            mesh = load_mesh(path)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the user
            log.error("failed to load %s: %s", path, exc)
            return
        surface = extract_surface(mesh)
        if self.renderer is not None:
            self.renderer.release()
        self.renderer = MeshRenderer(self.ctx, surface)

        lo, hi = mesh.bounding_box()
        self.state.bbox = (lo, hi)
        self.state.camera.aspect = self._aspect()
        self.state.camera.fit_to_bbox(lo, hi)
        self.state.home = self.state.camera.copy()
        self.wnd.title = f"meshview — {path}"

        log.info(
            "loaded %s: %d nodes, %d elements, %d surface triangles",
            path, mesh.n_nodes, mesh.n_elements, surface.n_triangles,
        )
        if mesh.ignored_elements:
            summary = ", ".join(f"{t}×{n}" for t, n in mesh.ignored_elements.items())
            log.warning("ignored unsupported element types: %s", summary)

    # --- helpers ---
    def _aspect(self) -> float:
        w, h = self.wnd.size
        return w / h if h else 1.0

    def _ndc(self, x: float, y: float) -> tuple[float, float]:
        w, h = self.wnd.size
        return (2.0 * x / w - 1.0, 1.0 - 2.0 * y / h)

    def _mods(self) -> frozenset[str]:
        m = self.wnd.modifiers
        out = set()
        if m.ctrl:
            out.add("ctrl")
        if m.shift:
            out.add("shift")
        if m.alt:
            out.add("alt")
        return frozenset(out)

    def _active_button(self) -> str | None:
        ms = self.wnd.mouse_states
        if ms.left:
            return "left"
        if ms.middle:
            return "middle"
        if ms.right:
            return "right"
        return None

    # --- render ---
    def on_render(self, time: float, frame_time: float) -> None:
        self.ctx.clear(*self._bg)
        if self.renderer is None:
            return
        cam = self.state.camera
        view = cam.view_matrix()
        mvp = cam.projection_matrix() @ view
        self.renderer.render(
            mvp,
            view[:3, :3],
            show_surface=self.show_surface,
            show_wireframe=self.show_wireframe,
        )

    def on_resize(self, width: int, height: int) -> None:
        self.ctx.viewport = (0, 0, width, height)
        if height:
            self.state.camera.aspect = width / height

    # --- input ---
    def on_mouse_press_event(self, x: int, y: int, button: int) -> None:
        self._press_button = _BUTTON_NAMES.get(button)
        self._moved = False

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        button = self._active_button()
        if button is None:
            return
        self._moved = True
        cur = self._ndc(x, y)
        prev = self._ndc(x - dx, y - dy)
        self.state = reduce(
            self.state,
            DragEvent(button, self._mods(), prev[0], prev[1], cur[0], cur[1]),
        )

    def on_mouse_release_event(self, x: int, y: int, button: int) -> None:
        if not self._moved and self._press_button is not None:
            nx, ny = self._ndc(x, y)
            self.state = reduce(
                self.state, ClickEvent(self._press_button, self._mods(), nx, ny)
            )
        self._press_button = None
        self._moved = False

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self.state = reduce(self.state, ScrollEvent(steps=y_offset))

    def on_key_event(self, key, action, modifiers) -> None:
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.F:
            self.state = reduce(self.state, KeyEvent("f"))
        elif key == keys.R:
            self.state = reduce(self.state, KeyEvent("r"))
        elif key == keys.S:
            self.show_surface = not self.show_surface
        elif key == keys.W:
            self.show_wireframe = not self.show_wireframe
        else:
            self._maybe_standard_view(key, keys)

    def _maybe_standard_view(self, key, keys) -> None:
        mapping = {
            keys.NUMPAD_1: "front",
            keys.NUMPAD_2: "back",
            keys.NUMPAD_3: "right",
            keys.NUMPAD_4: "left",
            keys.NUMPAD_5: "top",
            keys.NUMPAD_6: "bottom",
            keys.NUMPAD_7: "iso",
        }
        name = mapping.get(key)
        if name is None:
            return
        cam = self.state.camera
        cam.rotation = STANDARD_VIEWS[name].copy()
        if self.state.bbox is not None:
            cam.fit_to_bbox(*self.state.bbox)

    def on_files_dropped_event(self, x: int, y: int, paths: list[str]) -> None:
        if paths:
            self.load(paths[0])


def run_viewer(mesh_path: str | None = None) -> None:
    """Launch the interactive viewer, optionally loading ``mesh_path`` first."""

    ViewportConfig.initial_mesh_path = mesh_path
    mglw.run_window_config(ViewportConfig, args=[])
