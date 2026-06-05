"""Input mapping as a pure ``(state, event) -> state`` reducer.

Keeping interaction logic as a side-effect-free reducer means the entire
control layer — rotate / pan / zoom / fit / select, and preset remapping — can
be unit-tested by injecting synthetic events, with no window or GPU. The
viewport layer is the only place that translates real mouse/key callbacks into
the events defined here.

Default preset is **HyperMesh**: view manipulation is on ``Ctrl + button`` so a
plain left click stays free for picking/selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .arcball import arcball_rotation
from .camera import Camera

# Actions a binding can resolve to.
ROTATE = "rotate"
PAN = "pan"
ZOOM = "zoom"
FIT = "fit"
SELECT = "select"


@dataclass(frozen=True)
class ControlPreset:
    """Maps (button, modifiers) drag combos and the wheel to camera actions."""

    name: str
    # (button, frozenset(modifiers)) -> action
    drag_bindings: dict[tuple[str, frozenset[str]], str]
    wheel_action: str = ZOOM
    # quick-click (no drag) bindings, e.g. Ctrl+Middle = fit
    click_bindings: dict[tuple[str, frozenset[str]], str] = field(default_factory=dict)

    def drag_action(self, button: str, modifiers: frozenset[str]) -> str | None:
        return self.drag_bindings.get((button, frozenset(modifiers)))

    def click_action(self, button: str, modifiers: frozenset[str]) -> str | None:
        return self.click_bindings.get((button, frozenset(modifiers)))


CTRL = frozenset({"ctrl"})
SHIFT = frozenset({"shift"})
NONE: frozenset[str] = frozenset()

HYPERMESH = ControlPreset(
    name="hypermesh",
    drag_bindings={
        ("left", CTRL): ROTATE,
        ("middle", CTRL): ZOOM,
        ("right", CTRL): PAN,
    },
    wheel_action=ZOOM,
    click_bindings={
        ("middle", CTRL): FIT,
        ("left", NONE): SELECT,
    },
)

# Alternative "general" preset: plain left=rotate, shift+left=pan, wheel=zoom.
GENERAL = ControlPreset(
    name="general",
    drag_bindings={
        ("left", NONE): ROTATE,
        ("left", SHIFT): PAN,
        ("middle", NONE): PAN,
    },
    wheel_action=ZOOM,
    click_bindings={},
)

PRESETS = {p.name: p for p in (HYPERMESH, GENERAL)}


@dataclass
class ViewState:
    """Everything the reducer needs: the camera, the model box, and a preset."""

    camera: Camera
    preset: ControlPreset = HYPERMESH
    bbox: tuple = None  # (lo, hi) for fit/reset; set when a mesh is loaded
    home: Camera = None  # camera snapshot used by reset
    last_pick: tuple | None = None  # (x_ndc, y_ndc) of the most recent select


# --- events ---------------------------------------------------------------

@dataclass(frozen=True)
class DragEvent:
    button: str
    modifiers: frozenset[str]
    x0: float
    y0: float  # previous cursor, NDC [-1, 1]
    x1: float
    y1: float  # current cursor, NDC [-1, 1]


@dataclass(frozen=True)
class ScrollEvent:
    steps: float  # +ve = wheel up (zoom in)
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class ClickEvent:
    button: str
    modifiers: frozenset[str]
    x: float
    y: float


@dataclass(frozen=True)
class KeyEvent:
    key: str  # lower-case, e.g. "f", "r"


_ZOOM_PER_STEP = 0.9  # wheel up -> distance *= 0.9 (closer)


def reduce(state: ViewState, event) -> ViewState:
    """Apply ``event`` to ``state``, returning a new :class:`ViewState`."""

    if isinstance(event, DragEvent):
        return _reduce_drag(state, event)
    if isinstance(event, ScrollEvent):
        cam = state.camera.copy()
        cam.zoom(_ZOOM_PER_STEP ** event.steps)
        return replace(state, camera=cam)
    if isinstance(event, ClickEvent):
        action = state.preset.click_action(event.button, event.modifiers)
        if action == FIT and state.bbox is not None:
            cam = state.camera.copy()
            cam.fit_to_bbox(*state.bbox)
            return replace(state, camera=cam)
        if action == SELECT:
            return replace(state, last_pick=(event.x, event.y))
        return state
    if isinstance(event, KeyEvent):
        return _reduce_key(state, event)
    return state


def _reduce_drag(state: ViewState, ev: DragEvent) -> ViewState:
    action = state.preset.drag_action(ev.button, ev.modifiers)
    if action is None:
        return state
    cam = state.camera.copy()
    if action == ROTATE:
        cam.rotate(arcball_rotation((ev.x0, ev.y0), (ev.x1, ev.y1)))
    elif action == PAN:
        cam.pan(ev.x1 - ev.x0, ev.y1 - ev.y0)
    elif action == ZOOM:
        # Vertical drag zooms: drag up (dy>0) zooms in.
        cam.zoom(_ZOOM_PER_STEP ** ((ev.y1 - ev.y0) * 5.0))
    else:
        return state
    return replace(state, camera=cam)


def _reduce_key(state: ViewState, ev: KeyEvent) -> ViewState:
    if ev.key == "f" and state.bbox is not None:
        cam = state.camera.copy()
        cam.fit_to_bbox(*state.bbox)
        return replace(state, camera=cam)
    if ev.key == "r" and state.home is not None:
        return replace(state, camera=state.home.copy())
    return state
