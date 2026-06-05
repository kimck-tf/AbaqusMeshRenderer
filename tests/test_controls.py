"""Tests for the input-mapping reducer (no window required)."""

from __future__ import annotations

import numpy as np

from meshview.camera.camera import Camera
from meshview.camera.controls import (
    CTRL,
    GENERAL,
    HYPERMESH,
    NONE,
    ClickEvent,
    DragEvent,
    KeyEvent,
    ScrollEvent,
    ViewState,
    reduce,
)


def _state(**kw):
    cam = Camera(distance=5.0)
    bbox = (np.array([-1, -1, -1.0]), np.array([1, 1, 1.0]))
    return ViewState(camera=cam, home=cam.copy(), bbox=bbox, **kw)


def test_ctrl_left_drag_rotates():
    s = _state()
    out = reduce(s, DragEvent("left", CTRL, -0.3, 0.0, 0.3, 0.0))
    assert not np.allclose(out.camera.rotation, s.camera.rotation)


def test_ctrl_right_drag_pans():
    s = _state()
    out = reduce(s, DragEvent("right", CTRL, 0.0, 0.0, 0.2, 0.0))
    assert not np.allclose(out.camera.target, s.camera.target)


def test_ctrl_middle_drag_zooms():
    s = _state()
    out = reduce(s, DragEvent("middle", CTRL, 0.0, -0.2, 0.0, 0.2))
    assert not np.isclose(out.camera.distance, s.camera.distance)


def test_scroll_zooms_in():
    s = _state()
    out = reduce(s, ScrollEvent(steps=1))
    assert out.camera.distance < s.camera.distance


def test_plain_left_click_selects_without_moving_camera():
    s = _state()
    out = reduce(s, ClickEvent("left", NONE, 0.25, -0.1))
    assert out.last_pick == (0.25, -0.1)
    assert np.isclose(out.camera.distance, s.camera.distance)
    assert np.allclose(out.camera.rotation, s.camera.rotation)


def test_ctrl_middle_click_fits():
    s = _state()
    s.camera.distance = 999.0
    out = reduce(s, ClickEvent("middle", CTRL, 0.0, 0.0))
    assert out.camera.distance < 999.0


def test_f_key_fits():
    s = _state()
    s.camera.distance = 999.0
    out = reduce(s, KeyEvent("f"))
    assert out.camera.distance < 999.0


def test_r_key_resets_to_home():
    s = _state()
    moved = reduce(s, DragEvent("left", CTRL, -0.3, 0.0, 0.3, 0.0))
    back = reduce(moved, KeyEvent("r"))
    assert np.allclose(back.camera.rotation, s.home.rotation)


def test_plain_left_drag_does_not_rotate_in_hypermesh():
    s = _state()
    out = reduce(s, DragEvent("left", NONE, -0.3, 0.0, 0.3, 0.0))
    # No Ctrl => not a view manipulation; camera unchanged.
    assert np.allclose(out.camera.rotation, s.camera.rotation)


def test_general_preset_plain_left_drag_rotates():
    cam = Camera(distance=5.0)
    s = ViewState(camera=cam, preset=GENERAL, home=cam.copy())
    out = reduce(s, DragEvent("left", NONE, -0.3, 0.0, 0.3, 0.0))
    assert not np.allclose(out.camera.rotation, s.camera.rotation)


def test_reduce_is_pure_does_not_mutate_input():
    s = _state()
    rot_before = s.camera.rotation.copy()
    reduce(s, DragEvent("left", CTRL, -0.3, 0.0, 0.3, 0.0))
    assert np.allclose(s.camera.rotation, rot_before)  # original untouched


def test_presets_registry():
    assert HYPERMESH.name == "hypermesh"
    assert GENERAL.name == "general"
