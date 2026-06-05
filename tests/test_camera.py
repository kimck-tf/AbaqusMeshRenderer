"""Tests for camera matrices, arcball rotation, and fit-to-view."""

from __future__ import annotations

import numpy as np

from meshview.camera.arcball import (
    arcball_rotation,
    quat_identity,
    quat_multiply,
    quat_to_mat3,
)
from meshview.camera.camera import Camera, look_at, perspective


def test_look_at_maps_target_in_front():
    eye = np.array([0.0, 0.0, 5.0])
    view = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    # Looking straight down -Z: rotation block is identity.
    assert np.allclose(view[:3, :3], np.eye(3), atol=1e-9)
    # Target (origin) lands at z = -5 in eye space.
    origin_eye = view @ np.array([0, 0, 0, 1.0])
    assert np.allclose(origin_eye[:3], [0, 0, -5], atol=1e-9)
    # Eye maps to the origin.
    eye_eye = view @ np.array([*eye, 1.0])
    assert np.allclose(eye_eye[:3], [0, 0, 0], atol=1e-9)


def test_perspective_components():
    fovy, aspect, near, far = np.radians(60.0), 1.5, 0.5, 50.0
    p = perspective(fovy, aspect, near, far)
    t = np.tan(fovy / 2.0)
    assert np.isclose(p[0, 0], 1.0 / (aspect * t))
    assert np.isclose(p[1, 1], 1.0 / t)
    assert np.isclose(p[2, 2], (far + near) / (near - far))
    assert np.isclose(p[2, 3], (2 * far * near) / (near - far))
    assert np.isclose(p[3, 2], -1.0)


def test_arcball_identity_for_no_motion():
    q = arcball_rotation((0.1, 0.2), (0.1, 0.2))
    assert np.allclose(q, quat_identity(), atol=1e-9)


def test_arcball_horizontal_drag_rotates_about_vertical_axis():
    # Dragging horizontally across the centre rotates about the Y (up) axis.
    q = arcball_rotation((-0.5, 0.0), (0.5, 0.0))
    m = quat_to_mat3(q)
    # A Y-axis rotation leaves the Y component of basis vectors essentially put
    # and is a proper rotation (det = 1).
    assert np.isclose(np.linalg.det(m), 1.0, atol=1e-6)
    assert abs(m[1, 0]) < 1e-6 and abs(m[1, 2]) < 1e-6
    assert not np.allclose(m, np.eye(3))  # it actually rotated


def test_quat_multiply_identity():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(quat_multiply(quat_identity(), q), q)


def test_fit_to_bbox_frames_corners_in_ndc():
    cam = Camera(aspect=1.0)
    lo = np.array([-2.0, -1.0, 3.0])
    hi = np.array([4.0, 5.0, 7.0])
    cam.fit_to_bbox(lo, hi)
    proj = cam.projection_matrix()
    view = cam.view_matrix()
    mvp = proj @ view
    corners = np.array([[x, y, z, 1.0]
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    clip = corners @ mvp.T
    ndc = clip[:, :3] / clip[:, 3:4]
    assert np.all(np.abs(ndc) <= 1.0 + 1e-6)


def test_zoom_clamps_to_positive():
    cam = Camera(distance=1.0, min_distance=0.01)
    for _ in range(100):
        cam.zoom(0.5)
    assert cam.distance >= 0.01


def test_zoom_out_increases_distance():
    cam = Camera(distance=2.0)
    cam.zoom(2.0)
    assert np.isclose(cam.distance, 4.0)


def test_pan_moves_target():
    cam = Camera(distance=5.0)
    before = cam.target.copy()
    cam.pan(0.1, 0.0)
    assert not np.allclose(cam.target, before)
    # Panning purely horizontally moves the target along world X (no rotation).
    assert abs(cam.target[1] - before[1]) < 1e-9


def test_eye_distance_matches():
    cam = Camera(distance=7.0)
    assert np.isclose(np.linalg.norm(cam.eye() - cam.target), 7.0)
