"""Orbit camera with numpy view/projection matrices and fit-to-view.

The camera orbits a ``target`` point at a given ``distance``; its orientation is
a quaternion. All matrices follow the OpenGL convention (right-handed, looking
down -Z in eye space) so they feed straight into moderngl. No GPU or window
dependency lives here, which keeps the camera fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arcball import quat_identity, quat_multiply, quat_normalize, quat_to_mat3


def translation_matrix(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed view matrix (gluLookAt convention)."""

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective projection (maps to NDC z in [-1, 1])."""

    t = np.tan(fovy / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = 1.0 / (aspect * t)
    m[1, 1] = 1.0 / t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


@dataclass
class Camera:
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 5.0
    rotation: np.ndarray = field(default_factory=quat_identity)
    fovy: float = np.radians(45.0)
    aspect: float = 1.0
    near: float = 0.1
    far: float = 100.0
    min_distance: float = 1e-4

    def copy(self) -> Camera:
        return Camera(
            target=self.target.copy(),
            distance=self.distance,
            rotation=self.rotation.copy(),
            fovy=self.fovy,
            aspect=self.aspect,
            near=self.near,
            far=self.far,
            min_distance=self.min_distance,
        )

    # --- matrices ---
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_mat3(self.rotation)

    def eye(self) -> np.ndarray:
        # Camera sits along the world-space view +Z axis at ``distance``.
        forward_world = self.rotation_matrix().T @ np.array([0.0, 0.0, -1.0])
        return self.target - forward_world * self.distance

    def view_matrix(self) -> np.ndarray:
        r = np.eye(4)
        r[:3, :3] = self.rotation_matrix()
        # View = translate(-distance in z) @ R @ translate(-target)
        return translation_matrix([0, 0, -self.distance]) @ r @ translation_matrix(-self.target)

    def projection_matrix(self) -> np.ndarray:
        return perspective(self.fovy, self.aspect, self.near, self.far)

    # --- interactions (mutate in place) ---
    def rotate(self, delta_quat: np.ndarray) -> None:
        self.rotation = quat_normalize(quat_multiply(delta_quat, self.rotation))

    def zoom(self, factor: float) -> None:
        """Scale distance by ``factor`` (>1 moves away), clamped to be positive."""

        if factor <= 0:
            return
        self.distance = max(self.min_distance, self.distance * factor)

    def pan(self, dx_ndc: float, dy_ndc: float) -> None:
        """Translate the target by a screen-space delta given in NDC units."""

        rot = self.rotation_matrix()
        right_world = rot[0, :3]
        up_world = rot[1, :3]
        half_h = self.distance * np.tan(self.fovy / 2.0)
        world_dx = dx_ndc * half_h * self.aspect
        world_dy = dy_ndc * half_h
        # Dragging right should slide the model right => target moves left.
        self.target = self.target - right_world * world_dx - up_world * world_dy

    def fit_to_bbox(self, lo: np.ndarray, hi: np.ndarray, margin: float = 1.1) -> None:
        """Frame the axis-aligned box ``[lo, hi]`` within the current view."""

        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self.target = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        radius = max(radius, self.min_distance)
        self.distance = margin * radius / np.sin(self.fovy / 2.0)
        self.near = max(self.min_distance, self.distance - 1.5 * radius)
        self.far = self.distance + 1.5 * radius
