"""Quaternion helpers and arcball (trackball) rotation.

Quaternions are stored as ``[w, x, y, z]`` float64 arrays. Everything here is
pure numpy so the rotation logic can be unit-tested without a window.
"""

from __future__ import annotations

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return quat_identity()
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix."""

    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_two_vectors(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit vector ``v0`` onto ``v1``."""

    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    d = float(np.dot(v0, v1))
    if d > 1.0 - 1e-9:
        return quat_identity()
    if d < -1.0 + 1e-9:
        # Opposite vectors: rotate 180° about any orthogonal axis.
        axis = np.cross(v0, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v0, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(v0, v1)
    return quat_normalize(np.array([1.0 + d, axis[0], axis[1], axis[2]]))


def _map_to_sphere(x: float, y: float, radius: float = 1.0) -> np.ndarray:
    """Project a screen point in NDC onto the virtual trackball sphere."""

    r2 = x * x + y * y
    rr = radius * radius
    if r2 <= rr / 2.0:
        z = np.sqrt(rr - r2)  # inside: on the sphere
    else:
        z = (rr / 2.0) / np.sqrt(r2)  # outside: on the hyperbola
    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


def arcball_rotation(p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    """Rotation quaternion from dragging screen point ``p0`` to ``p1`` (NDC)."""

    v0 = _map_to_sphere(p0[0], p0[1])
    v1 = _map_to_sphere(p1[0], p1[1])
    return quat_from_two_vectors(v0, v1)
