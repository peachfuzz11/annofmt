from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

from annofmt.geometry._polygons import clip_convex, shoelace_area
from annofmt.geometry.bbox import BBox

_TWO_PI = 2.0 * math.pi


def _canonical_angle(angle: float) -> float:
    """Wrap an angle in radians into ``[-pi, pi)``."""
    wrapped = math.fmod(angle, _TWO_PI)
    if wrapped < 0:
        wrapped += _TWO_PI
    if wrapped >= math.pi:
        wrapped -= _TWO_PI
    return wrapped


class RBBox(BBox):
    """Immutable rotated bounding box.

    Shares its storage with :class:`BBox` — center ``x``/``y`` and extent
    ``w``/``h`` — and adds the rotation ``a`` in radians, canonically wrapped
    into ``[-pi, pi)``. The corner properties (``x_min`` etc.) are derived
    from the rotated extent, i.e. they describe the enclosing axis-aligned
    box. ``area`` stays ``w * h`` regardless of rotation.
    """

    __slots__ = ("a",)

    a: float

    def __init__(self, x: float, y: float, w: float, h: float, a: float = 0.0) -> None:
        names = ("x", "y", "w", "h", "a")
        values = tuple(float(v) for v in (x, y, w, h, a))
        for name, value in zip(names, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if values[2] < 0 or values[3] < 0:
            raise ValueError(f"w and h must be non-negative, got {values[2]}, {values[3]}")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "a", _canonical_angle(values[4]))

    @classmethod
    def from_degrees(cls, x: float, y: float, w: float, h: float, degrees: float) -> RBBox:
        """Build from an angle given in degrees."""
        return RBBox(x, y, w, h, math.radians(degrees))

    @classmethod
    def from_corners(
        cls,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Sequence[float],
    ) -> RBBox:
        """Build from four corners given in ring order (either winding).

        Raises ``ValueError`` when the points do not form a rectangle.
        """
        points = ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), (float(p3[0]), float(p3[1])))
        cx = sum(p[0] for p in points) / 4
        cy = sum(p[1] for p in points) / 4
        v1x, v1y = points[1][0] - points[0][0], points[1][1] - points[0][1]
        v2x, v2y = points[2][0] - points[1][0], points[2][1] - points[1][1]
        v3x, v3y = points[3][0] - points[2][0], points[3][1] - points[2][1]
        dot = v1x * v2x + v1y * v2y
        norm = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
        scale = max(math.hypot(v1x, v1y), math.hypot(v2x, v2y), 1e-12)
        closes_ring = abs(v3x + v1x) + abs(v3y + v1y) <= 1e-6 * scale
        if norm == 0 or abs(dot) > 1e-6 * norm or not closes_ring:
            raise ValueError(f"corners do not form a rectangle: {points}")
        return RBBox(cx, cy, math.hypot(v1x, v1y), math.hypot(v2x, v2y), math.atan2(v1y, v1x))

    @property
    def angle_rad(self) -> float:
        return self.a

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.a)

    @property
    def x_min(self) -> float:
        return self.x - self._half_enclosed_w

    @property
    def y_min(self) -> float:
        return self.y - self._half_enclosed_h

    @property
    def x_max(self) -> float:
        return self.x + self._half_enclosed_w

    @property
    def y_max(self) -> float:
        return self.y + self._half_enclosed_h

    @property
    def _half_enclosed_w(self) -> float:
        return (abs(self.w * math.cos(self.a)) + abs(self.h * math.sin(self.a))) / 2

    @property
    def _half_enclosed_h(self) -> float:
        return (abs(self.w * math.sin(self.a)) + abs(self.h * math.cos(self.a))) / 2

    def corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """The four corner points, counter-clockwise starting from the corner
        at ``(+w/2, +h/2)`` in the local frame."""
        cos_a = math.cos(self.a)
        sin_a = math.sin(self.a)
        half_w = self.w / 2
        half_h = self.h / 2

        def corner(hw: float, hh: float) -> tuple[float, float]:
            return (self.x + hw * cos_a - hh * sin_a, self.y + hw * sin_a + hh * cos_a)

        return (corner(half_w, half_h), corner(-half_w, half_h), corner(-half_w, -half_h), corner(half_w, -half_h))

    def rotate(self, d_angle: float, unit: str = "rad") -> RBBox:
        """Rotate by ``d_angle`` (radians by default, or ``unit="deg"``)."""
        if unit not in ("rad", "deg"):
            raise ValueError(f"unit must be 'rad' or 'deg', got {unit!r}")
        delta = math.radians(d_angle) if unit == "deg" else d_angle
        return RBBox(self.x, self.y, self.w, self.h, self.a + delta)

    def translate(self, dx: float, dy: float) -> RBBox:
        return RBBox(self.x + dx, self.y + dy, self.w, self.h, self.a)

    def scale(self, factor_w: float, factor_h: float) -> RBBox:
        """Scale the extent about the center; rotation is preserved."""
        scaled = BBox(0.0, 0.0, self.w, self.h).scale(factor_w, factor_h)
        return RBBox(self.x, self.y, scaled.w, scaled.h, self.a)

    def to_bbox(self) -> BBox:
        """Lossy enclosing axis-aligned box."""
        return BBox(self.x, self.y, 2 * self._half_enclosed_w, 2 * self._half_enclosed_h)

    def iou(self, other: object) -> float:
        """Exact IoU against another :class:`RBBox`.

        Raises :class:`TypeError` for any other type, including plain
        :class:`BBox`.
        """
        if type(other) is not RBBox:
            raise TypeError(f"RBBox.iou supports RBBox only, got {type(other).__name__}")
        intersection = clip_convex(self.corners(), other.corners())
        if not intersection:
            return 0.0
        inter_area = shoelace_area(intersection)
        union = self.area + other.area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __eq__(self, other: object) -> bool:
        if type(other) is not RBBox:
            return NotImplemented
        return (self.x, self.y, self.w, self.h, self.a) == (other.x, other.y, other.w, other.h, other.a)

    def __hash__(self) -> int:
        return hash((type(self), self.x, self.y, self.w, self.h, self.a))

    def __repr__(self) -> str:
        return f"RBBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, a={self.a})"
