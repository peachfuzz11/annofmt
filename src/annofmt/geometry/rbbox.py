from __future__ import annotations

import math
from dataclasses import dataclass

from annofmt.geometry._base import Geometry
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


@dataclass(frozen=True, slots=True)
class RBBox(Geometry):
    """Immutable rotated bounding box: center, extent and angle.

    ``angle`` is stored canonically in radians within ``[-pi, pi)``. Use
    :meth:`from_degrees` or the ``angle_deg`` property for degree-based
    workflows. A degenerate box (zero width or height) is allowed.

    Every axis-aligned :class:`BBox` is exactly representable as an ``RBBox``
    with zero angle, but not vice versa: use :meth:`as_bbox` for the lossy
    enclosing-box conversion.
    """

    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in (self.cx, self.cy, self.w, self.h, self.angle))
        names = ("cx", "cy", "w", "h", "angle")
        for name, value in zip(names, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if values[2] < 0 or values[3] < 0:
            raise ValueError(f"w and h must be non-negative, got {values[2]}, {values[3]}")
        object.__setattr__(self, "cx", values[0])
        object.__setattr__(self, "cy", values[1])
        object.__setattr__(self, "w", values[2])
        object.__setattr__(self, "h", values[3])
        object.__setattr__(self, "angle", _canonical_angle(values[4]))

    @classmethod
    def from_degrees(cls, cx: float, cy: float, w: float, h: float, degrees: float) -> RBBox:
        """Build from an angle given in degrees."""
        return cls(cx, cy, w, h, math.radians(degrees))

    @property
    def angle_rad(self) -> float:
        return self.angle

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle)

    @property
    def area(self) -> float:
        return self.w * self.h

    def corners(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """The four corner points, counter-clockwise starting from the corner
        at ``(+w/2, +h/2)`` in the local frame."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        half_w = self.w / 2
        half_h = self.h / 2

        def corner(hw: float, hh: float) -> tuple[float, float]:
            return (self.cx + hw * cos_a - hh * sin_a, self.cy + hw * sin_a + hh * cos_a)

        return (corner(half_w, half_h), corner(-half_w, half_h), corner(-half_w, -half_h), corner(half_w, -half_h))

    def rotate(self, d_angle: float, unit: str = "rad") -> RBBox:
        """Rotate by ``d_angle`` (radians by default, or ``unit="deg"``)."""
        if unit not in ("rad", "deg"):
            raise ValueError(f"unit must be 'rad' or 'deg', got {unit!r}")
        delta = math.radians(d_angle) if unit == "deg" else d_angle
        return RBBox(self.cx, self.cy, self.w, self.h, self.angle + delta)

    def as_bbox(self) -> BBox:
        """Lossy enclosing axis-aligned box."""
        cos_a = abs(math.cos(self.angle))
        sin_a = abs(math.sin(self.angle))
        half_w = (self.w * cos_a + self.h * sin_a) / 2
        half_h = (self.w * sin_a + self.h * cos_a) / 2
        return BBox(self.cx - half_w, self.cy - half_h, self.cx + half_w, self.cy + half_h)

    def bounds(self) -> BBox:
        return self.as_bbox()

    def translate(self, dx: float, dy: float) -> RBBox:
        return RBBox(self.cx + dx, self.cy + dy, self.w, self.h, self.angle)

    def scale(self, factor_w: float, factor_h: float) -> RBBox:
        """Scale the extent about the center; rotation is preserved."""
        scaled = BBox(0.0, 0.0, self.w, self.h).scale(factor_w, factor_h)
        return RBBox(self.cx, self.cy, scaled.width, scaled.height, self.angle)

    def iou(self, other: Geometry) -> float:
        """Exact IoU against another ``RBBox`` or any ``BBox`` (a ``BBox`` is
        a rotated box at angle 0)."""
        if isinstance(other, BBox):
            other_box = RBBox(
                other.x_min + other.width / 2,
                other.y_min + other.height / 2,
                other.width,
                other.height,
                0.0,
            )
        elif isinstance(other, RBBox):
            other_box = other
        else:
            raise TypeError(f"RBBox.iou supports RBBox and BBox, got {type(other).__name__}")
        intersection = clip_convex(self.corners(), other_box.corners())
        if not intersection:
            return 0.0
        inter_area = shoelace_area(intersection)
        union = self.area + other_box.area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
