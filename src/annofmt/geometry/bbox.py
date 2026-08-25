from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from annofmt.geometry._base import Geometry

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class BBox(Geometry):
    """Immutable axis-aligned bounding box in continuous float coordinates.

    Corners given in any corner order are normalized so ``x_min <= x_max``
    and ``y_min <= y_max``.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in (self.x_min, self.y_min, self.x_max, self.y_max))
        for name, value in zip(("x_min", "y_min", "x_max", "y_max"), values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        object.__setattr__(self, "x_min", min(values[0], values[2]))
        object.__setattr__(self, "x_max", max(values[0], values[2]))
        object.__setattr__(self, "y_min", min(values[1], values[3]))
        object.__setattr__(self, "y_max", max(values[1], values[3]))

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @classmethod
    def from_x1y1wh(cls, x1: float, y1: float, w: float, h: float) -> BBox:
        """Top-left corner plus extent."""
        return cls(x1, y1, x1 + w, y1 + h)

    @classmethod
    def from_x1y1x2y2(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Two arbitrary corners."""
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Center point plus extent."""
        return cls(x - w / 2, y - h / 2, x + w / 2, y + h / 2)

    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Normalized YOLO center-format coordinates (no image size involved).

        The box keeps normalized coordinates; map to pixel space with
        :meth:`scale` when an image size is known.
        """
        return cls.from_xywh(cx, cy, w, h)

    @classmethod
    def resolve(cls, **kwargs: Any) -> BBox:
        """Build a BBox from any supported keyword format.

        Supported keys: ``(x_min, y_min, x_max, y_max)``, ``(x1, y1, x2,
        y2)``, center ``(x, y, w, h)`` or top-left ``(x1, y1, w, h)``.
        """
        if "x_min" in kwargs and "x_max" in kwargs and "y_min" in kwargs and "y_max" in kwargs:
            return cls(**kwargs)
        elif "x1" in kwargs and "x2" in kwargs and "y1" in kwargs and "y2" in kwargs:
            return cls.from_x1y1x2y2(**kwargs)
        elif "x" in kwargs and "y" in kwargs and "w" in kwargs and "h" in kwargs:
            return cls.from_xywh(**kwargs)
        elif "x1" in kwargs and "y1" in kwargs and "w" in kwargs and "h" in kwargs:
            return cls.from_x1y1wh(**kwargs)
        else:
            raise ValueError(f"Could not format input {kwargs}")

    def to_xywh(self) -> dict[str, float]:
        """Center format as a dict."""
        return {"x": self.x_min + self.width / 2, "y": self.y_min + self.height / 2, "w": self.width, "h": self.height}

    def to_yolo(self) -> dict[str, float]:
        """Normalized YOLO center format as a dict (no image size involved)."""
        return {"cx": self.x_min + self.width / 2, "cy": self.y_min + self.height / 2, "w": self.width, "h": self.height}

    def to_x1x2y1y2(self) -> dict[str, float]:
        return {"x1": self.x_min, "x2": self.x_max, "y1": self.y_min, "y2": self.y_max}

    def to_x1y1x2y2(self) -> dict[str, float]:
        return {"x1": self.x_min, "y1": self.y_min, "x2": self.x_max, "y2": self.y_max}

    def to_x_min_y_min_x_max_y_max(self) -> dict[str, float]:
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max}

    def translate(self, dx: float, dy: float) -> BBox:
        return BBox(self.x_min + dx, self.y_min + dy, self.x_max + dx, self.y_max + dy)

    def scale(self, factor_w: float, factor_h: float) -> BBox:
        """Scale about the center by the given positive factors."""
        if not (math.isfinite(factor_w) and math.isfinite(factor_h)):
            raise ValueError("scale factors must be finite")
        if factor_w <= 0 or factor_h <= 0:
            raise ValueError(f"scale factors must be positive, got {factor_w}, {factor_h}")
        cx = self.x_min + self.width / 2
        cy = self.y_min + self.height / 2
        half_w = self.width * factor_w / 2
        half_h = self.height * factor_h / 2
        return BBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def reference(self, other: BBox) -> BBox:
        """Coordinates of this box relative to the top-left of ``other``."""
        return BBox(
            self.x_min - other.x_min,
            self.y_min - other.y_min,
            self.x_max - other.x_min,
            self.y_max - other.y_min,
        )

    def overlaps(self, other: BBox) -> bool:
        """True when the boxes overlap with positive intersection area."""
        return not (
            self.x_max <= other.x_min or self.x_min >= other.x_max or self.y_max <= other.y_min or self.y_min >= other.y_max
        )

    def get_intersection(self, other: BBox) -> BBox | None:
        """The overlapping box, or ``None`` when disjoint."""
        if not self.overlaps(other):
            return None
        return BBox(
            max(self.x_min, other.x_min),
            max(self.y_min, other.y_min),
            min(self.x_max, other.x_max),
            min(self.y_max, other.y_max),
        )

    def iou(self, other: Geometry) -> float:
        if isinstance(other, BBox):
            intersection = self.get_intersection(other)
            if intersection is None:
                return 0.0
            union = self.area + other.area - intersection.area
            if union <= 0:
                return 0.0
            return intersection.area / union
        from annofmt.geometry.rbbox import RBBox

        if isinstance(other, RBBox):
            return other.iou(self)
        raise TypeError(f"BBox.iou supports BBox (and RBBox via delegation), got {type(other).__name__}")

    def bounds(self) -> BBox:
        return self

    def as_bbox(self) -> BBox:
        return self
