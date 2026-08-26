"""Axis-aligned bounding box geometry with metadata support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BBox:
    """Immutable axis-aligned bounding box stored as center plus extent.
    
    x and y are the center coordinates. The corner values are exposed
    as derived read-only properties (x_min, y_min, x_max, y_max).
    Degenerate boxes with zero width or height are allowed.
    
    Metadata can be stored in the `meta` dict for tags, labels, params, etc.
    The meta dict is excluded from equality and hash comparisons.
    """

    x: float
    y: float
    w: float
    h: float
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        names = ("x", "y", "w", "h")
        values = tuple(float(v) for v in (self.x, self.y, self.w, self.h))
        for name, value in zip(names, values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if values[2] < 0 or values[3] < 0:
            raise ValueError(f"w and h must be non-negative, got {values[2]}, {values[3]}")
        # Ensure meta is a dict
        if not isinstance(self.meta, dict):
            raise TypeError(f"meta must be a dict, got {type(self.meta).__name__}")
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)

    @property
    def x_min(self) -> float:
        return self.x - self.w / 2

    @property
    def y_min(self) -> float:
        return self.y - self.h / 2

    @property
    def x_max(self) -> float:
        return self.x + self.w / 2

    @property
    def y_max(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    @classmethod
    def from_x1y1wh(cls, x1: float, y1: float, w: float, h: float, meta: dict[str, Any] | None = None) -> BBox:
        """Top-left corner plus extent."""
        return BBox(x1 + w / 2, y1 + h / 2, w, h, meta or {})

    @classmethod
    def from_x1y1x2y2(cls, x1: float, y1: float, x2: float, y2: float, meta: dict[str, Any] | None = None) -> BBox:
        """Two arbitrary corners, normalized."""
        return BBox((x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1), meta or {})

    @classmethod
    def resolve(cls, **kwargs: Any) -> BBox:
        """Build a BBox from any supported keyword format.

        Supported keys: (x_min, y_min, x_max, y_max), (x1, y1, x2, y2),
        center (x, y, w, h) or top-left (x1, y1, w, h).
        Also accepts meta dict.
        """
        meta = kwargs.pop('meta', {})
        if "x_min" in kwargs and "x_max" in kwargs and "y_min" in kwargs and "y_max" in kwargs:
            return cls.resolve(
                x1=kwargs["x_min"],
                y1=kwargs["y_min"],
                x2=kwargs["x_max"],
                y2=kwargs["y_max"],
                meta=meta,
            )
        elif "x1" in kwargs and "x2" in kwargs and "y1" in kwargs and "y2" in kwargs:
            return cls.from_x1y1x2y2(**kwargs, meta=meta)
        elif "x" in kwargs and "y" in kwargs and "w" in kwargs and "h" in kwargs:
            return BBox(**kwargs, meta=meta)
        elif "x1" in kwargs and "y1" in kwargs and "w" in kwargs and "h" in kwargs:
            return cls.from_x1y1wh(**kwargs, meta=meta)
        else:
            raise ValueError(f"Could not format input {kwargs}")

    def to_xywh(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def to_x1x2y1y2(self) -> dict[str, float]:
        return {"x1": self.x_min, "x2": self.x_max, "y1": self.y_min, "y2": self.y_max}

    def to_x1y1x2y2(self) -> dict[str, float]:
        return {"x1": self.x_min, "y1": self.y_min, "x2": self.x_max, "y2": self.y_max}

    def to_x_min_y_min_x_max_y_max(self) -> dict[str, float]:
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max}

    def translate(self, dx: float, dy: float) -> BBox:
        return BBox(self.x + dx, self.y + dy, self.w, self.h, self.meta.copy())

    def scale(self, factor_w: float, factor_h: float) -> BBox:
        """Scale about the center by the given positive factors."""
        if not (math.isfinite(factor_w) and math.isfinite(factor_h)):
            raise ValueError("scale factors must be finite")
        if factor_w <= 0 or factor_h <= 0:
            raise ValueError(f"scale factors must be positive, got {factor_w}, {factor_h}")
        return BBox(self.x, self.y, self.w * factor_w, self.h * factor_h, self.meta.copy())

    def reference(self, other: BBox) -> BBox:
        """Coordinates of this box relative to the top-left of other."""
        return BBox(
            self.x_min - other.x_min + self.w / 2,
            self.y_min - other.y_min + self.h / 2,
            self.w,
            self.h,
            self.meta.copy(),
        )

    def overlaps(self, other: BBox) -> bool:
        """True when this box and other overlap with positive area."""
        return not (
            self.x_max <= other.x_min or self.x_min >= other.x_max or self.y_max <= other.y_min or self.y_min >= other.y_max
        )

    def get_intersection(self, other: BBox) -> BBox | None:
        """The overlapping box, or None when disjoint."""
        if not self.overlaps(other):
            return None
        return BBox(
            (max(self.x_min, other.x_min) + min(self.x_max, other.x_max)) / 2,
            (max(self.y_min, other.y_min) + min(self.y_max, other.y_max)) / 2,
            min(self.x_max, other.x_max) - max(self.x_min, other.x_min),
            min(self.y_max, other.y_max) - max(self.y_min, other.y_min),
        )

    def iou(self, other: BBox) -> float:
        """IoU against another plain BBox."""
        if type(other) is not BBox:
            raise TypeError(f"BBox.iou supports plain BBox only, got {type(other).__name__}")
        intersection = self.get_intersection(other)
        if intersection is None:
            return 0.0
        union = self.area + other.area - intersection.area
        if union <= 0:
            return 0.0
        return intersection.area / union

    def to_bbox(self) -> BBox:
        return self

    def with_meta(self, **kwargs: Any) -> BBox:
        """Return a new BBox with updated metadata."""
        new_meta = self.meta.copy()
        new_meta.update(kwargs)
        return BBox(self.x, self.y, self.w, self.h, new_meta)

    def has_label(self, *labels: str) -> bool:
        """Check if any of the given labels are in metadata."""
        # Support both direct label check and tags list
        if "label" in self.meta:
            if isinstance(self.meta["label"], str):
                if self.meta["label"] in labels:
                    return True
            elif isinstance(self.meta["label"], (list, tuple)):
                if any(l in labels for l in self.meta["label"]):
                    return True
        if "tags" in self.meta:
            tags = self.meta["tags"]
            if isinstance(tags, str):
                if tags in labels:
                    return True
            elif isinstance(tags, (list, tuple)):
                if any(t in labels for t in tags):
                    return True
        return False
