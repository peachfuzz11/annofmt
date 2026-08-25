from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annofmt.geometry.bbox import BBox


class Geometry(ABC):
    """Common interface for all annotation geometries.

    All geometries are immutable value objects: every operation returns a new
    instance and never mutates the receiver. IoU is only defined between two
    values of the same geometry type.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def area(self) -> float:
        """Area covered by the geometry."""
        ...

    @abstractmethod
    def to_bbox(self) -> BBox:
        """Plain axis-aligned :class:`BBox` for this geometry.

        Lossy for rotated or interior-structured geometries.
        """
        ...

    @abstractmethod
    def translate(self, dx: float, dy: float) -> Geometry:
        """Shift the geometry by ``(dx, dy)``."""
        ...

    @abstractmethod
    def scale(self, factor_w: float, factor_h: float) -> Geometry:
        """Scale the geometry about its center."""
        ...

    @abstractmethod
    def iou(self, other: Geometry) -> float:
        """Intersection-over-union with another geometry of the same type.

        Raises :class:`TypeError` for any other combination; convert
        explicitly (e.g. ``other.to_bbox()``) first when a comparison across
        types is really wanted.
        """
        ...
