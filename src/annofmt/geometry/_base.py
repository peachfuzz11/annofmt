from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annofmt.geometry.bbox import BBox


class Geometry(ABC):
    """Common interface for all annotation geometries.

    All geometries are immutable value objects: every operation returns a new
    instance and never mutates the receiver.

    Coordinate conventions:
        - ``BBox`` and ``RBBox`` operate on continuous float coordinates.
        - ``Segm`` operates exclusively on integer pixel indices.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def area(self) -> float:
        """Area covered by the geometry."""
        ...

    @abstractmethod
    def bounds(self) -> BBox:
        """Axis-aligned bounding box of the geometry."""
        ...

    @abstractmethod
    def as_bbox(self) -> BBox:
        """Convert to a plain axis-aligned :class:`BBox`.

        Lossy for geometries that carry rotation or interior structure.
        """
        ...

    @abstractmethod
    def translate(self, dx: float, dy: float) -> Geometry:
        """Shift the geometry by ``(dx, dy)``."""
        ...

    @abstractmethod
    def scale(self, factor_w: float, factor_h: float) -> Geometry:
        """Scale the geometry horizontally and vertically about its center."""
        ...

    @abstractmethod
    def iou(self, other: Geometry) -> float:
        """Intersection-over-union with ``other``.

        Exact within a geometry family (``BBox``<->``BBox``, ``RBBox``<->
        ``RBBox``, ``Segm``<->``Segm``); ``RBBox`` also accepts ``BBox``
        exactly. All other combinations raise :class:`TypeError` because they
        would silently lose information.
        """
        ...
