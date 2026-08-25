from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import cast

from annofmt.geometry._base import Geometry
from annofmt.geometry._polygons import min_area_rect, signed_area
from annofmt.geometry.bbox import BBox
from annofmt.geometry.rbbox import RBBox

Index = int
Point2D = tuple[Index, Index]


def _to_index(value: object, name: str) -> Index:
    """Coerce a coordinate to an integer pixel index.

    Accepts anything ``operator.index()`` supports (int, numpy integers, ...)
    and real numbers with an integral value. Everything else is rejected:
    ``Segm`` works on indices only.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer index, got bool {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        pass
    if isinstance(value, Real):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        if value.is_integer():
            return int(value)
    raise ValueError(f"{name} must be an integer pixel index, got {value!r}")


def _rasterize(parts: Sequence[Sequence[Point2D]], height: int, width: int) -> bytearray:
    """Even-odd scanline fill of all parts combined into a row-major grid.

    Pixels are unit squares centered at ``(col + 0.5, row + 0.5)``; parts may
    overlap or contain holes regardless of winding order.
    """
    grid = bytearray(height * width)
    edges: list[tuple[float, float, float, float]] = []
    for part in parts:
        n = len(part)
        for i in range(n):
            x1, y1 = part[i]
            x2, y2 = part[(i + 1) % n]
            if y1 != y2:
                edges.append((float(x1), float(y1), float(x2), float(y2)))
    for row in range(height):
        yc = row + 0.5
        crossings: list[float] = []
        for ex1, ey1, ex2, ey2 in edges:
            if (ey1 <= yc < ey2) or (ey2 <= yc < ey1):
                t = (yc - ey1) / (ey2 - ey1)
                crossings.append(ex1 + t * (ex2 - ex1))
        if not crossings:
            continue
        crossings.sort()
        base = row * width
        for k in range(0, len(crossings) - 1, 2):
            start = max(0, math.ceil(crossings[k] - 0.5))
            stop = min(width, math.ceil(crossings[k + 1] - 0.5))
            for col in range(start, stop):
                grid[base + col] = 1
    return grid


def _runs_from_grid(grid: bytearray, height: int, width: int) -> tuple[Index, ...]:
    """COCO-style column-major run-length encoding.

    Counts alternate zeros/ones starting with zeros; a fully empty mask
    encodes as ``(height * width,)``.
    """
    runs: list[Index] = []
    current_value = 0
    count = 0
    for col in range(width):
        for row in range(height):
            value = grid[row * width + col]
            if value == current_value:
                count += 1
            else:
                runs.append(count)
                current_value = value
                count = 1
    runs.append(count)
    return tuple(runs)


def _grid_from_runs(runs: Sequence[Index], height: int, width: int) -> bytearray:
    """Inverse of :func:`_runs_from_grid`: scatter column-major counts into a
    row-major grid."""
    grid = bytearray(height * width)
    pos = 0
    value = 0
    for count in runs:
        length = _to_index(count, "run length")
        if length < 0:
            raise ValueError(f"run lengths must be non-negative, got {length}")
        if value:
            for k in range(pos, min(pos + length, len(grid))):
                row = k % height
                col = k // height
                grid[row * width + col] = 1
        pos += length
        value ^= 1
    total = sum(_to_index(run, "run length") for run in runs)
    if total != len(grid):
        raise ValueError(f"RLE counts sum to {total}, expected {height * width}")
    return grid


def _trace_loops(grid: bytearray, height: int, width: int) -> tuple[tuple[Point2D, ...], ...]:
    """Extract boundary loops of the filled region as lattice polygons.

    Boundary edges are directed consistently around filled cells, so outer
    boundaries and holes become separately wound loops that compose correctly
    under the even-odd fill convention. Collinear vertices are collapsed.
    """

    def filled(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width and grid[row * width + col] == 1

    outgoing: dict[Point2D, list[Point2D]] = {}
    for row in range(height):
        for col in range(width):
            if not filled(row, col):
                continue
            if not filled(row - 1, col):
                outgoing.setdefault((col, row), []).append((col + 1, row))
            if not filled(row, col + 1):
                outgoing.setdefault((col + 1, row), []).append((col + 1, row + 1))
            if not filled(row + 1, col):
                outgoing.setdefault((col + 1, row + 1), []).append((col, row + 1))
            if not filled(row, col - 1):
                outgoing.setdefault((col, row + 1), []).append((col, row))

    def turn_rank(origin: Point2D, incoming: tuple[int, int], target: Point2D) -> tuple[int, int]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        straightness = dx * incoming[0] + dy * incoming[1]
        cross = incoming[0] * dy - incoming[1] * dx
        return (-straightness, cross)

    loops: list[tuple[Point2D, ...]] = []
    while outgoing:
        start = next(iter(outgoing))
        loop: list[Point2D] = [start]
        current = start
        incoming: tuple[int, int] | None = None
        while True:
            candidates = outgoing.get(current)
            if not candidates:
                break
            if incoming is None or len(candidates) == 1:
                nxt = candidates.pop()
            else:
                direction = incoming
                origin = current
                nxt = max(candidates, key=lambda pt: turn_rank(origin, direction, pt))
                candidates.remove(nxt)
            if not candidates:
                del outgoing[current]
            incoming = (nxt[0] - current[0], nxt[1] - current[1])
            current = nxt
            if current == start:
                break
            loop.append(current)
        simplified: list[Point2D] = []
        m = len(loop)
        for i in range(m):
            prev_pt, pt, next_pt = loop[i - 1], loop[i], loop[(i + 1) % m]
            if (pt[0] - prev_pt[0]) * (next_pt[1] - pt[1]) != (pt[1] - prev_pt[1]) * (next_pt[0] - pt[0]):
                simplified.append(pt)
        if len(simplified) >= 3:
            loops.append(tuple(simplified))
    return tuple(loops)


@dataclass(frozen=True, slots=True)
class Segm(Geometry):
    """Immutable segmentation as polygons over integer pixel indices.

    Coordinates are integer indices only; fractional coordinates are
    rejected. ``parts`` holds one or more rings that combine under an
    even-odd rule, so a ring nested inside another acts as a hole regardless
    of winding. A single flat ring may also be passed directly to the
    constructor.

    Conventions:
        - Polygon vertices sit on the integer lattice.
        - Rasterization (:meth:`to_rle`) treats pixels as unit squares
          centered at ``(col + 0.5, row + 0.5)``.
        - :meth:`as_bbox` returns the enclosing box of the raw indices;
          :meth:`from_bbox` floors minimum and ceils maximum corners, making
          ``Segm.from_bbox(segm.as_bbox())`` stable for lattice-aligned boxes.
    """

    parts: tuple[tuple[Point2D, ...], ...]

    def __init__(self, parts: Iterable[Iterable[Sequence[float]]]) -> None:
        object.__setattr__(self, "parts", self._normalize_parts(parts))

    @staticmethod
    def _normalize_parts(parts: Iterable[Iterable[Sequence[float]]]) -> tuple[tuple[Point2D, ...], ...]:
        materialized: list[tuple[Sequence[float], ...]] = [tuple(part) for part in parts]
        if not materialized:
            return ()
        first_ring = materialized[0]
        if first_ring and isinstance(first_ring[0], Real):
            flat_ring = cast(tuple[Sequence[float], ...], tuple(materialized))
            materialized = [flat_ring]
        normalized: list[tuple[Point2D, ...]] = []
        for ring in materialized:
            points: list[Point2D] = []
            for point in ring:
                if len(point) != 2:
                    raise ValueError(f"each point needs exactly 2 coordinates, got {point!r}")
                points.append((_to_index(point[0], "x"), _to_index(point[1], "y")))
            if len(points) < 3:
                raise ValueError(f"each ring needs at least 3 points, got {len(points)}")
            normalized.append(tuple(points))
        return tuple(normalized)

    @classmethod
    def from_polygon(cls, points: Iterable[Sequence[float]]) -> Segm:
        """Single-ring segmentation."""
        return cls((tuple(points),))

    @classmethod
    def from_bbox(cls, bbox: BBox) -> Segm:
        """Exact rectangle polygon covering the box (floored min corners,
        ceiled max corners)."""
        x_min = math.floor(bbox.x_min)
        y_min = math.floor(bbox.y_min)
        x_max = math.ceil(bbox.x_max)
        y_max = math.ceil(bbox.y_max)
        return cls.from_polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    @classmethod
    def from_rbbox(cls, rbbox: RBBox) -> Segm:
        """Polygon over the four rotated-box corners, rounded to the lattice."""
        corners = rbbox.corners()
        return cls.from_polygon(tuple((int(round(x)), int(round(y))) for x, y in corners))

    @classmethod
    def from_rle(cls, runs: Sequence[Index], height: Index, width: Index) -> Segm:
        """Decode COCO-style column-major run-length counts into polygons."""
        height_i = _to_index(height, "height")
        width_i = _to_index(width, "width")
        if height_i <= 0 or width_i <= 0:
            raise ValueError("height and width must be positive")
        grid = _grid_from_runs(runs, height_i, width_i)
        return cls(_trace_loops(grid, height_i, width_i))

    def to_rle(self, height: Index, width: Index) -> tuple[Index, ...]:
        """Encode as COCO-style column-major run-length counts on a raster of
        the given size (even-odd fill)."""
        height_i = _to_index(height, "height")
        width_i = _to_index(width, "width")
        if height_i <= 0 or width_i <= 0:
            raise ValueError("height and width must be positive")
        grid = _rasterize(self.parts, height_i, width_i)
        return _runs_from_grid(grid, height_i, width_i)

    @property
    def area(self) -> float:
        """Sum of signed ring areas; oppositely wound inner rings subtract.

        Rings produced by :meth:`from_rle` carry consistent winding. For
        hand-built multi-part geometries this equals the covered area only
        when parts do not overlap unless they are wound as holes.
        """
        return abs(sum(signed_area(ring) for ring in self.parts))

    def bounds(self) -> BBox:
        return self.as_bbox()

    def as_bbox(self) -> BBox:
        """Lossy enclosing axis-aligned box of all indices."""
        xs = [point[0] for ring in self.parts for point in ring]
        ys = [point[1] for ring in self.parts for point in ring]
        if not xs:
            raise ValueError("empty segmentation has no bounds")
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def as_rbbox(self) -> RBBox:
        """Lossy minimum-area enclosing rotated box (rotating calipers)."""
        (cx, cy), w, h, angle = min_area_rect([point for ring in self.parts for point in ring])
        return RBBox(cx, cy, w, h, angle)

    def contains_point(self, x: Index, y: Index) -> bool:
        """Even-odd containment test at an integer index point."""
        xi = _to_index(x, "x")
        yi = _to_index(y, "y")
        inside = False
        for ring in self.parts:
            j = len(ring) - 1
            for i in range(len(ring)):
                x1, y1 = ring[j]
                x2, y2 = ring[i]
                if (y1 > yi) != (y2 > yi):
                    t = (yi - y1) / (y2 - y1)
                    if xi < x1 + t * (x2 - x1):
                        inside = not inside
                j = i
        return inside

    def translate(self, dx: float, dy: float) -> Segm:
        dxi = _to_index(dx, "dx")
        dyi = _to_index(dy, "dy")
        return Segm(tuple(tuple((x + dxi, y + dyi) for x, y in ring) for ring in self.parts))

    def scale(self, factor_w: float, factor_h: float) -> Segm:
        """Scale about the center and round back to integer indices."""
        bbox = self.as_bbox()
        scaled = bbox.scale(factor_w, factor_h)
        fx = scaled.width / bbox.width if bbox.width else factor_w
        fy = scaled.height / bbox.height if bbox.height else factor_h
        cx = bbox.x_min + bbox.width / 2
        cy = bbox.y_min + bbox.height / 2
        return Segm(
            tuple(
                tuple(
                    (int(round(cx + (x - cx) * fx)), int(round(cy + (y - cy) * fy)))
                    for x, y in ring
                )
                for ring in self.parts
            )
        )

    def iou(self, other: Geometry) -> float:
        """IoU against another ``Segm``, rasterized on the combined extent
        of both geometries."""
        if not isinstance(other, Segm):
            raise TypeError(f"Segm.iou supports Segm, got {type(other).__name__}; convert explicitly e.g. other.as_bbox()")
        if not self.parts or not other.parts:
            return 0.0
        height = max(point[1] for segm in (self, other) for ring in segm.parts for point in ring) + 1
        width = max(point[0] for segm in (self, other) for ring in segm.parts for point in ring) + 1
        grid_a = _rasterize(self.parts, height, width)
        grid_b = _rasterize(other.parts, height, width)
        intersection = sum(a & b for a, b in zip(grid_a, grid_b, strict=True))
        union = sum(a | b for a, b in zip(grid_a, grid_b, strict=True))
        if union == 0:
            return 0.0
        return intersection / union
